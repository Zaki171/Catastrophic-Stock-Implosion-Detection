import findspark
import pandas as pd
findspark.init()

from pyspark.sql import SparkSession
from pyspark import SparkConf

# for shared metastore (shared across all users)
spark = SparkSession.builder.appName("Building dataset").config("hive.metastore.uris", "thrift://amok:9083", conf=SparkConf()).getOrCreate() \

# for local metastore (your private, invidivual database) add the following config to spark session
spark.sql("USE 2023_11_02")

from pyspark.sql.functions import lit,col
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import xgboost as xgb
#from boruta import BorutaPy
#from fredapi import Fred
from sklearn.linear_model import Lasso
from sklearn.model_selection import TimeSeriesSplit
import csv
from pyspark.sql import functions as F
from functools import reduce
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder
from pyspark.ml import Pipeline
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.functions import udf
from pyspark.ml.regression import LinearRegression
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder, CrossValidatorModel
from pyspark.sql.window import Window
from datetime import datetime
import os

curr_dir = os.getcwd()
main_dir = os.path.dirname(curr_dir)



def get_fund_data(imp_df): #This function is used to get price data for a given dataframe consisting of stock IDs
    imp_df.createOrReplaceTempView("temp_table")
    query = f"""SELECT t.fsym_id, p.p_date AS date, p.p_price AS price , splits.p_split_date,
    IF(ISNULL(splits.p_split_factor),1,splits.p_split_factor) AS split_factor, ms.p_com_shs_out
                FROM temp_table t
                LEFT JOIN FF_SEC_COVERAGE c ON c.fsym_id = t.fsym_id
                LEFT JOIN sym_coverage sc ON sc.fsym_id = t.fsym_id
                INNER JOIN fp_basic_prices p ON p.fsym_id = sc.fsym_regional_id
                LEFT JOIN fp_basic_splits AS splits ON splits.p_split_date = p.p_date AND p.fsym_id = splits.fsym_id
                LEFT JOIN fp_basic_dividends AS divs ON divs.p_divs_exdate = p.p_date AND p.fsym_id = divs.fsym_id
                LEFT JOIN (SELECT sf.fsym_id, mp.price_date, sf.p_com_shs_out, sf.p_date AS shares_hist_date
                                        FROM fp_basic_shares_hist AS sf
                                        JOIN (SELECT p2.fsym_id, p2.p_date AS price_date, max(h.p_date) AS max_shares_hist_date
                                                FROM fp_basic_prices AS p2
                                                JOIN fp_basic_shares_hist AS h ON h.p_date <= p2.p_date AND p2.fsym_id = h.fsym_id
                                                GROUP BY p2.fsym_id, p2.p_date)
                                        mp ON mp.fsym_id = sf.fsym_id AND mp.max_shares_hist_date = sf.p_date)
                            ms ON ms.fsym_id = p.fsym_id AND ms.price_date = p.p_date
                WHERE p.p_date >= '2000-01-01'
                ORDER BY t.fsym_id, p.p_date"""

    

    adj = spark.sql(query)
    ###The following code was provided by Banking Science
    div_query = """SELECT fsym_id, p_divs_exdate AS date, SUM(p_divs_pd) AS div FROM fp_basic_dividends 
                    GROUP BY fsym_id, p_divs_exdate
                    ORDER BY fsym_id, p_divs_exdate"""
    df_div = spark.sql(div_query)
    df_div = df_div.withColumn('weekday', F.dayofweek('date'))
    dic = {1:2, 2:0, 3:0, 4:0, 5:0, 6:0, 7:1}
    df_div = df_div.na.replace(dic, subset="weekday")
    df_div = df_div.withColumn('date_adj', F.expr("date_sub(date, weekday)"))
    df = adj.join(df_div, (adj.fsym_id == df_div.fsym_id) & (adj.date == df_div.date_adj), how='left') \
          .select(adj.fsym_id, adj.date, adj.price, df_div.div, adj.split_factor, adj.p_com_shs_out) \
          .na.fill(0, subset='div') \
          .orderBy(F.col('fsym_id').asc(), F.col('date').desc())
    window_spec = Window.partitionBy("fsym_id").orderBy(F.desc("date"))

    df = df.withColumn("cum_split", F.lag(F.exp(F.sum(F.log("split_factor")).over(window_spec))).over(window_spec))
    df = df.na.fill(1, subset='cum_split') # Set cumulative split factor of latest date to 1

    # Split-adjusted price and dividends
    df = df.withColumn("price_split_adj", F.col('price') * F.col('cum_split'))
    df = df.withColumn("div_split_adj", F.col('div') * F.col('cum_split'))

    # Dividend factor
    df = df.withColumn('div_factor', (F.col('price_split_adj') -F.lag('div_split_adj').over(window_spec))/F.col('price_split_adj'))
    df = df.na.fill(1, subset='div_factor') # Set dividend factor of latest date to 1

    # Cumulative dividend factor
    df = df.withColumn("cum_div", F.exp(F.sum(F.log("div_factor")).over(window_spec)))
    df = df.na.fill(1, subset='cum_div') # Set cumulative dividend factor of latest date to 1

    # Price adjusted for splits and dividends
    df = df.withColumn("adj_shs_out", F.col("p_com_shs_out") / F.col("cum_split"))
    
    df = df.withColumn('adj_price', F.col('price_split_adj') * F.col('cum_div'))
    df = df.withColumn('Market_Value', F.col('adj_price') * F.col('adj_shs_out'))
    df = df.withColumn('weekly_return', F.log(df.adj_price / F.lead(df.adj_price).over(window_spec)))
    df = df.orderBy('fsym_id','date')
                         
    df = df.withColumn('year', F.year('date'))
    
    df = df.withColumn('week_of_year', F.weekofyear('date'))

    window_spec = Window.partitionBy('fsym_id', 'year', 'week_of_year').orderBy(col('date').desc())

    df = df.withColumn('row_num', F.row_number().over(window_spec))

    df = df.filter(col('row_num') == 1)
    df = df.select('fsym_id', 'date', 'adj_price', 'Market_Value').orderBy('fsym_id','date')
    return df

def get_fund_data_monthly(imp_df):
    imp_df.createOrReplaceTempView("temp_table")
    query = f"""SELECT t.fsym_id, p.p_date AS date, p.p_price AS price , splits.p_split_date,
    IF(ISNULL(splits.p_split_factor),1,splits.p_split_factor) AS split_factor, ms.p_com_shs_out
                FROM temp_table t
                LEFT JOIN FF_SEC_COVERAGE c ON c.fsym_id = t.fsym_id
                LEFT JOIN sym_coverage sc ON sc.fsym_id = t.fsym_id
                INNER JOIN fp_basic_prices p ON p.fsym_id = sc.fsym_regional_id
                LEFT JOIN fp_basic_splits AS splits ON splits.p_split_date = p.p_date AND p.fsym_id = splits.fsym_id
                LEFT JOIN fp_basic_dividends AS divs ON divs.p_divs_exdate = p.p_date AND p.fsym_id = divs.fsym_id
                LEFT JOIN (SELECT sf.fsym_id, mp.price_date, sf.p_com_shs_out, sf.p_date AS shares_hist_date
                                        FROM fp_basic_shares_hist AS sf
                                        JOIN (SELECT p2.fsym_id, p2.p_date AS price_date, max(h.p_date) AS max_shares_hist_date
                                                FROM fp_basic_prices AS p2
                                                JOIN fp_basic_shares_hist AS h ON h.p_date <= p2.p_date AND p2.fsym_id = h.fsym_id
                                                GROUP BY p2.fsym_id, p2.p_date)
                                        mp ON mp.fsym_id = sf.fsym_id AND mp.max_shares_hist_date = sf.p_date)
                            ms ON ms.fsym_id = p.fsym_id AND ms.price_date = p.p_date
                WHERE p.p_date >= '2000-01-01'
                ORDER BY t.fsym_id, p.p_date"""

    

    adj = spark.sql(query)
    div_query = """SELECT fsym_id, p_divs_exdate AS date, SUM(p_divs_pd) AS div FROM fp_basic_dividends 
                    GROUP BY fsym_id, p_divs_exdate
                    ORDER BY fsym_id, p_divs_exdate"""
    df_div = spark.sql(div_query)
    df_div = df_div.withColumn('weekday', F.dayofweek('date'))
    dic = {1:2, 2:0, 3:0, 4:0, 5:0, 6:0, 7:1}
    df_div = df_div.na.replace(dic, subset="weekday")
    df_div = df_div.withColumn('date_adj', F.expr("date_sub(date, weekday)"))
    df = adj.join(df_div, (adj.fsym_id == df_div.fsym_id) & (adj.date == df_div.date_adj), how='left') \
          .select(adj.fsym_id, adj.date, adj.price, df_div.div, adj.split_factor, adj.p_com_shs_out) \
          .na.fill(0, subset='div') \
          .orderBy(F.col('fsym_id').asc(), F.col('date').desc())
    window_spec = Window.partitionBy("fsym_id").orderBy(F.desc("date"))

    df = df.withColumn("cum_split", F.lag(F.exp(F.sum(F.log("split_factor")).over(window_spec))).over(window_spec))
    df = df.na.fill(1, subset='cum_split') # Set cumulative split factor of latest date to 1

    # Split-adjusted price and dividends
    df = df.withColumn("price_split_adj", F.col('price') * F.col('cum_split'))
    df = df.withColumn("div_split_adj", F.col('div') * F.col('cum_split'))

    # Dividend factor
    df = df.withColumn('div_factor', (F.col('price_split_adj') -F.lag('div_split_adj').over(window_spec))/F.col('price_split_adj'))
    df = df.na.fill(1, subset='div_factor') # Set dividend factor of latest date to 1

    # Cumulative dividend factor
    df = df.withColumn("cum_div", F.exp(F.sum(F.log("div_factor")).over(window_spec)))
    df = df.na.fill(1, subset='cum_div') # Set cumulative dividend factor of latest date to 1

    # Price adjusted for splits and dividends
    df = df.withColumn("adj_shs_out", F.col("p_com_shs_out") / F.col("cum_split"))
    
    df = df.withColumn('adj_price', F.col('price_split_adj') * F.col('cum_div'))
    df = df.withColumn('Market_Value', F.col('adj_price') * F.col('adj_shs_out'))
    df = df.withColumn('weekly_return', F.log(df.adj_price / F.lead(df.adj_price).over(window_spec)))
    df = df.orderBy('fsym_id','date')
                         
    df = df.withColumn('year', F.year('date'))
    df = df.withColumn('month_of_year', F.month('date'))

    # Define the window specification for monthly partitioning
    window_spec = Window.partitionBy('fsym_id', 'year', 'month_of_year').orderBy(F.col('date').desc())

    # Assign row numbers within each monthly partition
    df = df.withColumn('row_num', F.row_number().over(window_spec))

    # Filter to keep only the first row (latest date) within each monthly partition
    df = df.filter(F.col('row_num') == 1)

    # Select the desired columns and order the result
    df = df.select('fsym_id', 'date', 'adj_price', 'Market_Value').orderBy('fsym_id', 'date')
    return df




def get_fund_data_yearly(imp_df):
    imp_df.createOrReplaceTempView("temp_table")
    query = f"""SELECT t.fsym_id, p.p_date AS date, p.p_price AS price , splits.p_split_date,
    IF(ISNULL(splits.p_split_factor),1,splits.p_split_factor) AS split_factor, ms.p_com_shs_out
                FROM temp_table t
                LEFT JOIN FF_SEC_COVERAGE c ON c.fsym_id = t.fsym_id
                LEFT JOIN sym_coverage sc ON sc.fsym_id = t.fsym_id
                INNER JOIN fp_basic_prices p ON p.fsym_id = sc.fsym_regional_id
                LEFT JOIN fp_basic_splits AS splits ON splits.p_split_date = p.p_date AND p.fsym_id = splits.fsym_id
                LEFT JOIN fp_basic_dividends AS divs ON divs.p_divs_exdate = p.p_date AND p.fsym_id = divs.fsym_id
                LEFT JOIN (SELECT sf.fsym_id, mp.price_date, sf.p_com_shs_out, sf.p_date AS shares_hist_date
                                        FROM fp_basic_shares_hist AS sf
                                        JOIN (SELECT p2.fsym_id, p2.p_date AS price_date, max(h.p_date) AS max_shares_hist_date
                                                FROM fp_basic_prices AS p2
                                                JOIN fp_basic_shares_hist AS h ON h.p_date <= p2.p_date AND p2.fsym_id = h.fsym_id
                                                GROUP BY p2.fsym_id, p2.p_date)
                                        mp ON mp.fsym_id = sf.fsym_id AND mp.max_shares_hist_date = sf.p_date)
                            ms ON ms.fsym_id = p.fsym_id AND ms.price_date = p.p_date
                WHERE p.p_date >= '2000-01-01'
                ORDER BY t.fsym_id, p.p_date"""

    

    adj = spark.sql(query)
    div_query = """SELECT fsym_id, p_divs_exdate AS date, SUM(p_divs_pd) AS div FROM fp_basic_dividends 
                    GROUP BY fsym_id, p_divs_exdate
                    ORDER BY fsym_id, p_divs_exdate"""
    df_div = spark.sql(div_query)
    df_div = df_div.withColumn('weekday', F.dayofweek('date'))
    dic = {1:2, 2:0, 3:0, 4:0, 5:0, 6:0, 7:1}
    df_div = df_div.na.replace(dic, subset="weekday")
    df_div = df_div.withColumn('date_adj', F.expr("date_sub(date, weekday)"))
    df = adj.join(df_div, (adj.fsym_id == df_div.fsym_id) & (adj.date == df_div.date_adj), how='left') \
          .select(adj.fsym_id, adj.date, adj.price, df_div.div, adj.split_factor, adj.p_com_shs_out) \
          .na.fill(0, subset='div') \
          .orderBy(F.col('fsym_id').asc(), F.col('date').desc())
    window_spec = Window.partitionBy("fsym_id").orderBy(F.desc("date"))

    df = df.withColumn("cum_split", F.lag(F.exp(F.sum(F.log("split_factor")).over(window_spec))).over(window_spec))
    df = df.na.fill(1, subset='cum_split') # Set cumulative split factor of latest date to 1

    # Split-adjusted price and dividends
    df = df.withColumn("price_split_adj", F.col('price') * F.col('cum_split'))
    df = df.withColumn("div_split_adj", F.col('div') * F.col('cum_split'))

    # Dividend factor
    df = df.withColumn('div_factor', (F.col('price_split_adj') -F.lag('div_split_adj').over(window_spec))/F.col('price_split_adj'))
    df = df.na.fill(1, subset='div_factor') # Set dividend factor of latest date to 1

    # Cumulative dividend factor
    df = df.withColumn("cum_div", F.exp(F.sum(F.log("div_factor")).over(window_spec)))
    df = df.na.fill(1, subset='cum_div') # Set cumulative dividend factor of latest date to 1

    # Price adjusted for splits and dividends
    df = df.withColumn("adj_shs_out", F.col("p_com_shs_out") / F.col("cum_split"))
    
    df = df.withColumn('adj_price', F.col('price_split_adj') * F.col('cum_div'))
    df = df.withColumn('Market_Value', F.col('adj_price') * F.col('adj_shs_out'))
    df = df.withColumn('weekly_return', F.log(df.adj_price / F.lead(df.adj_price).over(window_spec)))
    df = df.orderBy('fsym_id','date')
                         
    df = df.withColumn('year', F.year('date'))

    # Define the window specification for monthly partitioning
    window_spec = Window.partitionBy('fsym_id', 'year').orderBy(F.col('date').desc())

    # Assign row numbers within each monthly partition
    df = df.withColumn('row_num', F.row_number().over(window_spec))

    # Filter to keep only the first row (latest date) within each monthly partition
    df = df.filter(F.col('row_num') < 5)

    # Select the desired columns and order the result
    df = df.select('fsym_id', 'date', 'adj_price', 'Market_Value').orderBy('fsym_id', 'date')
    return df



def get_macro_df():
    macro_df = pd.read_csv(f'{main_dir}/data/macro.csv')
    macro_df['Date'] = pd.to_datetime(macro_df['Date'])
    macro_df['GDP'] = macro_df['GDP'].fillna(method='ffill')
    latest_dates_index = macro_df.groupby(macro_df['Date'].dt.year)['Date'].idxmax()
    macro_df = macro_df.loc[latest_dates_index]
    macro_df['year'] = macro_df['Date'].dt.year
    macro_df = macro_df.drop('Date', axis=1)
    new_columns = {'Unemployment Rate': 'Unemployment_Rate'}
    macro_df.rename(columns=new_columns, inplace=True)
    return macro_df


def get_feature_col_names():
    csv_file_path = 'features.csv'
    data_list = []
    with open(csv_file_path, mode='r') as file:
    # Create a CSV reader object
        reader = csv.reader(file)
        for row in reader:
            data_list.append(row)
    col_list = data_list[0]
    return col_list

def get_not_null_cols(df, null_thresh, table='FF_ADVANCED_DER_AF'): #This function selects columns that have a proportion of null values < null_thresh
    df=spark.createDataFrame(df)
    df.createOrReplaceTempView("temp_table")
    query1 = f"""SELECT t.fsym_id AS fsym_id2, a.*
                FROM temp_table t
                LEFT JOIN {table} a ON t.fsym_id = a.fsym_id
                WHERE a.date > '2000-01-01'
                ORDER BY t.fsym_id, a.date
            """
    #we get all the available dates per stock, so these null values are only within the timeframe available
    q_df = spark.sql(query1)
    q_df = q_df.drop('date', 'adjdate', 'fsym_id2', 'fsym_id')
    num_rows = q_df.count()
    column_types = q_df.dtypes
    good_cols = []
    selected_columns = [F.col(c) for c, c_type in zip(q_df.columns, column_types) if c_type[1] == 'double']
    q_df = q_df.select(selected_columns)
    count_df = q_df.select( [(F.count(F.when(F.isnan(c) | F.col(c).isNull(), c))/num_rows).alias(c) for c in q_df.columns])
    count_dict = count_df.first().asDict()
    # print(count_dict)
    filtered_keys = [key for key, value in count_dict.items() if value <= null_thresh]
    return filtered_keys



    


def get_seq_means(df, all_feats=False):
    orig_df = spark.createDataFrame(df)
    table = "FF_ADVANCED_DER_AF"
    
    
    spark_df = spark.createDataFrame(df)
    spark_df.createOrReplaceTempView("temp_table")
    macro_df= spark.createDataFrame(get_macro_df())
    macro_df.createOrReplaceTempView("macro")
    if all_feats:
        col_names = get_not_null_cols(df)
        col_names += ['GDP', 'Unemployment Rate', 'CPI']
    else:
        col_names = get_feature_col_names()
    col_string = ', '.join('a.' + item for item in col_names)
    q=f"""SELECT t.fsym_id, a.date, {col_string}
                FROM temp_table t
                LEFT JOIN (SELECT * FROM {table} c LEFT JOIN macro m ON m.year = YEAR(c.date)) AS a ON t.fsym_id = a.fsym_id
                ORDER BY t.fsym_id, a.date"""

    
    features_df = spark.sql(q)


    feature_cols = [column for column in features_df.columns if column not in ['fsym_id', 'date']]
    
    grouped_df = features_df.groupBy("fsym_id").agg(
    *[F.mean(F.col(col)).alias(col) for col in feature_cols])

    
    orig_df = orig_df.withColumn('label', F.when(F.isnull('Implosion_Start_Date'), 0).otherwise(1))
    joined_df = grouped_df.join(orig_df.select("fsym_id", "label"), "fsym_id", "inner")
    joined_df=joined_df.orderBy('fsym_id')

    
    return joined_df


def get_tabular_dataset(filename, all_feats=False, imploded_only=False, prediction=False, null_thresh=0.2): #main function for extracting yearly data
    table = "FF_ADVANCED_DER_AF"
    df = pd.read_csv(filename, index_col=False)
    df['Implosion_Start_Date'] = pd.to_datetime(df['Implosion_Start_Date'])
    df['Implosion_End_Date'] = pd.to_datetime(df['Implosion_End_Date'])
    if imploded_only:
        df = df[df['Implosion_Start_Date'].notnull()]
        
    spark_df = spark.createDataFrame(df)
    spark_df.createOrReplaceTempView("temp_table")
    
    if all_feats:
        col_names = get_not_null_cols(df, null_thresh)
        col_names += ['GDP', 'Unemployment_Rate', 'CPI']
        # col_names += get_not_null_cols(df, null_thresh, table='FF_ADVANCED_AF')
    else:
        col_names = get_feature_col_names()
        
    macro_df = spark.createDataFrame(get_macro_df())
    macro_df.createOrReplaceTempView("macro")
    col_string = ', '.join('a.' + item for item in col_names)
    print(col_names)
    q=f"""SELECT t.fsym_id, a.date, {col_string}
            FROM temp_table t INNER JOIN (
                SELECT * FROM {table}
                LEFT JOIN macro m ON m.year = YEAR({table}.date)
                WHERE {table}.date >= "2000-01-01"
            ) a ON t.fsym_id = a.fsym_id
            ORDER BY t.fsym_id, a.date"""
    
    # q=f"""SELECT t.fsym_id, a.date_2 AS date, {col_string}
    #         FROM temp_table t 
    #         INNER JOIN (
    #             SELECT 
    #                 {table}.*,  -- Select all columns from {table}
    #                 m.*,        -- Select all columns from macro
    #                 b.*,        -- Select all columns from FF_ADVANCED_AF
    #                 b.fsym_id as ff_fsym_id,
    #                 b.date as date_2
    #             FROM {table}
    #             LEFT JOIN macro m ON m.year = YEAR({table}.date)
    #             LEFT JOIN FF_ADVANCED_AF b ON {table}.date = b.date AND {table}.fsym_id = b.fsym_id
    #             WHERE {table}.date >= "2000-01-01"
    #         ) a ON t.fsym_id = a.ff_fsym_id
    #         ORDER BY t.fsym_id, a.date_2"""
    
    features_df = spark.sql(q)
  
    joined_df = features_df.join(spark_df.select("fsym_id", "Implosion_Start_Date"), "fsym_id", "inner")
    
    joined_df = joined_df.withColumn('year_date', F.year('date'))
    joined_df = joined_df.withColumn('year_Implosion_Start_Date', F.year('Implosion_Start_Date'))

    # joined_df = joined_df.join(macro_df.select('GDP', 'Unemployment Rate', 'CPI', 'year'), 
    #                            joined_df['year_date'] == macro_df['year'], 'inner')
    # joined_df = joined_df.withColumnRenamed('Unemployment Rate', 'Unemployment_Rate')

    
    if imploded_only:
        joined_df = joined_df.withColumn('label', F.when((F.col('year_date') > F.col('year_Implosion_Start_Date')), 
                                                         2).otherwise(F.when(F.col('year_date')==F.col('year_Implosion_Start_Date'), 1).otherwise(0)))
        joined_df = joined_df.filter((F.col('label') == 0) | (F.col('label') == 1))
    
    else:
        joined_df = joined_df.withColumn('label', 
            F.when(
                (F.col('year_Implosion_Start_Date').isNotNull()) &
                (F.col('year_date') > F.col('year_Implosion_Start_Date')),
                2
            ).otherwise(
                F.when(
                    (F.col('year_Implosion_Start_Date').isNotNull()) &
                    (F.col('year_date') == F.col('year_Implosion_Start_Date')),
                    1
                ).otherwise(0)
            )
        )

        joined_df = joined_df.filter((F.col('label') == 0) | (F.col('label') == 1))
    
    joined_df = joined_df.drop('year_date', 'year_Implosion_Start_Date', 'Implosion_Start_Date', 'year')
    
    joined_df=joined_df.orderBy('fsym_id', 'date')

    if prediction:
        ws = Window.partitionBy('fsym_id').orderBy(F.col('date').desc())
        joined_df = joined_df.withColumn('label', F.lag(F.col('label')).over(ws))
        joined_df = joined_df.filter(F.col('label').isNotNull())
    return joined_df



def get_tabular_dataset_qf(filename, all_feats=False, imploded_only=False, prediction=False, null_thresh=0.2): #function for extracting monthly data
    table = "FF_ADVANCED_DER_QF"
    df = pd.read_csv(filename, index_col=False)
    df['Implosion_Start_Date'] = pd.to_datetime(df['Implosion_Start_Date'])
    df['Implosion_End_Date'] = pd.to_datetime(df['Implosion_End_Date'])
    if imploded_only:
        df = df[df['Implosion_Start_Date'].notnull()]
        
    spark_df = spark.createDataFrame(df)
    spark_df.createOrReplaceTempView("temp_table")
    
    if all_feats:
        col_names = get_not_null_cols(df, null_thresh, table="FF_ADVANCED_DER_QF")
        col_names += ['GDP', 'Unemployment_Rate', 'CPI']
        # col_names = get_not_null_cols(df, null_thresh, table="FF_ADVANCED_AF")
    else:
        col_names = get_feature_col_names()
        
    macro_df = spark.createDataFrame(get_macro_df())
    macro_df.createOrReplaceTempView("macro")
    col_string = ', '.join('a.' + item for item in col_names)
    print(col_names)
    q=f"""SELECT t.fsym_id, a.date, {col_string}
            FROM temp_table t INNER JOIN (
                SELECT * FROM {table}
                LEFT JOIN macro m ON m.year = YEAR({table}.date)
                WHERE {table}.date >= "2000-01-01"
            ) a ON t.fsym_id = a.fsym_id
            ORDER BY t.fsym_id, a.date"""
    
    
#     q=f"""SELECT t.fsym_id, a.date_2 AS date, {col_string}
#             FROM temp_table t 
#             INNER JOIN (
#                 SELECT 
#                     {table}.*,  -- Select all columns from {table}
#                     m.*,        -- Select all columns from macro
#                     b.*,        -- Select all columns from FF_ADVANCED_AF
#                     b.fsym_id as ff_fsym_id,
#                     b.date as date_2
#                 FROM {table}
#                 LEFT JOIN macro m ON m.year = YEAR({table}.date)
#                 LEFT JOIN FF_ADVANCED_AF b ON {table}.date = b.date AND {table}.fsym_id = b.fsym_id
#                 WHERE {table}.date >= "2000-01-01"
#             ) a ON t.fsym_id = a.ff_fsym_id
#             ORDER BY t.fsym_id, a.date_2"""
    
    features_df = spark.sql(q)
    price_df = get_fund_data_monthly(spark_df)
    big_df = price_df.join(features_df, ["fsym_id" , "date"], "left")
     
    big_df = big_df.join(spark_df.select("fsym_id", "Implosion_Start_Date"), "fsym_id", "inner")
    big_df=big_df.orderBy('fsym_id', 'date')
    
    big_df = big_df.withColumn('label', 
            F.when(
                (F.col('Implosion_Start_Date').isNotNull()) &
                (F.year('date') > F.year('Implosion_Start_Date')) |
                ((F.year('date') == F.year('Implosion_Start_Date')) & (F.month('date') > F.month('Implosion_Start_Date'))),
                2
            ).otherwise(
                F.when(
                    (F.col('Implosion_Start_Date').isNotNull()) &
                    (F.year('date') == F.year('Implosion_Start_Date')) & 
                    (F.month('date') == F.month('Implosion_Start_Date')),
                    1
                ).otherwise(0)
            )
        )

    big_df =big_df.filter((F.col('label') == 0) | (F.col('label') == 1))
 
    
    return big_df


