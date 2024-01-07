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
from pyspark.ml.feature import VectorAssembler
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

def get_fund_data(imp_df):
    imp_df.createOrReplaceTempView("temp_table")
    query = f"""SELECT t.fsym_id, p.p_date AS date, p.p_price AS price , splits.p_split_date,
    IF(ISNULL(splits.p_split_factor),1,splits.p_split_factor) AS split_factor
                FROM temp_table t
                LEFT JOIN FF_SEC_COVERAGE c ON c.fsym_id = t.fsym_id
                LEFT JOIN sym_coverage sc ON sc.fsym_id = t.fsym_id
                INNER JOIN fp_basic_prices p ON p.fsym_id = sc.fsym_regional_id
                LEFT JOIN fp_basic_splits AS splits ON splits.p_split_date = p.p_date AND p.fsym_id = splits.fsym_id
                LEFT JOIN fp_basic_dividends AS divs ON divs.p_divs_exdate = p.p_date AND p.fsym_id = divs.fsym_id
                WHERE p.p_date >= '2001-01-01'
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
          .select(adj.fsym_id, adj.date, adj.price, df_div.div, adj.split_factor) \
          .na.fill(0, subset='div') \
          .orderBy(F.col('fsym_id').asc(), F.col('date').desc())
    window_spec = Window.partitionBy("fsym_id").orderBy(F.desc("date"))

    df = df.withColumn("cum_split", F.lag(F.exp(F.sum(F.log("split_factor")).over(window_spec))).over(window_spec))
    df = df.na.fill(1, subset='cum_split') # Set cumulative split factor of latest date to 1

    # Split-adjusted price and dividends
    df = df.withColumn("price_split_adj", F.col('price') * F.col('cum_split'))
    df = df.withColumn("div_split_adj", F.col('div') * F.col('cum_split'))

    # Dividend factor
    df = df.withColumn('div_factor', (F.col('price_split_adj') - F.lag('div_split_adj').over(window_spec))/F.col('price_split_adj'))
    df = df.na.fill(1, subset='div_factor') # Set dividend factor of latest date to 1

    # Cumulative dividend factor
    df = df.withColumn("cum_div", F.exp(F.sum(F.log("div_factor")).over(window_spec)))
    df = df.na.fill(1, subset='cum_div') # Set cumulative dividend factor of latest date to 1

    # Price adjusted for splits and dividends
    df = df.withColumn('adj_price', F.col('price_split_adj') * F.col('cum_div'))
    df = df.orderBy('fsym_id','date')
                         
    df = df.withColumn('year', F.year('date'))
    
    df = df.withColumn('week_of_year', F.weekofyear('date'))

    window_spec = Window.partitionBy('fsym_id', 'year', 'week_of_year').orderBy(col('date').desc())

    df = df.withColumn('row_num', F.row_number().over(window_spec))

    df = df.filter(col('row_num') == 1)
    df = df.select('fsym_id', 'date', 'adj_price').orderBy('fsym_id','date')
    return df


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

def get_not_null_cols(df, table='FF_ADVANCED_DER_AF'):
    df=spark.createDataFrame(df)
    df.createOrReplaceTempView("temp_table")
    query1 = f"""SELECT t.fsym_id, a.*
                FROM temp_table t
                LEFT JOIN {table} a ON t.fsym_id = a.fsym_id
                ORDER BY t.fsym_id, a.date
            """
    #we get all the available dates per stock, so these null values are only within the timeframe available
    q_df = spark.sql(query1)
    column_types = q_df.dtypes

    null_pcts = [F.sum(F.col(col_name).isNull().count()) / F.count("*") for col_name in q_df.columns]

    columns_to_drop = [col_name for col_name, null_pct in zip(q_df.columns, null_pcts) if null_pct > 0.2]

    q_df = q_df.drop(*columns_to_drop)

    cols = q_df.columns

    return cols
    

def get_full_series_stocks(imp_df_price):
    if os.path.exists('stocks_with_data_since_2001.csv'):
        print("File found")
        fsym_ids_2001 = []
        with open('stocks_with_data_since_2001.csv', mode='r') as file:
            reader = csv.reader(file)
            for row in reader:
                fsym_ids_2001.append(row)
        fsym_ids_2001 = fsym_ids_2001[0]

    else:
        
        price_data = get_fund_data(spark.createDataFrame(imp_df_price))

        window_spec = Window.partitionBy('fsym_id').orderBy(col('p_date'))

        price_data = price_data.withColumn('row_num', F.row_number().over(window_spec))

        price_data = price_data.filter(col('row_num') == 1)

        start_dates = price_data.groupBy('year').count().orderBy('year')
        df_2001 = price_data.filter(col('year') == 2001) #get the stocks that have data dating back to 2001
        fsym_ids_2001 = [df_2001.select('fsym_id').distinct().rdd.flatMap(lambda x: x).collect()]

        csv_file_path = "stocks_with_data_since_2001.csv"
        with open(csv_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            for row in fsym_ids_2001:
                writer.writerow(row)
        fsym_ids_2001 = fsym_ids_2001[0]
        
    
    return fsym_ids_2001


def get_features_all_stocks_seq(df, all_feats=False):
    orig_df = spark.createDataFrame(df)
    table = "FF_ADVANCED_DER_AF"
    def generate_dates():
        start_date = datetime(2001, 1, 1)
        end_date = datetime(2022, 12, 31)
        current_date = start_date

        while current_date <= end_date:
            yield current_date
            current_date += pd.DateOffset(years=1) #generating placeholder dates
    
    df['date_for_year'] = df['fsym_id'].map({k: list(generate_dates()) for k, g in df.groupby('fsym_id')})
    df=df.explode('date_for_year')
    df['date_for_year']= pd.to_datetime(df['date_for_year'])
    df['year'] = df['date_for_year'].dt.year
    
    
    spark_df = spark.createDataFrame(df)
    spark_df.createOrReplaceTempView("temp_table")
    if all_feats:
        col_names = get_not_null_cols(df)
    else:
        col_names = get_feature_col_names()
    col_string = ', '.join('a.' + item for item in col_names)
    q=f"""SELECT t.fsym_id, t.year, a.date, {col_string}
                FROM temp_table t
                LEFT JOIN {table} a ON t.fsym_id = a.fsym_id AND t.year = YEAR(a.date)
                LEFT JOIN FF_BASIC_AF b ON b.fsym_id = t.fsym_id and t.year = YEAR(b.date)
                ORDER BY t.fsym_id, t.year"""

    
    features_df = spark.sql(q)
    features_df = features_df.withColumn("non_null_2001", F.when((F.col("year") == 2001) & (F.col("date").isNotNull()), 1).otherwise(0))
    
    ws = Window.partitionBy("fsym_id")

    features_df = features_df.withColumn("group_non_null_2001", F.sum("non_null_2001").over(ws))

    features_df = features_df.filter((F.col("group_non_null_2001") > 0))

    features_df = features_df.drop("non_null_2001", "group_non_null_2001")
    feature_cols = [column for column in features_df.columns if column not in ['fsym_id', 'date', 'year']]
    sequences = []

    # Group by year and count nulls for each specified column
#     features_df.orderBy('fsym_id','year').show(200)
#     null_counts_per_year = features_df.groupBy("year").agg(
#         *[F.sum(F.when(F.col(column).isNull(), 1).otherwise(0)).alias(f"{column}_null_count") for column in feature_cols]
#     )
#     null_counts_per_year.orderBy('year').show(100)
    

#     grouped_df = features_df.groupBy("fsym_id").agg(
#         *[F.collect_list(col).alias(col) for col in feature_cols]) #creating lists per cell
    
    #How to fill gaps?
#     row_count_23_values = grouped_df.filter(
#         *[F.size(F.col(col)) == 23 for col in feature_cols]
#         ).count()

#     print(f"Number of rows with 23 values in all columns: {row_count_23_values}")
    
#     row_count_23_values = grouped_df.filter(
#         *[F.size(F.col(col)) > 15 for col in feature_cols]
#         ).count()

#     print(f"Number of rows with more than 15 values in all columns: {row_count_23_values}")
    
#     length_freq_info = {}
#     for col_name in feature_cols:
#         length_freq_info[col_name] = (
#             grouped_df.select(F.size(col(col_name)).alias("length"))
#             .groupBy("length")
#             .count()
#             .orderBy("length")
#             .collect()
#         )

#     # Print or use the length and frequency information
#     for col_name, info in length_freq_info.items():
#         print(f"Column: {col_name}")
#         for row in info:
#             print(f"Length: {row['length']}, Frequency: {row['count']}")
#         print("\n")
    
    #PADDING WITH 0s
    # grouped_df_padded = grouped_df.select("fsym_id",
    #     *[F.expr(f"IF(size({col}) < 23, concat({col}, array_repeat(0, 23 - size({col}))), {col})").alias(col) for col 
    #       in feature_cols])
    
    #forward filling
    window_spec = Window.partitionBy('fsym_id').orderBy('year')
    for c in feature_cols:
        features_df = features_df.withColumn(
            c, F.last(c, ignorenulls=True).over(window_spec)
        )
        
    # features_df.orderBy('fsym_id','year').show(200)
    # null_counts_per_year = features_df.groupBy("year").agg(
    #     *[F.sum(F.when(F.col(column).isNull(), 1).otherwise(0)).alias(f"{column}_null_count") for column in feature_cols]
    # )
    # null_counts_per_year.orderBy('year').show(100)
    # features_df.orderBy('year', 'fsym_id').show(100)
    
    #SCALING
    window_spec2 = Window.partitionBy('year').orderBy('year')
    for c in feature_cols:
        features_df = features_df.withColumn(c, ((F.col(c) - F.mean(F.col(c)).over(window_spec2)) /
                                            F.stddev(F.col(c)).over(window_spec2)))

    grouped_df = features_df.groupBy("fsym_id").agg(
        *[F.collect_list(col).alias(col) for col in feature_cols]) #creating lists per cell
    
    #PADDING VALUES WITH 0S
    grouped_df_padded = grouped_df.select("fsym_id",
        *[F.expr(f"IF(size({col}) < 22, concat({col}, array_repeat(0, 22 - size({col}))), {col})").alias(col) for col 
          in feature_cols])
    
    orig_df = orig_df.withColumn('label', F.when(F.isnull('Implosion_Start_Date'), 0).otherwise(1))
    joined_df = grouped_df_padded.join(orig_df.select("fsym_id", "label"), "fsym_id", "inner")
    joined_df=joined_df.orderBy('fsym_id')

    
    return joined_df





def get_tabular_dataset(all_feats=False, imploded_only=False):
    table = "FF_ADVANCED_DER_AF"
    df = pd.read_csv('imploded_stocks_price.csv', index_col=False)
    if imploded_only:
        df = df[df['Implosion_Start_Date'].notnull()]
        
    spark_df = spark.createDataFrame(df)
    spark_df.createOrReplaceTempView("temp_table")
    
    if all_feats:
        col_names = get_not_null_cols(df)
    else:
        col_names = get_feature_col_names()
        
        
    col_string = ', '.join('a.' + item for item in col_names)
    q=f"""SELECT t.fsym_id, a.date, {col_string}
                FROM temp_table t
                LEFT JOIN {table} a ON t.fsym_id = a.fsym_id
                WHERE a.date >= "2001-01-01"
                ORDER BY t.fsym_id, a.date"""
    features_df = spark.sql(q)
  
    joined_df = features_df.join(spark_df.select("fsym_id", "Implosion_Start_Date"), "fsym_id", "inner")
    
    joined_df = joined_df.withColumn('year_date', F.year('date'))
    joined_df = joined_df.withColumn('year_Implosion_Start_Date', F.year('Implosion_Start_Date'))
    
    joined_df = joined_df.withColumn('label', F.when(F.col('year_date') == F.col('year_Implosion_Start_Date'), 1).otherwise(0))
    
    joined_df = joined_df.drop('year_date', 'year_Implosion_Start_Date', 'Implosion_Start_Date')
    
    joined_df=joined_df.orderBy('fsym_id', 'date')
    
    return joined_df
    
    
    
#     feature_cols = [col for col in features_df.columns if col not in ['fsym_id', 'date']]
#     sequences = []
    

#     grouped_df = features_df.groupBy("fsym_id").agg(
#         *[F.collect_list(col).alias(col) for col in feature_cols]) #creating lists per cell
    
#     #How to fill gaps?
    
#     grouped_df_padded = grouped_df.select("fsym_id",
#         *[F.expr(f"IF(size({col}) < 23, concat({col}, array_repeat(0, 23 - size({col}))), {col})").alias(col) for col 
#           in feature_cols]) 

    
#     spark_df = spark_df.withColumn('label', F.when(F.isnan('Implosion_Start_Date'), 0).otherwise(1))
#     joined_df = grouped_df_padded.join(spark_df.select("fsym_id", "label"), "fsym_id", "inner")
#     joined_df=joined_df.orderBy('fsym_id')

    
#     return joined_df
