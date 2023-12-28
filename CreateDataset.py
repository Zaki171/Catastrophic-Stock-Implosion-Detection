import findspark
import pandas as pd
findspark.init()

from pyspark.sql import SparkSession
from pyspark import SparkConf

# for shared metastore (shared across all users)
spark = SparkSession.builder.appName("Building dataset").config("hive.metastore.uris", "thrift://bialobog:9083", conf=SparkConf()).getOrCreate() \

# for local metastore (your private, invidivual database) add the following config to spark session
spark.sql("USE 2023_04_01")

import pyspark.pandas as ps
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
    query2 = f"""SELECT t.fsym_id, p.p_date, p.p_price, splits.p_split_date, splits.p_split_factor, 
                divs.p_divs_exdate, divs.p_divs_s_pd, divs.p_divs_pd, ms.p_com_shs_out 
                FROM temp_table t
                LEFT JOIN sym_ticker_region s ON s.fsym_id = t.fsym_id
                LEFT JOIN FF_SEC_COVERAGE c ON c.fsym_id = s.fsym_id
                LEFT JOIN sym_coverage sc ON sc.fsym_id = s.fsym_id
                INNER JOIN fp_basic_prices p ON p.fsym_id = sc.fsym_regional_id
                LEFT JOIN fp_basic_dividends divs ON divs.p_divs_exdate = p.p_date AND p.fsym_id = divs.fsym_id
                LEFT JOIN fp_basic_splits AS splits ON splits.p_split_date = p.p_date AND p.fsym_id = splits.fsym_id
                LEFT JOIN (SELECT sf.fsym_id, mp.price_date, sf.p_com_shs_out, sf.p_date AS shares_hist_date
                                        FROM fp_basic_shares_hist AS sf
                                        JOIN (SELECT p2.fsym_id, p2.p_date AS price_date, max(h.p_date) AS max_shares_hist_date
                                                FROM fp_basic_prices AS p2
                                                JOIN fp_basic_shares_hist AS h ON h.p_date <= p2.p_date AND p2.fsym_id = h.fsym_id
                                                GROUP BY p2.fsym_id, p2.p_date)
                                        mp ON mp.fsym_id = sf.fsym_id AND mp.max_shares_hist_date = sf.p_date)
                            ms ON ms.fsym_id = p.fsym_id AND ms.price_date = p.p_date
                WHERE p.p_date >= '2001-01-01'
                ORDER BY s.fsym_id, p.p_date
                """
    adj = spark.sql(query2)
    imp_df = imp_df.toPandas()
    adj = adj.withColumn("temp_cum_split_factor", F.when(adj.p_date==adj.p_split_date, F.lit(adj.p_split_factor)).otherwise(F.lit(1.0)))
    adj = adj.withColumn("cum_split_factor", F.lit(0.0))

    window_spec = Window.partitionBy('fsym_id').orderBy(F.desc('p_date'))

    adj = adj.withColumn('cum_split_factor_no_lag', 
                        F.exp(F.sum(F.log('temp_cum_split_factor')).over(window_spec)))

    adj = adj.withColumn('cum_split_factor', 
                        F.when(F.row_number().over(window_spec) == 1, 1.0)
                        .otherwise(F.lag('cum_split_factor_no_lag', default=1.0).over(window_spec)))
    
    adj = adj.withColumn('split_adj_price', adj.p_price * adj.cum_split_factor)
    
    adj = adj.withColumn('Market_Value', col('split_adj_price') * col('p_com_shs_out'))
    
    adj = adj.withColumn('year', F.year('p_date'))
    
    adj =adj.withColumn('week_of_year', F.weekofyear('p_date'))

    window_spec = Window.partitionBy('fsym_id', 'year', 'week_of_year').orderBy(col('p_date').desc())

    adj = adj.withColumn('row_num', F.row_number().over(window_spec))

    adj = adj.filter(col('row_num') == 1).orderBy('p_date')
    adj = adj.drop('temp_cum_split_factor', 'cum_split_factor', 'cum_split_factor_no_lag', 'row_num', 'p_split_date', 'p_split_factor')
    return adj


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

    # Collect column names that are not of type float
    columns_to_drop = [col_name for col_name, col_type in column_types if col_type != 'double']

    # Drop columns that are not of type float
    q_df = q_df.drop(*columns_to_drop)

    q_df = ps.DataFrame(q_df)
    null_pcts = q_df.isnull().sum()/len(q_df)
    cols = null_pcts[null_pcts <= 0.2].index.tolist()
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


def get_features_all_stocks_seq(df):
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
    col_names = get_feature_col_names()
    col_string = ', '.join('a.' + item for item in col_names)
    q=f"""SELECT t.fsym_id, t.year, a.date, {col_string}
                FROM temp_table t
                LEFT JOIN {table} a ON t.fsym_id = a.fsym_id AND t.year = YEAR(a.date)
                LEFT JOIN FF_BASIC_AF b ON b.fsym_id = t.fsym_id and t.year = YEAR(b.date)
                ORDER BY t.fsym_id, t.year"""
    features_df = spark.sql(q)
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
    
    orig_df = orig_df.withColumn('label', F.when(F.isnan('Implosion_Start_Date'), 0).otherwise(1))
    joined_df = grouped_df_padded.join(orig_df.select("fsym_id", "label"), "fsym_id", "inner")
    joined_df=joined_df.orderBy('fsym_id')

    
    return joined_df





def get_tabular_dataset(all_feats=False):
    table = "FF_ADVANCED_DER_AF"
    df = pd.read_csv('imploded_stocks_price.csv', index_col=False)
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
