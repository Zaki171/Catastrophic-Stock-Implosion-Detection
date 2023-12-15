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

def get_features_all_stocks():
    table = "FF_ADVANCED_DER_AF"
    df = pd.read_csv('imploded_stocks_price.csv', index_col=False)
    spark_df = spark.createDataFrame(df)
    spark_df.createOrReplaceTempView("temp_table")
    col_names = get_feature_col_names()
    col_string = ', '.join('a.' + item for item in col_names)
    q=f"""SELECT t.fsym_id, a.date, {col_string}
                FROM temp_table t
                LEFT JOIN {table} a ON t.fsym_id = a.fsym_id
                WHERE a.date >= "2000-01-01"
                ORDER BY t.fsym_id, a.date"""
    features_df = spark.sql(q)
    feature_cols = [col for col in features_df.columns if col not in ['fsym_id', 'date']]
    sequences = []
    

    grouped_df = features_df.groupBy("fsym_id").agg(
        *[F.collect_list(col).alias(col) for col in feature_cols]) #creating lists per cell
    
    #How to fill gaps?
    
    grouped_df_padded = grouped_df.select("fsym_id",
        *[F.expr(f"IF(size({col}) < 23, concat({col}, array_repeat(0, 23 - size({col}))), {col})").alias(col) for col 
          in feature_cols]) 

    
    spark_df = spark_df.withColumn('label', F.when(F.isnan('Implosion_Start_Date'), 0).otherwise(1))
    joined_df = grouped_df_padded.join(spark_df.select("fsym_id", "label"), "fsym_id", "inner")
    joined_df=joined_df.orderBy('fsym_id')

    
    return joined_df
