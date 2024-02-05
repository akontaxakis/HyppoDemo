# pip install pyspark or use your favorite way to set Spark Home, here we use findspark
#set path to SPARK_HOME
PATH = "C:/Users/adoko/data/"
# Create Spark session and configure according to your environment
from pyspark.sql import SparkSession

if __name__ == '__main__':

    spark = SparkSession.builder \
        .appName("Local PySpark") \
        .config('spark.master', 'local') \
        .getOrCreate()



    df_test_raw = spark.read.parquet(PATH + 'testUndersampled.snappy.parquet')
    df_test_raw.printSchema()

    df_train_raw = spark.read.parquet(PATH + 'trainUndersampled.snappy.parquet')
    df_train_raw.printSchema()

    from pyspark.ml.functions import vector_to_array

    df_test = ( df_test_raw
                 .withColumn('HLF_input', vector_to_array('HLF_input'))
                 .withColumn('encoded_label', vector_to_array('encoded_label'))
                 .select('HLF_input', 'encoded_label')
              )

    df_test.printSchema()

    df_train = ( df_train_raw
                   .withColumn('HLF_input', vector_to_array('HLF_input'))
                   .withColumn('encoded_label', vector_to_array('encoded_label'))
                   .select('HLF_input', 'encoded_label')
               )

    df_train.printSchema()