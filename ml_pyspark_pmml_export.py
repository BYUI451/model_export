# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC #### Packages and functions

# COMMAND ----------

import pandas as pd
import numpy as np
import pyecharts as pye

from plotnine import *
from functools import reduce
from pyspark.sql import Window, DataFrame

import pyspark.sql.functions as F
import pyspark.sql.types as T

# COMMAND ----------

# MAGIC %run ./functions

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Base check out data
# MAGIC 
# MAGIC - Built using the `wide_tables()` function in `functions`.
# MAGIC - Not sure how this will need to be used for future data.
# MAGIC - See `multiple_book_challenge` for the data compilation that is needed to go into `wide_tables()`

# COMMAND ----------

spark.sql("use library_g25")
dat = spark.table('wcheckouts_checkout')
display(dat)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Compile all months (model data)
# MAGIC 
# MAGIC Final `library_g25.model_months_all` is what I will use for my model.

# COMMAND ----------

m2 = build_month(2, dat)
m3 = build_month(3, dat)
m4 = build_month(4, dat)
m5 = build_month(5, dat)
m6 = build_month(6, dat)
m7 = build_month(7, dat)
m8 = build_month(8, dat)
m9 = build_month(9, dat)
m10 = build_month(10, dat)
dfs = [m2,m3,m4, m5, m6, m7, m8, m9, m10]
# https://stackoverflow.com/questions/37612622/spark-unionall-multiple-dataframes
# https://www.geeksforgeeks.org/reduce-in-python/
# https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.sql.DataFrame.unionAll.html#:~:text=unionAll,-DataFrame.&text=Return%20a%20new%20DataFrame%20containing,to%20UNION%20ALL%20in%20SQL.
# https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.sql.DataFrame.unionByName.html
df = reduce(DataFrame.unionAll, dfs)
df.write.mode('overwrite').saveAsTable("library_g25.model_months_all")
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Machine Learning
# MAGIC 
# MAGIC > Using standard 5-fold cross-validation, no practical effect of the dependencies within the data could be found, regarding whether the final error is under- or overestimated. On the contrary, last block evaluation tends to yield less robust error measures than cross-validation and blocked cross-validation. [ref](https://www.sciencedirect.com/science/article/abs/pii/S0020025511006773)
# MAGIC 
# MAGIC > However, for non-stationary time-series, they recommend instead using a variation on Hold-Out, called Rep-Holdout. In Rep-Holdout, a point `a` is chosen in the time-series `Y` to mark the beginning of the testing data. The point `a` is determined to be within a window. [ref](https://www.semanticscholar.org/paper/Evaluating-time-series-forecasting-models%3A-An-study-Cerqueira-Torgo/1ef6b9c734ceb775bc6055af93918b5db5bf190d)
# MAGIC 
# MAGIC - [code](https://github.com/vcerqueira/performance_estimation)
# MAGIC - [Good ML on time series post](https://towardsdatascience.com/time-series-machine-learning-regression-framework-9ea33929009a)
# MAGIC - [Good time series and machine learning slides](http://di.ulb.ac.be/map/gbonte/ftp/time_ser.pdf)
# MAGIC - [My other notes](https://github.com/BYUI451/spark_guide/tree/main/MLlib)

# COMMAND ----------

# The pyspark.ml methods used in this example.
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, VectorIndexer
# https://spark.apache.org/docs/latest/ml-classification-regression.html
from pyspark.ml.regression import GBTRegressor, DecisionTreeRegressor, RandomForestRegressor, FMRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator


# COMMAND ----------

# MAGIC %run ./functions

# COMMAND ----------

print(df.columns)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ##### Define ML columns

# COMMAND ----------

# column names
dropcols = ['BibNum']
categoricalCols = []
numericCols = ['number_months_available', 'total_checkouts', 'previous_quarter_checkouts', 'previous_month_checkouts', 'current_month_checkouts', 'total_collection_central',  'previous_quarter_collection_central', 'previous_month_collection_central', 'current_month_collection_central', 'total_collection_other', 'previous_quarter_collection_other',
 'previous_month_collection_other', 'current_month_collection_other']
targetCol = 'target' 
featureInputs = [c + "OHE" for c in categoricalCols] + numericCols # we are adding the one hot encoder label (OHE) to the column names because that is done with our econder transform
print(featureInputs)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ##### Spit data

# COMMAND ----------

# https://spark.apache.org/docs/3.1.1/api/python/reference/api/pyspark.sql.DataFrame.randomSplit.html
trainDF, testDF = df.randomSplit([0.70, 0.30], seed=36)
print(trainDF.cache().count()) # Cache because accessing training data multiple times
print(testDF.count())

# COMMAND ----------

# https://stackoverflow.com/questions/47637760/stratified-sampling-with-pyspark
# Taking 70% of each `number_months_available`
train = df.sampleBy("number_months_available", fractions={2: 0.7, 3: 0.7, 4: 0.7, 5: 0.7, 6:0.7, 7:0.7, 8:0.7, 9:0.7, 10:0.7}, seed=36)
test = df.subtract(train)
print(train.count()) # Cache because accessing training data multiple times
print(test.count())

# COMMAND ----------

df.groupBy("number_months_available").count().withColumnRenamed('count', 'full').join(
  train.groupBy("number_months_available").count().withColumnRenamed('count', 'strata'), 'number_months_available'
).join(
trainDF.groupBy("number_months_available").count().withColumnRenamed('count', 'random'), 'number_months_available'
).show()



# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ##### Build pipeline

# COMMAND ----------

vecAssembler = VectorAssembler(inputCols=featureInputs, outputCol="features")
gbt = GBTRegressor(maxDepth=8, leafCol="leafId", labelCol = targetCol)
rf = RandomForestRegressor(numTrees=10, maxDepth=5, labelCol = targetCol)

my_seed = 1936
rf.setSeed(my_seed)
gbt.setSeed(my_seed)


# COMMAND ----------

pipeline_gbt = Pipeline(stages=[vecAssembler, gbt]) # Define the pipeline based on the stages created in previous steps that works all the way to the model.
Model_gbt = pipeline_gbt.fit(train) # Define the pipeline model.
predDF_gbt = Model_gbt.transform(test) # Apply the pipeline model to the test dataset.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ##### Evaluate performance

# COMMAND ----------

print(model_results(predDF_gbt,labelCol = targetCol))
importance_plot(Model_gbt)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Export Model

# COMMAND ----------

# read pickled model via pipeline api
from pyspark.ml.pipeline import PipelineModel
from pyspark2pmml import PMMLBuilder

# COMMAND ----------

Model_gbt.write().overwrite().save("dbfs:/FileStore/library/model_gbt_hathaway")
train.write.saveAsTable("library_g25.train_hathaway")

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ##### Export PMML File

# COMMAND ----------

# This is the same thing as Model_gbt
model_load = PipelineModel.load("dbfs:/FileStore/library/model_gbt_hathaway") # spark API format
train = spark.table("library_g25.train_hathaway")
pmmlBuilder = PMMLBuilder(spark, train, model_load) # databricks defaults our spark session to `spark`.
pmmlBuilder.buildFile("/dbfs/FileStore/library/library.pmml") # File API format

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ##### Downloading
# MAGIC 
# MAGIC [Data Thirst](https://datathirst.net/) created [DBFS-Explorer](https://datathirst.net/projects/dbfs-explorer/) that works on Mac or Windows.  You can see their source code on [GitHub](https://github.com/DataThirstLtd/DBFS-Explorer).  They seem to work quite easilty to provide a GUI file explorer to our files on Databricks.
# MAGIC 
# MAGIC You could also use the [Databricks CLI](https://docs.databricks.com/dev-tools/cli/index.html)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Notes on Export
# MAGIC 
# MAGIC - [pyspark.ml.util.GeneralMLWriter](https://spark.apache.org/docs/3.1.1/api/python/reference/api/pyspark.ml.util.GeneralMLWriter.html)
# MAGIC 
# MAGIC 1. [jpmml-sparkml](https://github.com/jpmml/jpmml-sparkml)
# MAGIC 2. [PySpark2PMML](https://github.com/jpmml/pyspark2pmml)
# MAGIC 3. [Deploying Apache Spark ML pipeline models on Openscoring REST web service](https://openscoring.io/blog/2020/02/16/deploying_sparkml_pipeline_openscoring_rest/) 

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ##### Openscoring

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC - https://github.com/openscoring/openscoring
# MAGIC - https://openscoring.io/blog/2020/02/16/deploying_sparkml_pipeline_openscoring_rest/
# MAGIC - https://github.com/openscoring/openscoring-python
# MAGIC - https://github.com/openscoring/openscoring-docker
# MAGIC - https://github.com/openscoring/papis.io
