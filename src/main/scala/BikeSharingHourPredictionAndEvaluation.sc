import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.{VectorAssembler, VectorIndexer}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql._
import org.apache.spark.{SparkConf, SparkContext}

import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)

val conf = new SparkConf().setAppName("retail").setMaster("local[*]")
val sc = new SparkContext(conf)
val spark = SparkSession.builder().appName("retail1").
            master("local[1*").getOrCreate()
import spark.implicits._

val dataRDD = sc.textFile("D:\\hairat dataset\\hour.csv")

val dataRDDNoHeader = dataRDD.filter(line => line.split(",")(0) != "instant")

val data0 = dataRDDNoHeader.map(line => {val arr = line.split(",")
    (arr(2).toDouble, arr(3).toDouble, arr(4).toDouble, arr(5).toDouble,
     arr(6).toDouble, arr(7).toDouble, arr(8).toDouble, arr(9).toDouble,
     arr(10).toDouble, arr(11).toDouble, arr(12).toDouble,
     arr(13).toDouble, arr(16).toDouble)})

val data = data0.toDF()

data.printSchema()
data.show

val data1 = data.select(data("_13").as("label"),
                        $"_1",$"_2",$"_3",$"_4",$"_5",$"_6",
                        $"_7",$"_8",$"_9",$"_10",$"_11",$"_12")

val assembler = new VectorAssembler().
  setInputCols(Array("_1","_2","_3","_4","_5","_6",
                    "_7","_8","_9","_10","_11","_12")).
                setOutputCol("iniFeatures")

var vectIdxr = new VectorIndexer().
  setInputCol("iniFeatures").
  setOutputCol("features").setMaxCategories(10)

val lr = new LinearRegression()
  .setFeaturesCol("features")
  .setLabelCol("label")

val paramGrid = new ParamGridBuilder().
                      addGrid(lr.regParam, Array(0.1, 0.01, 0.001)).
                      addGrid(lr.fitIntercept).
                      addGrid(lr.elasticNetParam, Array(0.0, 0.2, 0.4, 0.6, 0.8, 1.0)).
                      build()

val pipln = new Pipeline().setStages(Array(assembler, vectIdxr, lr))

val crossValidator = new CrossValidator()
  .setEstimator(pipln)
  .setEvaluator(new RegressionEvaluator().setMetricName("r2"))
  .setEstimatorParamMaps(paramGrid)
  .setNumFolds(3)

val Array(training, test)=data1.randomSplit(Array(0.8, 0.2), seed = 7000)
val dataCheck = Seq((7.68305391, 1.0, 0.0, 1.0, 4.0, 0.0, 4.0, 1.0, 1.0, 0.26, 0.2576, 0.56, 0.1642)).
 toDF("label","_1","_2","_3","_4","_5","_6","_7","_8","_9","_10","_11","_12")

val lrModel = crossValidator.fit(training)
lrModel.transform(test).select(/*"features","label", "prediction"*/ "*" ).show(100)
lrModel.transform(dataCheck).select("*").show

lrModel.getEstimatorParamMaps.zip(lrModel.avgMetrics)
