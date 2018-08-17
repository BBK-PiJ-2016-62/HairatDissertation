import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql._
import org.apache.spark
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.types._
import org.apache.spark.ml.feature.{VectorAssembler, VectorIndexer}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.mllib.evaluation.RegressionMetrics


val conf = new SparkConf().setAppName("retail").setMaster("local[*]")
val sc = new SparkContext(conf)
val spark = SparkSession.builder().appName("retail1").master("local[1*").getOrCreate()
import spark.implicits._

val dataRDD = sc.textFile("D:\\hairat dataset\\hour.csv")

val data0 = dataRDD.filter(line => line.split(",")(0) != "instant").
  map(line => {val arr = line.split(",")
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
  setOutputCol("features").setMaxCategories(4)

val lr = new LinearRegression()

val pipln = new Pipeline().setStages(Array(assembler, vectIdxr, lr))

val Array(training, test)=data1.randomSplit(Array(0.8, 0.2), seed = 7000)

val lrModel = pipln.fit(training)

val fullPredictions = lrModel.transform(test).cache()

val predictions = fullPredictions.select("prediction").rdd.map(_.getDouble(0))

val labels = fullPredictions.select("label").rdd.map(_.getDouble(0))

val RMSE = new RegressionMetrics(predictions.zip(labels)).rootMeanSquaredError

println(s" Root mean squared error (RMSE): $RMSE")

/*
val testSummary = fullPredictions.summary()
testSummary.*/