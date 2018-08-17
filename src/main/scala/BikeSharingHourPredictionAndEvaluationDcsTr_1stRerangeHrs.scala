import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.{VectorAssembler, VectorIndexer}
import org.apache.spark.ml.regression.{DecisionTreeRegressor, LinearRegression}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql._
import org.apache.spark.{SparkConf, SparkContext}

object BikeSharingHourPredictionAndEvaluationDcsTr_1stRerangeHrs {

  def main(args: Array[String]): Unit = {

    import org.apache.log4j._
    Logger.getLogger("org").setLevel(Level.ERROR)

    val conf = new SparkConf().setAppName("retail").setMaster("local[*]")
    val sc = new SparkContext(conf)
    val spark = SparkSession.builder().appName("retail1").
      master("local[1*").getOrCreate()
    import spark.implicits._

    val dataRDD = sc.textFile("D:\\hairat dataset\\hour.csv")

    val dataRDDNoHeader = dataRDD.filter(line => line.split(",")(0) != "instant")

    val data0 = dataRDDNoHeader.map(line => {
      val arr = line.split(",")
      (arr(2).toDouble, arr(3).toDouble, arr(4).toDouble, arr(5).toDouble,  //rerangeHrsAccordToCnt(arr(5)),
        arr(6).toDouble, arr(7).toDouble, arr(8).toDouble, arr(9).toDouble,
        arr(10).toDouble, arr(11).toDouble, arr(12).toDouble,
        arr(13).toDouble, arr(16).toDouble)
    })

    val data = data0.toDF()

    val data1 = data.select(data("_13").as("label"),
      $"_1", $"_2", $"_3", $"_4", $"_5", $"_6",
      $"_7", $"_8", $"_9", $"_10", $"_11", $"_12")

    val assembler = new VectorAssembler().
      setInputCols(Array("_1", "_2", "_3", "_4", "_5", "_6",
        "_7", "_8", "_9", "_10", "_11", "_12")).
      setOutputCol("iniFeatures")

    var vectIdxr = new VectorIndexer().
      setInputCol("iniFeatures").
      setOutputCol("features").setMaxCategories(24)

    val regrsr = new DecisionTreeRegressor()
      .setFeaturesCol("features")
      .setLabelCol("label")

    val paramGrid = new ParamGridBuilder().
      //addGrid(regrsr.regParam, Array(0.1, 0.01, 0.001)).
      //addGrid(regrsr.fitIntercept).
      //addGrid(regrsr.elasticNetParam, Array(0.0, 0.2, 0.4, 0.6, 0.8, 1.0)).
      build()

    val pipln = new Pipeline().setStages(Array(assembler, vectIdxr, regrsr))

    val crossValidator = new CrossValidator()
      .setEstimator(pipln)
      .setEvaluator(new RegressionEvaluator().setMetricName("r2"))
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(3)

    val Array(training, test) = data1.randomSplit(Array(0.8, 0.2), seed = 7000)
    val dataCheck = Seq((7.68305391, 1.0, 0.0, 1.0, 4.0, 0.0, 4.0, 1.0, 1.0, 0.26, 0.2576, 0.56, 0.1642)).
      toDF("label", "_1", "_2", "_3", "_4", "_5", "_6", "_7", "_8", "_9", "_10", "_11", "_12")

    val model = crossValidator.fit(training)
    val result = model.transform(test).select("features","label", "prediction")
    val checkResult = model.transform(dataCheck).select("*")
    val metrics = model.getEstimatorParamMaps.zip(model.avgMetrics)
    val highestR2 = metrics.map(a => a._2).foldLeft(0.0)((a,b)=> if (a>b)a; else b)

    val Array(training1, test1) = data0.randomSplit(Array(0.8, 0.2), seed = 7000)
    training1.take(5).foreach(println)
    println("Schema of the dataframe to be feed into the machine learning process:")
    data1.printSchema()
    println("First twenty rows of the dataframe to be feed into the machine learning process")
    data1.show
    println("First 100 rows of the result dataframe:")
    result.show(100)
    println("One-row dataframe for checking:")
    checkResult.show()
    println("Metrics. Only those metrics in the tuple which contains the highest r2 value\n" +
      "(i.e. the last/second item of a tuple) are actually used for the final model selected:")
    for (m <- metrics) println(m)
    println("So the model's r2 is: " + highestR2)
  }
  def rerangeHrsAccordToCnt(hr: String): Double = {

    val rerangedDbl = hr.toInt match { case 17 => 23.0; case 18 => 22.0; case 8 => 21.0; case 16 => 20.0;
                      case 19 => 19.0; case 13 => 18.0; case 12 => 17.0; case 15 => 16.0; case 14 => 15.0;
                      case 20 => 14.0; case 9 => 13.0; case 7 => 12.0; case 11 => 11.0; case 10 =>10.0
                      case 21 => 9.0; case 22 => 8.0; case 23 => 7.0; case 6 => 6.0; case 0 =>5.0
                      case 1 => 4.0; case 2 => 3.0; case 5 => 2.0; case 3 => 1.0; case 4 =>0.0
    }
    rerangedDbl
  }
}

//0.6511867923904234  201808142011