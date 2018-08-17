import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.{VectorAssembler, VectorIndexer}
import org.apache.spark.ml.regression.{DecisionTreeRegressor, LinearRegression}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql._
import org.apache.spark.{SparkConf, SparkContext}

import scala.util.control.Breaks._

object BikeSharingHourPredictionAndEvaluationDcsTr_3rdYrMnthHrWDWthr {

  def main(args: Array[String]): Unit = {

    import org.apache.log4j._
    Logger.getLogger("org").setLevel(Level.ERROR)

    val conf = new SparkConf().setAppName("retail").setMaster("local[*]")
    val sc = new SparkContext(conf)
    val spark = SparkSession.builder().appName("retail1").
      master("local[*]").getOrCreate()
    import spark.implicits._

    val dataRDD = sc.textFile("D:\\hairat dataset\\hour.csv")

    val dataRDDNoHeader = dataRDD.filter(line => line.split(",")(0) != "instant")

    val data00 = dataRDDNoHeader.map(line => {
      val arr = line.split(",")
      (arr(4).toDouble, arr(3).toDouble, arr(5).toDouble, arr(8).toDouble, arr(9).toDouble,
        arr(11).toDouble, arr(12).toDouble, arr(13).toDouble, arr(16).toDouble)
    }).cache()

    val sortMnthList = sortMnth(data00.map(row => (row._1, row._9)))
    val sortHrList = sortHr(data00.map(row => (row._3, row._9)))
    val sortWthrList = sortWthr(data00.map(row => (row._5, row._9)))
    val sortHumList = sortHum(data00.map(row => (row._7, row._9)))
    val sortWindSpList = sortWndSp(data00.map(row => (row._8, row._9)))

    val data0 = data00.map(row=>(rerangeMnthAccordToCnt1(row._1)(sortMnthList), row._2,
      rerangeHrsAccordToCnt1(row._3)(sortHrList), row._4,
      rerangewthrAccordToCnt1(row._5)(sortWthrList),
      rerangeHumAccordToCnt1(row._8)(sortHumList),
      rerangeWndSpAccordToCnt1(row._9)(sortWindSpList), row._9))

    val data = data0.toDF()

    val data1 = data.select(data("_9").as("label"),
      $"_1", $"_2", $"_3", $"_4", $"_5", $"_6", $"_7", $"_8")

    val assembler = new VectorAssembler().
      setInputCols(Array("_1", "_2", "_3", "_4", "_5", "_6", "_7", "_8")).
      setOutputCol("iniFeatures")

    var vectIdxr = new VectorIndexer().
      setInputCol("iniFeatures").
      setOutputCol("features").setMaxCategories(24)

    val regrsr = new DecisionTreeRegressor()
      .setFeaturesCol("features")
      .setLabelCol("label")
      .setMaxBins(32)

    val paramGrid = new ParamGridBuilder().
      //addGrid(regrsr.regParam, Array(0.1, 0.01, 0.001)).
      //addGrid(regrsr.fitIntercept).
      //addGrid(regrsr.elasticNetParam, Array(0.0, 0.2, 0.4, 0.6, 0.8, 1.0)).
      build()

    val pipln = new Pipeline().setStages(Array(assembler, vectIdxr, regrsr))

    val crossValidator = new CrossValidator()
      .setEstimator(pipln)
      .setEvaluator(new RegressionEvaluator().setMetricName("rmse"))
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(3)

    val Array(training, test) = data1.randomSplit(Array(0.8, 0.2), seed = 7000)
    //val dataCheck = Seq((28.6826451, 11.0, 1.0, 2.0, 1.0, 1.0, 31.0, 37.0, 41.0, 24.0)).
    //  toDF("label", "_1", "_2", "_3", "_4", "_5", "_6", "_7", "_8", "_9")

    val model = crossValidator.fit(training)
    val result = model.transform(test)
    val resultSummary = result.summary()
    resultSummary.show()
    result.select("*").show(100)
    //val checkResult = model.transform(dataCheck).select("*")
    val metrics = model.getEstimatorParamMaps.zip(model.avgMetrics)
    val highestR2 = metrics.map(a => a._2).foldLeft(0.0)((a,b)=> if (a>b)a; else b)

    println("Schema of the dataframe to be feed into the machine learning process:")
    data1.printSchema()
    println("First twenty rows of the dataframe to be feed into the machine learning process")
    data1.show
    println("First 100 rows of the result dataframe:")
    result.show(400)
    //println("One-row dataframe for checking:")
    //checkResult.show()
    println("Metrics. Only those metrics in the tuple which contains the highest r2 value\n" +
      "(i.e. the last/second item of a tuple) are actually used for the final model selected:")
    for (m <- metrics) println(m)
    println("So the model's r2 is: " + highestR2)
  }
  def sortMnth(data: RDD[(Double, Double)]) = {
    val data0 = data.map(tuple => (tuple._1, (tuple._2, 1))).
      reduceByKey((a, b) => (a._1 + b._1, a._2 + b._2)).
      map(mnthCount => (mnthCount._1, mnthCount._2._1 / mnthCount._2._2)).
      map(a => (a._2, a._1)).sortByKey().collect.toList
    data0
  }
  def rerangeMnthAccordToCnt1(wndSp: Double)(winSpList: List[(Double, Double)]): Double = {
    var a = 0.0
    breakable {
      for (wspl <- winSpList) {
        if (wndSp == wspl._2) {
          a =  wspl._1
          break()
        }
      }
    }
    a
  }
  def sortHr(data: RDD[(Double, Double)]) = {
    val data0 = data.map(tuple => (tuple._1, (tuple._2, 1))).
      reduceByKey((a, b) => (a._1 + b._1, a._2 + b._2)).
      map(hrCount => (hrCount._1, hrCount._2._1 / hrCount._2._2)).
      map(a => (a._2, a._1)).sortByKey().collect.toList
    data0
  }
  def rerangeHrsAccordToCnt1(wndSp: Double)(winSpList: List[(Double, Double)]): Double = {
    var a = 0.0
    breakable {
      for (wspl <- winSpList) {
        if (wndSp == wspl._2) {
          a =  wspl._1
          break()
        }
      }
    }
    a
  }
  def sortWthr(data: RDD[(Double, Double)]) = {
    val data0 = data.map(tuple => (tuple._1, (tuple._2, 1))).
      reduceByKey((a, b) => (a._1 + b._1, a._2 + b._2)).
      map(wthrCount => (wthrCount._1, wthrCount._2._1 / wthrCount._2._2)).
      map(a => (a._2, a._1)).sortByKey().collect.toList
    data0
  }
  def rerangewthrAccordToCnt1(wndSp: Double)(winSpList: List[(Double, Double)]): Double = {
    var a = 0.0
    breakable {
      for (wspl <- winSpList) {
        if (wndSp == wspl._2) {
          a = wspl._1
          break()
        }
      }
    }
    a
  }
  def sortTemp(data: RDD[(Double, Double)]) = {
    val data0 = data.map(tuple => (tuple._1, (tuple._2, 1))).
      reduceByKey((a, b) => (a._1 + b._1, a._2 + b._2)).
      map(tempCount => (tempCount._1, tempCount._2._1 / tempCount._2._2)).
      map(a => (a._2, a._1)).sortByKey().collect.toList
    data0
  }
  def rerangeTempAccordToCnt1(wndSp: Double)(winSpList: List[(Double, Double)]): Double = {
    var a = 0.0
    breakable {
      for (wspl <- winSpList) {
        if (wndSp == wspl._2) {
          a = wspl._1
          break()
        }
      }
    }
    a
  }
  def sortHum(data: RDD[(Double, Double)]) = {
    val data0 = data.map(tuple => (tuple._1, (tuple._2, 1))).
      reduceByKey((a, b) => (a._1 + b._1, a._2 + b._2)).
      map(humCount => (humCount._1, humCount._2._1 / humCount._2._2)).
      map(a => (a._2, a._1)).sortByKey().collect.toList
    data0
  }
  def rerangeHumAccordToCnt1(wndSp: Double)(winSpList: List[(Double, Double)]): Double = {
    var a = 0.0
    breakable {
      for (wspl <- winSpList) {
        if (wndSp == wspl._2) {
          a = wspl._1
          break()
        }
      }
    }
    a
  }
  def sortWndSp(data: RDD[(Double, Double)]) = {
    val data0 = data.map(tuple => (tuple._1, (tuple._2, 1))).
      reduceByKey((a, b) => (a._1 + b._1, a._2 + b._2)).
      map(wspCount => (wspCount._1, wspCount._2._1 / wspCount._2._2)).
      map(a => (a._2, a._1)).sortByKey().collect.toList
    data0
  }
  def rerangeWndSpAccordToCnt1(wndSp: Double)(wnSpList: List[(Double, Double)]): Double = {
    var a = 0.0
    breakable {
      for (wspl <- wnSpList) {
        if (wndSp == wspl._2) {
          a = wspl._1
          break()
        }
      }
    }
    a
  }
}

//0.6519878263608245 201808142011
//0.6743955023649476 use case 17 => 461.45  201808151537
//0.67503801406323 further set categories to 2 only 201808151649
//74.33333333333333/ 111.57928118393235/ 175.16549295774647/ 204.8692718829405/

/*val rerangedDblWndSp = wndSp.toDouble match { case 0.8507=>29.0; case 0.3881=>28.0; case 0.2836=>27.0; case 0.2985=>26.0;
case 0.4179=>25.0; case 0.2537=>24.0; case 0.2239=>23.0; case 0.3582=>22.0; case 0.4478=>21.0; case 0.5224=>20.0;
case 0.194=>19.0; case 0.3284=>18.0; case 0.4627=>17.0; case 0.1642=>16.0; case 0.4925=>15.0; case 0.6567=>14.0;
case 0.7463=>13.0; case 0.1343=>12.0; case 0.1045=>11.0; case 0.5821=>10.0; case 0.5522=>9.0; case 0.6119=>8.0;
case 0.0=>7.0; case 0.0896=>6.0; case 0.6418=>5.0; case 0.7164=>4.0; case 0.6866=>3.0; case 0.806=>2.0;
case 0.8358=>1.0; case 0.7761=>0.0;*/