import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.{VectorAssembler, VectorIndexer}
import org.apache.spark.ml.regression.DecisionTreeRegressor
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import org.apache.spark.{SparkConf, SparkContext}

import scala.util.control.Breaks.{break, breakable}

object BikeSharingDcsTrRgrssnWhlYr {

  def main(args: Array[String]): Unit = {

    //for reducing logs
    import org.apache.log4j._
    Logger.getLogger("org").setLevel(Level.ERROR)

    //setting up spark context and spark session
    val conf = new SparkConf().setAppName("retail").setMaster("local[*]")
    val sc = new SparkContext(conf)
    val spark = SparkSession.builder().appName("retail1").
      master("local[*]").getOrCreate()
    import spark.implicits._

    //importing data and storing it as a resilient dataset (RDD)
    val dataRDD = sc.textFile("hour.csv")

    //delete header
    val dataRDDNoHeader = dataRDD.filter(line => line.split(",")(0) != "instant")

    val data00 = dataRDDNoHeader.map(line => {
      val arr = line.split(",")
      (arr(4).toDouble, arr(3).toDouble, arr(5).toDouble, arr(8).toDouble, arr(9).toDouble,
        arr(11).toDouble, arr(12).toDouble, arr(13).toDouble, arr(16).toDouble)
    }).cache()

    val sortMnthList = sortMnth(data00.map(row => (row._1, row._9)))
    val sortHrList = sortHr(data00.map(row => (row._3, row._9)))
    val sortWthrList = sortWthr(data00.map(row => (row._5, row._9)))
    val sortTempList = sortTemp(data00.map(row => (row._6, row._9)))
    val sortHumList = sortHum(data00.map(row => (row._7, row._9)))
    val sortWindSpList = sortWndSp(data00.map(row => (row._8, row._9)))

    val data0 = data00.map(row=>(rerangeMnthAccordToCnt1(row._1)(sortMnthList), row._2,
      rerangeHrsAccordToCnt1(row._3)(sortHrList), row._4,
      rerangewthrAccordToCnt1(row._5)(sortWthrList),
      rerangeTempAccordToCnt1(row._6)(sortTempList),
      rerangeHumAccordToCnt1(row._7)(sortHumList),
      rerangeWndSpAccordToCnt1(row._8)(sortWindSpList), row._9))


    //change the RDD to dataframe
    val data = data0.toDF()

    //a value which is a method for rearranging the dataframe to begin with a column named label
    // and then followed by nine columns is created
    val data1 = data.select(data("_9").as("label"),
      $"_1", $"_2", $"_3", $"_4", $"_5", $"_6", $"_7", $"_8")

    //a value which is a method for forming an array column of tuples which consisting
    // of the nine columns and is named "iniFeatures"
    val assembler = new VectorAssembler().
      setInputCols(Array("_1", "_2", "_3", "_4", "_5", "_6", "_7", "_8")).
      setOutputCol("iniFeatures")

    //a value which is a method for categorising columns with values of which showing category  nature
    // (converting the lowest value to 0.0, the next one 1.0, etc.)
    var vectIdxr = new VectorIndexer().
      setInputCol("iniFeatures").
      setOutputCol("features").setMaxCategories(2)

    //a value which is for creating a new decision tree regressor object
    val regrsr = new DecisionTreeRegressor()
      .setFeaturesCol("features")
      .setLabelCol("label")

    //a value which is for creating a parameter grid. Since default parameters should be used, the grid here is empty.
    //Whether creating this variable or not is optional
    val paramGrid = new ParamGridBuilder().
      //addGrid(regrsr.setMaxDepth("none")). //better keep all to defualt value. showing this method here for learning
      build()

    //a value for creating a pipeline for execution three methods mentioned above
    val pipln = new Pipeline().setStages(Array(assembler, vectIdxr, regrsr))

    //a value for creating a cross validator which, when executed, can perform the methods mentioned in the above
    //pipeline on the data feed in. It can also create several ML models by using every combination of the parameters
    //placed in the paramGrid and select the best one
    val crossValidator = new CrossValidator()
      .setEstimator(pipln)
      .setEvaluator(new RegressionEvaluator().setMetricName("rmse"))  //more reliable than r2
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(10)  //import data will be split to 10 sets(here), each set will be treated as the test
                       //set and run with parameters set at paraGrid and find the best model.

    //splitting the dataframe into a training set and a test set
    val Array(training, test) = data1.randomSplit(Array(0.8, 0.2), seed = 700)

    //the training set is used to create a decision tree regression model
    val model = crossValidator.fit(training)

    //forming a value which is an array of tuples of the parameters used by the models and the
    // evaluation result of each such model
    val metrics = model.getEstimatorParamMaps.zip(model.avgMetrics)

    //Only the model having the best evaluation result would be selected and this variable is a method for
    //finding the evaluation result of the model used (the best model/ here rmse is used for evaluation)
    val lowestRmseOfTrainingSet = metrics.map(a => a._2).foldLeft(Double.MaxValue)((a,b)=> if (a>b)b; else a)

    //feed the test set to the model for evaluating it
    val result = model.transform(test)

    //an object for evaluation of the model is created. rmse is used for the evaluation
    val evaluatorForTestSetRmse = new RegressionEvaluator().
          setLabelCol("label").
          setPredictionCol("prediction").
          setMetricName("rmse")

    //result of the test set is evaluated(rmse)
    val testSetRmse = evaluatorForTestSetRmse.evaluate(result)

    //another object for evaluation of the model is created. r2 is used for the evaluation
    val evaluatorForTestSetR2 = new RegressionEvaluator().
      setLabelCol("label").
      setPredictionCol("prediction").
      setMetricName("r2")

    //result of the test set is evaluated(r2)
    val testSetR2 = evaluatorForTestSetR2.evaluate(result)

    //below until the end are for printing results of this program
    println("Schema of the dataframe to be feed into the machine learning process:")
    data1.printSchema()
    println("")
    println("First twenty rows of the dataframe to be feed into the machine learning process")
    data1.show
    println("")
    //(printing the first, the middle and the end of the table of evaluated test set
    // which consists of labels and predictions)
    //(predictions are closer to labels at the middle than at the beginning and the end of the table)
    result.createTempView("result1")
    val tmpTableResult = spark.sql("select row_number() over (order by label) as rnk, * from result1")
    tmpTableResult.createTempView("result2")
    val testSetRows = tmpTableResult.count()
    val midTbl1stRow = (testSetRows/2).toInt - 49; val midTbllstRow = midTbl1stRow + 99
    println("Table of the first 100 rows of the test set with labels and predictions: ")
    val fist100RowsTable = spark.sql("select * from result2 where rnk <= 100").toDF()
    fist100RowsTable.show(100)
    println("Tables of the middle 100 rows of the test set with labels and predictions: ")
    val middle100RowTable = spark.sql(s"""select * from result2 where rnk between $midTbl1stRow and $midTbllstRow""").toDF()
    middle100RowTable.show(100)
    println("Tables of the last 100 rows of the test set with labels and predictions: ")
    val last100RowTable = spark.sql(s"""select * from result2 where rnk >= ${testSetRows-99}""").toDF()
    last100RowTable.show()

    print("")
    println("RMSE of the training set is: " + lowestRmseOfTrainingSet)
    println("RMSE of the test set obtained by using the model produced with the training set: " + testSetRmse)
    println("R-Squared(R2) of the test set obtained by using the model produced with the training set: " + testSetR2)
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

//RMSE of the training set is: 91.70286201037962
//RMSE of the test set obtained by using the model produced with the training set: 88.96083804055233
//R-Squared(R2) of the test set obtained by using the model produced with the training set: 0.7596510721062182