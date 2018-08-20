import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.{VectorAssembler, VectorIndexer}
import org.apache.spark.ml.regression.DecisionTreeRegressor
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import org.apache.spark.{SparkConf, SparkContext}

import scala.util.control.Breaks._

object BikeSharing_5th_83_DcsTrRgrssnOneMnth {

  def main(args: Array[String]): Unit = {

    //for reducing logs
    import org.apache.log4j._
    Logger.getLogger("org").setLevel(Level.ERROR)

    //setting up spark context and spark session
    val conf = new SparkConf().setAppName("retail").setMaster("local[*]")
    val sc = new SparkContext(conf)
    val spark = SparkSession.builder().appName("retail").
      master("local[*]").getOrCreate()
    import spark.implicits._

    //importing data and storing it as a resilient distributed dataset (RDD)
    val dataRDD = sc.textFile("hour.csv")

    //delete header
    val dataRDDNoHeader = dataRDD.filter(line => line.split(",")(0) != "instant")

    //choosing useful columns
    val data000 = dataRDDNoHeader.map(line => {
      val arr = line.split(",")
      (arr(3).toDouble, arr(4).toDouble, arr(5).toDouble, arr(8).toDouble, arr(9).toDouble,
        arr(10).toDouble, arr(11).toDouble, arr(12).toDouble, arr(13).toDouble, arr(16).toDouble)
    }).filter(row => row._2 == 6.0).cache()

    //split data into training set and data set
    val set1 = data000.randomSplit(Array(0.8, 0.2), seed = 7000)
    val data00 = set1(0)
    val data01 = set1(1)

    //rerange column values according to the average count values of the training set
    val sortHrList = sortItem(data00.map(row => (row._3, row._10)))
    val sortWthrList = sortItem(data00.map(row => (row._5, row._10)))
    val sortTempList = sortItem(data00.map(row => (row._6, row._10)))
    val sortAtempList = sortItem(data00.map(row => (row._7, row._10)))
    val sortHumList = sortItem(data00.map(row => (row._8, row._10)))
    val sortWindSpList = sortItem(data00.map(row => (row._9, row._10)))

    //revaluing fields according to the reranged column values. withdrawing the mnth column row._2
    val data0 = data00.map(row=>(row._1, rerangeAccordToCnt(row._3)(sortHrList), row._4,
      rerangeAccordToCnt(row._5)(sortWthrList),
      rerangeAccordToCnt(row._6)(sortTempList),
      rerangeAccordToCnt(row._7)(sortAtempList),
      rerangeAccordToCnt(row._8)(sortHumList),
      rerangeAccordToCnt(row._9)(sortWindSpList), row._10))
    val dataTt0 = data01.map(row=>(row._1, rerangeAccordToCnt(row._3)(sortHrList), row._4,
      rerangeAccordToCnt(row._5)(sortWthrList),
      rerangeAccordToCnt(row._6)(sortTempList),
      rerangeAccordToCnt(row._7)(sortAtempList),
      rerangeAccordToCnt(row._8)(sortHumList),
      rerangeAccordToCnt(row._9)(sortWindSpList), row._10))

    //change the RDDs to dataframes
    val data = data0.toDF()
    val dataTt = dataTt0.toDF()

    //a value which is a function for rearranging the columns of the dataframes to begin with a column named label
    // and then followed by nine columns is created
    val data1 = data.select(data("_9").as("label"),
      $"_1", $"_2", $"_3", $"_4", $"_5", $"_6", $"_7", $"_8")
    val dataTt1 = dataTt.select(dataTt("_9").as("label"),
      $"_1", $"_2", $"_3", $"_4", $"_5", $"_6", $"_7",$"_8")

    //a value which is a function for forming an array column of tuples which consisting
    // of seven items and is named "iniFeatures"
    val assembler = new VectorAssembler().
      setInputCols(Array("_1", "_2", "_3", "_4", "_5", "_6", "_7","_8")).
      setOutputCol("iniFeatures"/*"features"*/)

    //a value which is a function for categorising columns with values of which showing category  nature
    // (converting the lowest value to 0.0, the next one 1.0, etc.)
    var vectIdxr = new VectorIndexer().
      setInputCol("iniFeatures").
      setOutputCol("features").setMaxCategories(12)

    //a value which is for creating a new decision tree regressor object
    val regrsr = new DecisionTreeRegressor()
      .setFeaturesCol("features")
      .setLabelCol("label")

    //a value which is for creating a parameter grid. Since default parameters should be used, the grid here is empty.
    //Whether creating this variable or not is optional
    val paramGrid = new ParamGridBuilder().
      //addGrid(regrsr.setMaxDepth("none")). //better keep all to defualt value. showing this method here for learning
      build()

    //a value for creating a pipeline for execution three/two methods mentioned above
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

    //the training set is used to create a decision tree regression model
    val model = crossValidator.fit(data1)

    //forming a value which is an array of tuples of the parameters used by the models and the
    // evaluation result of each such model
    val metrics = model.getEstimatorParamMaps.zip(model.avgMetrics)

    //Only the model having the best evaluation result would be selected and this variable is a method for
    //finding the evaluation result of the model used (the best model/ here rmse is used for evaluation)
    val lowestRmseOfTrainingSet = metrics.map(a => a._2).foldLeft(Double.MaxValue)((a,b)=> if (a>b)b; else a)

    //feed the test set to the model for evaluating it
    val result = model.transform(dataTt1)

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
    last100RowTable.show(100)

    print("")
    println("RMSE of the training set is: " + lowestRmseOfTrainingSet)
    println("RMSE of the test set obtained by using the model produced with the training set: " + testSetRmse)
    println("R-Squared(R2) of the test set obtained by using the model produced with the training set: " + testSetR2)
  }
  def sortItem(data: RDD[(Double, Double)]) = {
    val data0 = data.map(tuple => (tuple._1, (tuple._2, 1))).
      reduceByKey((a, b) => (a._1 + b._1, a._2 + b._2)).
      map(counts => (counts._1, counts._2._1 / counts._2._2)).
      map(a => (a._2, a._1)).sortByKey().collect.toList
    data0
  }
  def rerangeAccordToCnt(item: Double)(sortedList: List[(Double, Double)]): Double = {
    var a = 0.0
    breakable {
      for (srtd <- sortedList) {
        if (item == srtd._2) {
          a =  srtd._1
          break()
        }
      }
    }
    a
  }
}

//May 5:
//RMSE of the training set is: 84.83881134599919
//RMSE of the test set obtained by using the model produced with the training set: 79.62615774055519
//R-Squared(R2) of the test set obtained by using the model produced with the training set: 0.8252721831032866
//June 6
//RMSE of the training set is: 87.78382132550354
//RMSE of the test set obtained by using the model produced with the training set: 73.69486263612023
//R-Squared(R2) of the test set obtained by using the model produced with the training set: 0.8597514726616671

// 6 RMSE of the training set is: 89.68584908815377
//RMSE of the test set obtained by using the model produced with the training set: 79.31358675423822
//R-Squared(R2) of the test set obtained by using the model produced with the training set: 0.833683034434869 */