import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.{VectorAssembler, VectorIndexer}
import org.apache.spark.ml.regression.DecisionTreeRegressor
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import org.apache.spark.{SparkConf, SparkContext}

object BikeSharingDcsTrRgrssnOneMnthLongList {

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

    //change each row of the RDD to a tuple of 10 dobuble type numbers. Methods also used to rerange values
    /*val data0 = dataRDDNoHeader.map(line => {
      val arr = line.split(",")
      (arr(3).toDouble, rerangeMnthsAccordToCnt(arr(4)), rerangeHrsAccordToCnt(arr(5)),
        arr(8).toDouble,  rerangeWthrAccordToCnt(arr(9)), rerangeTempAccordToCnt(arr(10)),
        rerangeAtempAccordToCnt(arr(11)), rerangeHumAccordToCnt(arr(12)),
        rerangeWndSpAccordToCnt(arr(13)), arr(16).toDouble)
    })*/

    val data00 = dataRDDNoHeader.map(line => {
      val arr = line.split(",")
      (arr(3).toDouble, arr(4).toDouble, rerangeHrsAccordToCnt(arr(5)),
        arr(8).toDouble, arr(9).toDouble, arr(10).toDouble,
        arr(11).toDouble, arr(12).toDouble,
        arr(13).toDouble, arr(16).toDouble)
    }).filter(row => row._2 == 5).cache

    val listWthr = abc(data00)

    val data0 = data00.map(row => (row._1, row._2, row._3, row._4, rerangeWthrAccordToCnt(row._5)(listWthr),
                              row._6, row._7, row._8, row._9, row._10))

    //change the RDD to dataframe
    val data = data00.toDF()

    //a value which is a method for rearranging the dataframe to begin with a column named label
    // and then followed by nine columns is created
    val data1 = data.select(data("_10").as("label"),
      $"_1", $"_2", $"_3", $"_4", $"_5", $"_6", $"_7", $"_8", $"_9")

    //a value which is a method for forming an array column of tuples which consisting
    // of the nine columns and is named "iniFeatures"
    val assembler = new VectorAssembler().
      setInputCols(Array("_1", "_2", "_3", "_4", "_5", "_6", "_7", "_8", "_9")).
      setOutputCol(/*"iniFeatures*/"features")

    //a value which is a method for categorising columns with values of which showing category  nature
    // (converting the lowest value to 0.0, the next one 1.0, etc.)
    /*var vectIdxr = new VectorIndexer().
      setInputCol("iniFeatures").
      setOutputCol("features").setMaxCategories(2)*/

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
    val pipln = new Pipeline().setStages(Array(assembler,/*vectIdxr, */regrsr))

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
  def rerangeHrsAccordToCnt(hr: String): Double = {

    val rerangedDblHr = hr.toInt match { case 17 => 23.0; case 18 => 22.0; case 8 => 21.0; case 16 => 20.0;
    case 19 => 19.0; case 13 => 18.0; case 12 => 17.0; case 15 => 16.0; case 14 => 15.0;
    case 20 => 14.0; case 9 => 13.0; case 7 => 12.0; case 11 => 11.0; case 10 =>10.0;
    case 21 => 9.0; case 22 => 8.0; case 23 => 7.0; case 6 => 6.0; case 0 =>5.0;
    case 1 => 4.0; case 2 => 3.0; case 5 => 2.0; case 3 => 1.0; case 4 =>0.0
    }
    rerangedDblHr
  }
  def rerangeMnthsAccordToCnt(mnth: String): Double = {

    val rerangedDblMnth = mnth.toInt match { case 9 => 12.0; case 6 => 11.0; case 8 => 10.0; case 7 => 9.0;
                      case 5 => 8.0; case 10 => 7.0; case 4 => 6.0; case 11 => 5.0; case 3 => 4.0;
                      case 12 => 3.0; case 2 => 2.65; case 1 => 1.2
    }
    rerangedDblMnth
  }
  def abc(dataRDDNoHeader: RDD[(Double, Double, Double, Double, Double, Double, Double,
                                  Double, Double, Double)]): List[(Double, Double)] = {

    val data00 = dataRDDNoHeader.map(a => (a._5, a._10)).
    /*val data00 = dataRDDNoHeader.map(line => {
      val arr = line.split(",")
      (arr(9).toInt, arr(16).toDouble)
    }).*/
      map(tuple => (tuple._1, (tuple._2, 1.0))).
      reduceByKey((a, b) => (a._1 + b._1, a._2 + b._2))

    val data0 = data00.map(weatherCount => (weatherCount._1, weatherCount._2._1 / weatherCount._2._2)).
      map(a => (a._2, a._1)).sortByKey().collect.toList

    data0
  }
  def rerangeWthrAccordToCnt(wthr: Double)(listWthr: List[(Double, Double)]): Double = {

    if (wthr == listWthr.head._2) 0//listWthr(0)._1
    else if (wthr == listWthr(1)._2) 1//listWthr(1)._1
    else if (wthr == listWthr(2)._2) 2//listWthr(2)._1
    else 3//listWthr(3)._1
  }
  def rerangeTempAccordToCnt(temp: String): Double = {

    val rerangedDblTemp = temp.toDouble match { case 0.98 => 49.0; case 0.88 => 48.0; case 0.8 => 47.0; case 0.76 => 46.0;
                    case 0.82 => 45.0; case 0.84 => 44.0; case 0.86 => 43.0; case 0.92 => 42.0; case 0.78 => 41.0;
                    case 0.74 => 40.0; case 0.9 => 39.0; case 1.0 => 38.0; case 0.72 => 37.0; case 0.96 => 36.0; case 0.7 => 35.0;
                    case 0.66 => 34.0; case 0.64 => 33.0; case 0.6 => 32.0; case 0.58 => 31.0; case 0.94 => 30.0; case 0.56 => 29.0;
                    case 0.68 => 28.0; case 0.62 => 27.0; case 0.52 => 26.0; case 0.54 => 25.0; case 0.5 => 24.0; case 0.48 => 23.0;
                    case 0.42 => 22.0; case 0.4 => 21.0; case 0.38 => 20.0; case 0.46 => 19.0; case 0.44 => 18.0; case 0.36 => 17.0;
                    case 0.34 => 16.0; case 0.32 => 15.0; case 0.3 => 14.0; case 0.28 => 13.0; case 0.26 => 12.0; case 0.24 => 11.0;
                    case 0.2 => 10.0; case 0.22 => 9.0; case 0.16 => 8.0; case 0.18 => 7.0; case 0.12 => 6.0; case 0.14 => 5.0;
                    case 0.1 => 4.0; case 0.06 => 3.0; case 0.02 => 2.0; case 0.04 => 1.0; case 0.08 => 0.0;
        }
    rerangedDblTemp
  }
  def rerangeAtempAccordToCnt(atemp: String): Double = {

    val rerangedDblAtemp = atemp.toDouble match { case 0.7273 => 64.0; case 0.7576 => 63.0; case 0.7727 => 62.0; case 0.6515 => 61.0;
                    case 0.6212 => 60.0; case 0.803 => 59.0; case 0.7424 => 58.0; case 0.7121 => 57.0; case 0.9091 => 56.0;
                    case 0.6818 => 55.0; case 0.8636 => 54.0; case 0.8182 => 53.0; case 0.7879 => 52.0; case 0.8485 => 51.0;
                    case 0.697 => 50.0; case 0.6667 => 49.0; case 0.9242 => 48.0; case 0.6364 => 47.0; case 0.8788 => 46.0;
                    case 0.8333 => 45.0; case 0.8939 => 44.0; case 0.6061 => 43.0; case 0.5303=> 42.0; case 0.5 => 41.0;
                    case 0.4848 => 40.0; case 0.4697 => 39.0; case 0.5152 => 38.0; case 0.5455 => 37.0; case 0.4242 => 36.0;
                    case 0.5909 => 35.0; case 0.4091 => 34.0; case 0.3939 => 33.0; case 0.4545 => 32.0; case 0.4394 => 31.0;
                    case 0.5758 => 30.0; case 0.3333 => 29.0; case 0.3485 => 28.0; case 0.3788 => 27.0; case 0.3636 => 26.0;
                    case 0.9848 => 25.0; case 0.3182 => 24.0; case 0.303 => 23.0; case 0.5606 => 22.0; case 0.9545 => 21.0;
                    case 0.2879 => 20.0; case 1.0 => 19.0; case 0.2727 => 18.0; case 0.2424 => 17.0; case 0.2576 => 16.0;
                    case 0.2273 => 15.0; case 0.2121 => 14.0; case 0.1818 => 13.0; case 0.197 => 12.0; case 0.1212 => 11.0;
                    case 0.0909 => 10.0; case 0.1515 => 9.0; case 0.1364 => 8.0; case 0.1061 => 7.0; case 0.1667 => 6.0;
                    case 0.0606 => 5.0; case 0.0758 => 4.0; case 0.0455 => 3.0; case 0.0 => 2.0; case 0.0303 => 1.0; case 0.0152 => 0.0
        }
      rerangedDblAtemp
    }
  def rerangeHumAccordToCnt(hum: String): Double = {

    val rerangedDblHum = hum.toDouble match { case 0.2 => 88.0; case 0.27 => 87.0; case 0.17 => 86.0; case 0.22 => 85.0;
                              case 0.24 => 84.0; case 0.18 => 83.0; case 0.34 => 82.0; case 0.23 => 81.0; case 0.3 => 80.0;
                              case 0.31 => 79.0; case 0.36 => 78.0; case 0.29 => 77.0; case 0.26 => 82.0; case 0.32 => 81.0;
                              case 0.39 => 74.0; case 0.37 => 73.0; case 0.35 => 72.0; case 0.38 => 71.0; case 0.19 => 70.0;
                              case 0.43 => 69.0; case 0.33 => 68.0; case 0.28 => 67.0; case 0.21=> 66.0; case 0.25 => 65.0;
                              case 0.4 => 64.0; case 0.42 => 63.0; case 0.46 => 62.0; case 0.45 => 61.0; case 0.58 => 60.0;
                              case 0.55 => 59.0; case 0.41 => 58.0; case 0.48 => 57.0; case 0.16 => 56.0; case 0.62 => 55.0;
                              case 0.51 => 54.0; case 0.54 => 53.0; case 0.49 => 52.0; case 0.53 => 51.0; case 0.52 => 50.0;
                              case 0.57 => 49.0; case 0.5 => 48.0; case 0.74 => 47.0; case 0.44 => 46.0; case 0.66 => 45.0;
                              case 0.47 => 44.0; case 0.61 => 43.0; case 0.73=> 42.0; case 0.64 => 41.0; case 0.68 => 40.0;
                              case 0.85 => 39.0; case 0.59 => 38.0; case 0.65 => 37.0; case 0.6 => 36.0; case 0.63 => 35.0;
                              case 0.72 => 34.0; case 0.78 => 33.0; case 0.7 => 32.0; case 0.69 => 31.0; case 0.67 => 30.0;
                              case 0.77 => 29.0; case 0.79 => 28.0; case 0.9 => 27.0; case 0.56 => 26.0; case 0.83 => 25.0;
                              case 0.71 => 24.0; case 0.82 => 23.0; case 0.89 => 22.0; case 0.96 => 21.0; case 0.84 => 20.0;
                              case 0.76 => 19.0; case 0.91 => 18.0; case 0.88 => 17.0; case 0.81 => 16.0; case 0.1 => 15.0;
                              case 0.94 => 14.0; case 0.75 => 13.0; case 0.87 => 12.0; case 0.08 => 11.0; case 0.92 => 10.0;
                              case 0.15 => 9.0; case 0.8 => 8.0; case 1.0 => 7.0; case 0.93 => 6.0; case 0.97 => 5.0;
                              case 0.86 => 4.0; case 0.12 => 3.0; case 0.0 => 2.0; case 0.14 => 1.0; case 0.13 => 0.0
    }
    rerangedDblHum
  }
  def rerangeWndSpAccordToCnt(wndSp: String): Double = {

    val rerangedDblWndSp = wndSp.toDouble match { case 0.8507=>14.0; case 0.3881=>14.0; case 0.2836=>13.0; case 0.2985=>13.0;
                            case 0.4179=>12.0; case 0.2537=>12.0; case 0.2239=>11.0; case 0.3582=>11.0; case 0.4478=>10.0; case 0.5224=>10.0;
                            case 0.194=>9.0; case 0.3284=>9.0; case 0.4627=>8.0; case 0.1642=>8.0; case 0.4925=>7.0; case 0.6567=>7.0;
                            case 0.7463=>6.0; case 0.1343=>6.0; case 0.1045=>5.0; case 0.5821=>5.0; case 0.5522=>4.0; case 0.6119=>4.0;
                            case 0.0=>3.0; case 0.0896=>3.0; case 0.6418=>2.0; case 0.7164=>2.0; case 0.6866=>1.0; case 0.806=>1.0;
                            case 0.8358=>1.0; case 0.7761=>1.0;
    }
    rerangedDblWndSp
  }
}

//May 5:
//RMSE of the training set is: 82.40917170207214
//RMSE of the test set obtained by using the model produced with the training set: 81.85204095686058
//R-Squared(R2) of the test set obtained by using the model produced with the training set: 0.7976330490671808
