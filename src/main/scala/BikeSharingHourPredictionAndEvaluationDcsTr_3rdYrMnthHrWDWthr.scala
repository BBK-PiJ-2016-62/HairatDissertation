import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.{VectorAssembler, VectorIndexer}
import org.apache.spark.ml.regression.{DecisionTreeRegressor, LinearRegression}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql._
import org.apache.spark.{SparkConf, SparkContext}

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

    val data0 = dataRDDNoHeader.map(line => {
      val arr = line.split(",")
      (rerangeMnthsAccordToCnt(arr(4)), arr(3).toDouble, rerangeHrsAccordToCnt(arr(5)),
        arr(8).toDouble,  rerangeWthrAccordToCnt(arr(9)), rerangeTempAccordToCnt(arr(10)),
        rerangeAtempAccordToCnt(arr(11)), rerangeHumAccordToCnt(arr(12)),
        rerangeWndSpAccordToCnt(arr(13)), arr(16).toDouble)
    })

    val data = data0.toDF()

    val data1 = data.select(data("_10").as("label"),
      $"_1", $"_2", $"_3", $"_4", $"_5", $"_6", $"_7", $"_8", $"_9")

    val assembler = new VectorAssembler().
      setInputCols(Array("_1", "_2", "_3", "_4", "_5", "_6", "_7", "_8", "_9")).
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
  def rerangeWthrAccordToCnt(wthr: String): Double = {

    val rerangedDblwthr = wthr.toInt match { case 4 => 3.0; case 3 => 2.0; case 2 => 1.0; case 1 => 0.0}
    rerangedDblwthr
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