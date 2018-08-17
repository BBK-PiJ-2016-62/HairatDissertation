import org.apache.spark.sql._
import org.apache.spark.{SparkConf, SparkContext}

object BikeSharingHourStudyColumn_2ndmnth {

  def main(args: Array[String]): Unit = {

    import org.apache.log4j._
    Logger.getLogger("org").setLevel(Level.ERROR)

    val conf = new SparkConf().setAppName("retail").setMaster("local[*]")
    val sc = new SparkContext(conf)
    val spark = SparkSession.builder().appName("retail1").
      master("local[1*").getOrCreate()

    val dataRDD = sc.textFile("hour.csv")

    val dataRDDNoHeader = dataRDD.filter(line => line.split(",")(0) != "instant")
    val data0 = dataRDDNoHeader.map(line => {
      val arr = line.split(",")
      (arr(4).toInt, arr(16).toDouble)}).
      map(tuple => (tuple._1, (tuple._2, 1))).
      reduceByKey((a, b)=> (a._1+b._1, a._2+b._2)).sortByKey().
      map(mnthCnt => (mnthCnt._1, mnthCnt._2._1/mnthCnt._2._2)).
      map( a => (a._2, a._1)).sortByKey().collect.toList

    for(d <- data0) print(d._2+ "/ ")
    println("")
    for(d <- data0) print(d._1+ "/ ")
  }

  def allowForMnthDiffDays(mnthCnt: (Int, (Double, Double)))={

    if (mnthCnt._1 == 2) (2, mnthCnt._2._1/(mnthCnt._2._2 * 28.5))  //2011 is a leap year
    if (mnthCnt._1 == 4 || mnthCnt._1 == 6 || mnthCnt._1 == 9 || mnthCnt._1 == 11)
      (mnthCnt._1, mnthCnt._2._1/(mnthCnt._2._2 * 30))
    else (mnthCnt._1, mnthCnt._2._1/(mnthCnt._2._2 * 31))
    //data0.take(12).foreach(println)
  }
  /*
  output:
  1/ 2/ 12/ 3/ 11/ 4/ 10/ 5/ 7/ 8/ 6/ 9/
94.42477256822953/ 112.86502609992543/ 142.30343897505057/ 155.41072640868975/ 177.33542101600557/
187.26096033402922/ 222.15851137146797/ 222.90725806451613/ 231.81989247311827/ 238.09762711864406/
240.51527777777778/ 240.7731384829506/
   */
}