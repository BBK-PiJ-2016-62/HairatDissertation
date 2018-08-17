import org.apache.spark.rdd._
import org.apache.spark.sql._
import org.apache.spark.{SparkConf, SparkContext}

object BikeSharingHourStudyColumn_1sthr {

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
      (/*arr(4).toInt, */arr(5).toInt, arr(16).toDouble)}).//filter(row => row._1 == 5).map(row => (row._2, row._3)).
      map(tuple => (tuple._1, (tuple._2, 1))).
      reduceByKey((a, b)=> (a._1+b._1, a._2+b._2)).
      map(hourCount => (hourCount._1, hourCount._2._1/hourCount._2._2)).
      map( a => (a._2, a._1)).sortByKey().collect.toList

    for(d <- data0) print(d._2+ "/ ")
    println("")
    for(d <- data0) print(d._1+ "/ ")
  }
  /*
  output:
  4/ 3/ 5/ 2/ 1/ 0/ 6/ 23/ 22/ 21/ 10/ 11/ 7/ 9/ 20/ 14/ 15/ 12/ 13/ 19/ 16/ 8/ 18/ 17/
6.352941176470588/ 11.727403156384504/ 19.88981868898187/ 22.86993006993007/ 33.3756906077348/
53.89807162534435/ 76.04413793103448/ 87.83104395604396/ 131.33516483516485/ 172.31456043956044/
173.6685006877579/ 208.1430536451169/ 212.0646492434663/ 219.30949105914718/ 226.03021978021977/
240.94924554183814/ 251.2331961591221/ 253.31593406593407/ 253.66117969821673/ 311.52335164835165/
311.9835616438356/ 359.01100412654745/ 425.510989010989/ 461.45205479452056/
   */
}