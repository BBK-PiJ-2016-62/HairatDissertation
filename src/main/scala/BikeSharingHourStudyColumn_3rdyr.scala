import org.apache.spark.sql._
import org.apache.spark.{SparkConf, SparkContext}

object BikeSharingHourStudyColumn_3rdDropYr {

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
      (arr(3).toInt, arr(16).toDouble)}).
      map(tuple => (tuple._1, (tuple._2, 1))).
      reduceByKey((a, b)=> (a._1+b._1, a._2+b._2)).
      map(yrCount => (yrCount._1, yrCount._2._1/yrCount._2._2)).
      map( a => (a._2, a._1)).sortByKey().collect.toList

    for(d <- data0) print(d._2+ "/ ")
    println("")
    for(d <- data0) print(d._1+ "/ ")
  }
  /* 201808151135
  output:
  0/ 1/
143.79444765760556/ 234.6663613464621/
   */
}