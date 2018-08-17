import org.apache.spark.sql._
import org.apache.spark.{SparkConf, SparkContext}

object BikeSharingHourStudyColumn_6thweathersit {

  def main(args: Array[String]): Unit = {

    import org.apache.log4j._
    Logger.getLogger("org").setLevel(Level.ERROR)
    val data0 = produceList()

    for(d <- data0) print(d._2+ "/ ")
    println("")
    for(d <- data0) print(d._1+ "/ ")
    println(data0(1)._2)

  }
  def produceList()= {

    val conf = new SparkConf().setAppName("retail").setMaster("local[*]")
    val sc = new SparkContext(conf)
    val spark = SparkSession.builder().appName("retail1").
      master("local[1*").getOrCreate()

    val dataRDD = sc.textFile("hour.csv")

    val dataRDDNoHeader = dataRDD.filter(line => line.split(",")(0) != "instant")
    val data00 = dataRDDNoHeader.map(line => {
      val arr = line.split(",")
      (arr(9).toInt, arr(16).toDouble)
    }).
      map(tuple => (tuple._1, (tuple._2, 1))).
      reduceByKey((a, b) => (a._1 + b._1, a._2 + b._2))

    val data0 = data00.map(weatherCount => (weatherCount._1, weatherCount._2._1 / weatherCount._2._2)).
      map(a => (a._2, a._1)).sortByKey().collect.toList

    data0
  }

  /*
  output:  201808151158
  4/ 3/ 2/ 1/
74.33333333333333/ 111.57928118393235/ 175.16549295774647/ 204.8692718829405/
   */
}