/*import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.rdd._

val conf = new SparkConf().setAppName("retail").setMaster("local[*]")
val sc = new SparkContext(conf)

val dataRDD = sc.textFile("D:\\hairat dataset\\train datasets.csv")

val dataRDDNoEmpty = dataRDD.filter(line => line.split(",")(0) != "User_ID").
                          map(line => {val arr = line.split(",")
                              if(arr(9).isEmpty) arr(9)="0"
                              if(arr(10).isEmpty) arr(10)="0"
                              arr(9).toDouble; arr(10).toDouble
                             (arr(2), arr(3), arr(4), arr(5), arr(6),
                               arr(7), arr(8), arr(9), arr(10), arr(11))})
dataRDDNoEmpty.take(30).foreach(println)*/

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.{VectorAssembler, VectorIndexer}
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql._
import org.apache.spark.{SparkConf, SparkContext}
import scala.util.control.Breaks._

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
  ( /*rerangeMnthsAccordToCnt(arr(4))*/ arr(4).toDouble, arr(3).toDouble, arr(5).toDouble,//rerangeHrsAccordToCnt(arr(5)),
    arr(8).toDouble, arr(9).toDouble, arr(10).toDouble,
    arr(11).toDouble, arr(12).toDouble,
    /*rerangeWndSpAccordToCnt(arr(13))*/arr(13).toDouble, arr(16).toDouble)
}).filter(row => row._1 == 5.0).cache()

def sortHr(data: RDD[(Double, Double)]) = {
  val data0 = data.map(tuple => (tuple._1, (tuple._2, 1))).
    reduceByKey((a, b) => (a._1 + b._1, a._2 + b._2)).
    map(hrCount => (hrCount._1, hrCount._2._1 / hrCount._2._2)).
    map(a => (a._2, a._1)).sortByKey().collect.toList
  data0
}
def rerangeHrAccordToCnt1(wndSp: Double)(winSpList: List[(Double, Double)]): Double = {
  var a = 0.0
  var b = -1.0
  breakable {
    for (wspl <- winSpList) {
      b += 1.0
      if (wndSp == wspl._2) {
        a = b //wspl._1
        break()
      }
    }
  }
  a
}

def sortWindSp(data: RDD[(Double, Double)]) = {
  val data0 = data.map(tuple => (tuple._1, (tuple._2, 1))).
    reduceByKey((a, b) => (a._1 + b._1, a._2 + b._2)).
    map(wspCount => (wspCount._1, wspCount._2._1 / wspCount._2._2)).
    map(a => (a._2, a._1)).sortByKey().collect.toList
  data0
}

val sortHrList = sortHr(data00.map(row => (row._3, row._10)))
//val sortWindSpList = sortWindSp(data00.map(row => (row._9, row._10)))

for (sh <- sortHrList) println (sh._1 + " " + sh._2 + "/ ")
println("")
//for (sw <- sortWindSpList) println (sw._1 + " " + sw._2 + "/ ")

rerangeHrAccordToCnt1(0)(sortHr(data00.map(row => (row._3, row._10))))


/*
val data = dataRDD.filter(line => (line.split(",")(0) != "User_ID")).
  map(line => {val arr = line.split(",");
        (arr(0), (arr(2), arr(3), arr(4), arr(5), arr(6), arr(7), arr(11).toInt))}).
        reduceByKey((a,b)=>(a._1, a._2, a._3, a._4, a._5, a._6, a._7+b._7)).
  map(a => (a._2._1, a._2._2, a._2._3, a._2._4, a._2._5, a._2._6, a._2._7))
data.take(20).foreach(println)
*/

/*
val isNaN = udf((value : Float) => {
   if (value.equals(Float.NaN) || value == null) true else false })

val result = data.filter(isNaN(data("column2"))).count()
 */