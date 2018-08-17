import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql._
import org.apache.log4j.{Level, Logger}

object abc {

  def main(args: Array[String]): Unit = {

    Logger.getLogger("org").setLevel(Level.ERROR)

    val conf = new SparkConf().setAppName("retail").setMaster("local[*]")
    val sc = new SparkContext(conf)
    val spark = SparkSession.builder().appName("retail1").master("local[1*").getOrCreate()
    import spark.implicits._

    val dataRDD = sc.textFile("D:\\hairat dataset\\train datasets.csv")

    val data = dataRDD.filter(line => line.split(",")(0) != "User_ID").
      map(line => ( if (line.split(",")(2) == "F") 1.0 else 2.0,
          {val age = line.split(",")(3)
          if (age == "0-17") 1.0; else if (age == "18-25") 2.0
          else if (age == "26-35") 3.0; else if (age == "36-45") 4.0
          else if (age == "46-50" || age == "51-55") 5.0; else 6.0},
          line.split(",")(4).toDouble,
          {val cityCat = line.split(",")(5)
          if (cityCat == "A") 1.0
          else if (cityCat == "B") 2.0; else 3.0},
          {val stayCCityYr = line.split(",")(6)
          if (stayCCityYr == "4+") 5.0 else stayCCityYr.toDouble+1.0},
          if (line.split(",")(7) == "0") 1.0 else 2.0,
          line.split(",")(11).toDouble)
      )

    val data1 = data.toDF()
    data1.printSchema()
    data1.show
  }
}
