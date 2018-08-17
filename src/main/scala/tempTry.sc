import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.SparkSession

val conf = new SparkConf().setAppName("retail").setMaster("local[*]")
val sc = new SparkContext(conf)
val spark = SparkSession.builder().appName("retail1").
  master("local[1*").getOrCreate()
import spark.implicits._






val someDF = Seq((8, "bat"),(64, "mouse"), (10, "cat"),(4, "dog")).
  toDF("number", "word").show