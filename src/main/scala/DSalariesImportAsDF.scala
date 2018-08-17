import org.apache.spark.sql.SparkSession

object HelloScala {

  def main(args: Array[String]): Unit = {

    import org.apache.log4j._
    Logger.getLogger("org").setLevel(Level.ERROR)

    val spark = SparkSession.builder().appName("abc").master("local[*]").getOrCreate()

    val data = spark.read.option("header", "true").
      option("inferSchema", "true").
      csv("D:\\Salaries.csv")
  }

}
