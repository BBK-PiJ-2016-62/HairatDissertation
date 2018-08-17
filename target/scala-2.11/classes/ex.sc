import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.regression.LinearRegression

import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)

val spark = SparkSession.builder().master("local[*]").getOrCreate()
import spark.implicits._

val data = spark.read.option("header", "true").
  option("inferSchema", "true").
  format("csv").
  load("D:\\hairat dataset\\train datasets.csv")

val data1 = data.drop("User_ID").drop("Product_ID")/*.
  drop("Product_Category_1").drop("Product_Category_2").
  drop("Product_Category_3")*/

val df0 = data1.map(row => {
  val gender0 = row.getAs[String]("Gender")
  val gender = if (gender0 == "F") 1.0 else 2.0
  val age0 = row.getAs[String]("Age")
  val age = {if (age0 == "0-17") 1.0; else if (age0 == "18-25") 2.0;
              else if (age0 == "26-35") 3.0; else if (age0 == "36-45") 4.0;
              else if (age0 == "46-50" || age0 == "51-55") 5.0; else 6.0}
  val city_category0 = row.getAs[String]("City_Category")
  val city_category = {if (city_category0 == "A") 1.0;
                else if (city_category0 == "B") 2.0; else 3.0}
  val stayInCurCYrs0 = row.getAs[String]("Stay_In_Current_City_Years")
  val stayInCurCYrs = {if (stayInCurCYrs0 == "4+") 5.0 else (stayInCurCYrs0.toDouble+1.0) }
  (gender, age, row.getAs[Int]("Occupation").toDouble, city_category, stayInCurCYrs,
  (row.getAs[Int]("Marital_Status").toDouble+1.0),
  row.getAs[Int]("Purchase").toDouble)
})


case class TydiedData(Gender: Double, Age: Double, Occuplation: Double, City_Category: Double,
                      Stay_In_Current_City_Years: Double, Marital_Status: Double, Purchase: Double)

val df1 = df0.as[TydiedData]
df0.printSchema()
df0.show

/*import df0.sparkSession.implicits._
import org.apache.spark.sql.functions.col
val df = df0.select(col("_7").as("label"),col("_1"),col("_2"),
           col("_3),col("_4"),col("_5"),col("_6"))*/
/*
val assembler = (new VectorAssembler().setInputCols(Array("_1","_2","_3","4","_5","_6")).
                  setOutputCol("features"))

val output = assembler.transform(df).select($"label", $"features")*/




