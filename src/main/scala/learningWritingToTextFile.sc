import java.io.{BufferedWriter, File, FileWriter}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.SparkSession

val conf = new SparkConf().setAppName("retail").setMaster("local[*]")
val sc = new SparkContext(conf)
val spark = SparkSession.builder().appName("retail1").
  master("local[*]").getOrCreate()
import spark.implicits._


val dataCheck = Seq((28.6826451, 11.0, 0.0, 2.0, 1.0, 0.0, 31.0, 37.0, 41.0, 24.0)).
  toDF("label", "_1", "_2", "_3", "_4", "_5", "_6", "_7", "_8", "_9")

def writeToHairatMLProject(){
val a = "haha"
val file = new File("D:\\abcdefgh.txt")
val bw = new BufferedWriter(new FileWriter(file, true))
bw.write(dataCheck.show.toString)
bw.close()
}
//dataCheck.write.mode("Append").format("text").save("D\\abcdefgh.txt")
//writeToHairatMLProject()
dataCheck.write.text("D:\\abcdefgh.txt")
dataCheck.show
def ooo()={
  val a = 5 + 3
  " "+ a.toString
}