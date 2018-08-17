import org.apache.spark.sql._
import org.apache.spark.{SparkConf, SparkContext}

object BikeSharingHourStudyColumn_9thhum {

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
      (arr(12).toDouble, arr(16).toDouble)}).
      map(tuple => (tuple._1, (tuple._2, 1))).
      reduceByKey((a, b)=> (a._1+b._1, a._2+b._2)).
      map(humCount => (humCount._1, humCount._2._1/humCount._2._2)).  //1st output
      map( a => (a._2, a._1)).sortByKey().collect.toList
      //map(yrCount => (yrCount._1, yrCount._2._1/yrCount._2._2)).sortByKey().collect.toList  //second output

    for(d <- data0) print(d._2+ "/ ")
    println("")
    for(d <- data0) print(d._1+ "/ ")
  }
/*
output:  201808151158
1st:
0.13/ 0.14/ 0.0/ 0.12/ 0.86/ 0.97/ 0.93/ 1.0/ 0.8/ 0.15/ 0.92/ 0.08/ 0.87/ 0.75/ 0.94/ 0.1/ 0.81/ 0.88/ 0.91/ 0.76/ 0.84/ 0.96/ 0.89/ 0.82/
0.71/ 0.83/ 0.56/ 0.9/ 0.79/ 0.77/ 0.67/ 0.69/ 0.7/ 0.78/ 0.72/ 0.63/ 0.6/ 0.65/ 0.59/ 0.85/ 0.68/ 0.64/ 0.73/ 0.61/ 0.47/ 0.66/ 0.44/ 0.74/
0.5/ 0.57/ 0.52/ 0.53/ 0.49/ 0.54/ 0.51/ 0.62/ 0.16/ 0.48/ 0.41/ 0.55/ 0.58/ 0.45/ 0.46/ 0.42/ 0.4/ 0.25/ 0.21/ 0.28/ 0.33/ 0.43/ 0.19/
.38/ 0.35/ 0.37/ 0.39/ 0.32/ 0.26/ 0.29/ 0.36/ 0.31/ 0.3/ 0.23/ 0.34/ 0.18/ 0.24/ 0.22/ 0.17/ 0.27/ 0.2/
17.0/ 19.0/ 28.318181818181817/ 29.0/ 60.86842105263158/ 64.0/ 66.38066465256797/ 68.77407407407408/ 72.33644859813084/ 73.0/ 76.0/ 77.0/
86.19877049180327/ 88.44144144144144/ 93.93035714285715/ 107.0/ 107.91636363636364/ 115.07458143074581/ 119.0/ 123.05022831050228/
124.71774193548387/ 125.0/ 126.26359832635983/ 129.68227424749165/ 147.59067357512953/ 148.03968253968253/ 160.31935483870967/
160.85714285714286/ 161.47899159663865/ 161.89285714285714/ 165.26708074534162/ 165.40389972144845/ 167.58372093023254/
168.914373088685/ 169.32984293193718/ 171.98773006134968/ 173.34831460674158/ 175.4625322997416/ 178.09926470588235/ 182.6/
189.0174418604651/ 191.3835616438356/ 191.74763406940062/ 193.63988095238096/ 195.7327935222672/ 198.24742268041237/ 199.35245901639345/
200.84457478005865/ 207.91353383458647/ 212.39826839826839/ 212.54807692307693/ 219.36704119850188/ 219.50152905198777/ 224.4773519163763/
231.45419847328245/ 234.23076923076923/ 241.6/ 242.74583333333334/ 244.56206896551726/ 244.9034090909091/ 246.80232558139534/
249.7983870967742/ 250.8006329113924/ 251.331914893617/ 259.0357142857143/ 259.6949152542373/ 261.0/ 261.680412371134/ 269.65432098765433/
270.3037037037037/ 274.6875/ 283.73118279569894/ 283.86503067484665/ 284.875/ 286.77033492822966/ 287.3838383838384/ 288.0/
294.0660377358491/ 296.72727272727275/ 297.91525423728814/ 300.8141592920354/ 304.8478260869565/ 310.7142857142857/ 330.1/
334.05357142857144/ 334.14814814814815/ 340.4/ 352.90140845070425/ 398.0/
2nd
28.318181818181817/ 77.0/ 107.0/ 29.0/ 17.0/ 19.0/ 73.0/ 241.6/ 340.4/ 330.1/ 274.6875/ 398.0/ 261.0/ 334.14814814814815/
304.8478260869565/ 334.05357142857144/ 259.6949152542373/ 288.0/ 352.90140845070425/ 261.680412371134/ 294.0660377358491/ 300.8141592920354/
297.91525423728814/ 287.3838383838384/ 269.65432098765433/ 310.7142857142857/ 283.86503067484665/ 296.72727272727275/ 284.875/
283.73118279569894/ 286.77033492822966/ 259.0357142857143/ 244.56206896551726/ 251.331914893617/ 270.3037037037037/ 199.35245901639345/
249.7983870967742/ 250.8006329113924/ 195.7327935222672/ 242.74583333333334/ 219.50152905198777/ 207.91353383458647/ 231.45419847328245/
212.54807692307693/ 219.36704119850188/ 224.4773519163763/ 244.9034090909091/ 160.31935483870967/ 212.39826839826839/ 246.80232558139534/
178.09926470588235/ 173.34831460674158/ 193.63988095238096/ 234.23076923076923/ 171.98773006134968/ 191.3835616438356/ 175.4625322997416/
198.24742268041237/ 165.26708074534162/ 189.0174418604651/ 165.40389972144845/ 167.58372093023254/ 147.59067357512953/ 169.32984293193718/
191.74763406940062/ 200.84457478005865/ 88.44144144144144/ 123.05022831050228/ 161.89285714285714/ 168.914373088685/ 161.47899159663865/
72.33644859813084/ 107.91636363636364/ 129.68227424749165/ 148.03968253968253/ 124.71774193548387/ 182.6/ 60.86842105263158/
86.19877049180327/ 115.07458143074581/ 126.26359832635983/ 160.85714285714286/ 119.0/ 76.0/ 66.38066465256797/ 93.93035714285715/
125.0/ 64.0/ 68.77407407407408/
0.0/ 0.08/ 0.1/ 0.12/ 0.13/ 0.14/ 0.15/ 0.16/ 0.17/ 0.18/ 0.19/ 0.2/ 0.21/ 0.22/ 0.23/ 0.24/ 0.25/ 0.26/ 0.27/ 0.28/ 0.29/ 0.3/ 0.31/ 0.32/
0.33/ 0.34/ 0.35/ 0.36/ 0.37/ 0.38/ 0.39/ 0.4/ 0.41/ 0.42/ 0.43/ 0.44/ 0.45/ 0.46/ 0.47/ 0.48/ 0.49/ 0.5/ 0.51/ 0.52/ 0.53/ 0.54/ 0.55/
0.56/ 0.57/ 0.58/ 0.59/ 0.6/ 0.61/ 0.62/ 0.63/ 0.64/ 0.65/ 0.66/ 0.67/ 0.68/ 0.69/ 0.7/ 0.71/ 0.72/ 0.73/ 0.74/ 0.75/ 0.76/ 0.77/ 0.78/
0.79/ 0.8/ 0.81/ 0.82/ 0.83/ 0.84/ 0.85/ 0.86/ 0.87/ 0.88/ 0.89/ 0.9/ 0.91/ 0.92/ 0.93/ 0.94/ 0.96/ 0.97/ 1.0/
*/
}