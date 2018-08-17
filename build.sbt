name := "HairatDissertation"

version := "1.0"

val sparkVersion = "2.3.1"

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % sparkVersion,
  "org.apache.spark" %% "spark-sql" % sparkVersion,
  "org.apache.spark" %% "spark-mllib" % sparkVersion
)

resolvers ++= Seq(
  "Apache Repository" at "https://repository.apache.org/content/repositories/releases"
)

scalaVersion := "2.11.8"