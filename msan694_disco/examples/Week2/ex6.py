from pyspark import SparkConf, SparkContext

#Create SparkContext
conf = SparkConf().setMaster("local[*]").setAppName("read_lines")
sc = SparkContext(conf = conf)

#Load Data.
lines=sc.textFile("../Data/README.md")
print(lines.count())
print(lines.first())

sc.stop()
