from pyspark import SparkContext, SparkConf

conf = SparkConf().setMaster("spark://ip-172-31-17-229.us-west-2.compute.internal:7077").setAppName("Woodbridge_Diane")
sc = SparkContext(conf = conf)

print  sc.parallelize([1,2,3,4]).mean()

print sc.textFile("file:///root/example/input_2.txt").count()

sc.stop()
