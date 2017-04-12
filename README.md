# python-kmeans

 spark-submit --master yarn --deploy-mode client --executor-memory 1g --name pysparkclustering --conf "spark.app.id=pysparkclustering" pysparkclustering.py hdfs://192.168.0.102:8020/wordcount/all_data-comma.txt 2 > output-kmeans-out.2.txt
