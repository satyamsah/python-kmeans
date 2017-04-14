#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""
A K-means clustering program using MLlib.
This example requires NumPy (http://www.numpy.org/).
"""
from __future__ import print_function

import sys

import numpy as np
from numpy import array
from pyspark import SparkContext
from pyspark.mllib.clustering import KMeans


#def parseVectorAndDropLastColumn(line):
#    return np.array([float(x) for x in line.split(',')])


def parseVectorAndDropLastColumn(line):
    arr=[]
    for x in range(len(line.split(','))-1):
        arr.append(float(line.split(',')[x]))
    return np.array(arr)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: kmeans <file> <k>", file=sys.stderr)
        exit(-1)
    sc = SparkContext(appName="KMeans")
    lines = sc.textFile(sys.argv[1])
    data = lines.map(parseVectorAndDropLastColumn)
    k = int(sys.argv[2])
    model = KMeans.train(data, k)
    with open(sys.argv[1]) as file:
       rows=file.readlines();

    rowOfDataPoints=[]
    for row in rows:
       row=row.rstrip("\n")

       singleOutputDataPoint=[]
       #adding the column vectors
       for x in range(len(row.split(','))-1):
          singleOutputDataPoint.append(float(row.split(',')[x]))

       #adding predicted class value
       singleOutputDataPoint.append(float(model.predict(array(singleOutputDataPoint))))

       # adding real class value
       singleOutputDataPoint.append(float(row.split(',')[-1]))
       rowOfDataPoints.append(singleOutputDataPoint)

    a = np.asarray(rowOfDataPoints)
    np.savetxt("output.csv", a, delimiter=",",fmt="%1.3f")

    '''
    print("-----")
    print(model.predict(array([1,1,7,2,19,0,19,0,0,0,348,276,403,9,0,1,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,1,0,0,1,0,0,0,1,0,0,1,0,0,1,0,0,0,1,0,0,1,0,0,0,1,0,0,1,0,1,0,1,1,0,0,1,0,0,1,0,0,1,0,1,1,0,1,0,0,1,0])))
    print("-----")
    '''
    #print("Final centers: " + str(model.clusterCenters))

    print("1st cluster center"+str(model.clusterCenters[0]))
    print("2nd cluster center"+str(model.clusterCenters[1]))

    print("Total Cost: " + str(model.computeCost(data)))
sc.stop()
