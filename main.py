import random as rand
from clustering import clustering
from point import Point
import csv

geo_locs = []
factor=-1
#loc_ = Point(0.0, 0.0)  #tuples for location
#geo_locs.append(loc_)
#read the fountains location from the csv input file and store each fountain location as a Point(latit,longit) object
#f = open('/home/kazem/Downloads/Hackathon/drinkingFountains.csv', 'r')
f = open('arc.csv', 'r')
reader = csv.reader(f, delimiter=",")
for line in reader:
    loc_ = Point(float(line[0]), float(line[1]))  #tuples for location
    geo_locs.append(loc_)
#print len(geo_locs)
#for p in geo_locs:
#    print "%f %f" % (p.latit, p.longit)
#let's run k_means clustering. the second parameter is the no of clusters
cluster = clustering(geo_locs, 8 )
Score = cluster.Calculating_k_means(False)

precision, recall, fscore, support = Score(y_test, predicted)

print('precision: {}'.format(precision))
print('recall: {}'.format(recall))
print('fscore: {}'.format(fscore))
print('support: {}'.format(support))
if Score == factor:
    print("Error in arguments!");
else:
    #the clustering results is a list of lists where each list represents one cluster
    print ("clustering results:")
    cluster.print_clusters(cluster.clusters)