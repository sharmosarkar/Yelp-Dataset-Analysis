import os, numbers
import json_to_csv_converter as convert
import pandas as pd
import constants


REVIEW_DATA_FILE =constants.REVIEW_DATA_FILE
BUSINESS_DATA_FILE = constants.BUSINESS_DATA_FILE
BUSINESS_COLUMNS = constants.BUSINESS_COLUMNS
REVIEW_COLUMNS = constants.REVIEW_COLUMNS



## read datasets and filter on conditions
import numpy as np
import sklearn.cluster
import distance
import pandas

df = pd.read_csv('new_business.csv', encoding='ISO-8859-1')
category_list = df['categories'].tolist()

for cat in category_list:
	#print (type(cat))
	#words = cat
	text_file = open("Output.txt", "w+")
	words = cat.replace('[','').split(" ") #Replace this line
	words = np.asarray(words) #So that indexing with a list will work
	lev_similarity = -1*np.array([[distance.levenshtein(w1,w2) for w1 in words] for w2 in words])

	affprop = sklearn.cluster.AffinityPropagation(affinity="precomputed", damping=0.5)
	affprop.fit(lev_similarity)
	for cluster_id in np.unique(affprop.labels_):
		try:
		    exemplar = words[affprop.cluster_centers_indices_[cluster_id]]
		    cluster = np.unique(words[np.nonzero(affprop.labels_==cluster_id)])
		    cluster_str = ", ".join(cluster)
		    print(" - *%s:* %s" % (exemplar, cluster_str))
		    text_file.write(" - *%s:* %s\n" % (exemplar, cluster_str))
		    text_file.write()
		except TypeError:
			continue
text_file.close()




^^^()()^^^