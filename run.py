import corpus_reader,lda,utils,scores
import pandas as pd
import optparse
import numpy as np
# import recommendation as reco
import svm_lib



def main():
	parser = optparse.OptionParser()
	parser.add_option("-d", dest="data_folder", help="data location")
	(options, args) = parser.parse_args()

	if not options.data_folder:
		parser.error("need data locations(-d)")

	business_df , review_df = corpus_reader.read_corpus_json(options.data_folder)
	#print (review_df.get_value(0,'business_id'))
	### options.K, options.alpha, options.beta, docs, voca.size()
	K = 20
	alpha = beta = 0.5 
	review_lst = review_df['text'].tolist()
	tagged_review_list = lda.run_lda(K, alpha, beta, review_lst)
	tagged_review_dict = utils.lst_to_dict (tagged_review_list)
	temporary_review_df = pd.DataFrame(tagged_review_dict)
	tagged_review_df = pd.merge(left=review_df,right=temporary_review_df)
	#print (tagged_review_df)
	# print (tagged_review_df.shape)
	sentiment_score_df = pd.DataFrame(scores.calculate_sentiment_score(review_lst))
	tagged_review_df = pd.merge(left=tagged_review_df,right=sentiment_score_df)
	#print (tagged_review_df)
	
	## Calculating Topicwise Stars
	max_star = pd.DataFrame(tagged_review_df.groupby(['business_id'], sort=False, as_index=False)['stars'].max())
	min_star = pd.DataFrame(tagged_review_df.groupby(['business_id'], sort=False, as_index=False)['stars'].min())
	tagged_review_df = tagged_review_df.join(max_star, on='business_id' , rsuffix='_max')
	tagged_review_df = tagged_review_df.join(min_star, on='business_id' , rsuffix='_min')
	
	## making additional columns to hold Topic Stars
	topic_names = list(tagged_review_df['topics'][0].keys())
	for topic in topic_names:
		tagged_review_df[topic] = tagged_review_df.apply(lambda row: scores.calculate_topic_star(row,topic), axis=1)

	## aggregating on business level (every business now should have some star rating for every Topic)
	df_1 = pd.DataFrame()
	for topic in topic_names:
		topic_avg_star = pd.DataFrame(np.ceil(tagged_review_df.groupby(['business_id'], sort=False)[topic].mean()).abs())
		df_1 = df_1.append(topic_avg_star)
	a = df_1.groupby(df_1.index).max()
	business_df = business_df.join(a,on='business_id')
	print (business_df)

	## Start of Prediction (Predicting Yelp Stars)
	category_list = list(set(business_df['categories'].tolist()))
	city_list = list(set(business_df['city'].tolist()))
	cat_numerical = []
	city_numerical = []
	business_df['categories'].map(lambda x : cat_numerical.append(category_list.index(x)))
	business_df['city'].map(lambda x : city_numerical.append(city_list.index(x)))
	#print (cat_numerical , city_numerical)
	mm = pd.DataFrame({'city_num':city_numerical})
	bb = pd.DataFrame({'cat_num':cat_numerical})
	cols = ['review_count',0,1,2]
	# df = pd.DataFrame(business_df['city'].replace({'Dravosburg':1},regex=True))
	reco_X_df = business_df[cols]
	reco_X_df['city_num'] = mm['city_num']
	reco_X_df['cat_num']= bb['cat_num']
	reco_Y_df = pd.DataFrame(business_df['stars']) 	## Stars to predict
	print(reco_X_df , reco_Y_df)
	svm_lib.initialize_SVM(reco_X_df,reco_Y_df)



if __name__ == "__main__":
    main()




^^^()()^^^