import constants

from textblob import TextBlob
MAX_STAR = constants.MAXIMUM_STAR
TOLERANCE_FACTOR = constants.TOLERANCE_FACTOR

def calculate_sentiment_score(review_list):
	sentiment_score = []
	for review in review_list:
		blob = TextBlob(review)
		score = blob.sentiment.polarity
		sentiment_score.append(score)
	#print (sentiment_score)
	return {'text': review_list, 'sentiment_score':sentiment_score}


def calculate_topic_star(row,topic):
	topic_score_dict = row['topics']
	score = topic_score_dict[topic]
	topic_star = row['sentiment_score']*row['stars']*score * MAX_STAR 	
	#print('topic',topic,'\t',topic_star)
	return topic_star




^^^()()^^^