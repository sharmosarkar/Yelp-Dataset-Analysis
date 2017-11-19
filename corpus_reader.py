import os, numbers
import json_to_csv_converter as convert
import pandas as pd
import constants


REVIEW_DATA_FILE =constants.REVIEW_DATA_FILE
BUSINESS_DATA_FILE = constants.BUSINESS_DATA_FILE
BUSINESS_COLUMNS = constants.BUSINESS_COLUMNS
REVIEW_COLUMNS = constants.REVIEW_COLUMNS


def read_corpus_json(data_folder):
	b = c =''
	business_file = os.path.join(data_folder, BUSINESS_DATA_FILE) 
	review_file = os.path.join(data_folder, REVIEW_DATA_FILE) 
	#print (review_file, business_file)
	def make_clean_df (filename,column_lst):
		filename = os.path.join(data_folder,filename)
		df = pd.read_csv(filename, encoding='ISO-8859-1')
		required_df = pd.DataFrame(df, columns=column_lst)
		for col in column_lst:
			if not isinstance(required_df[col][0], numbers.Number) :
				required_df[col] = required_df[col].map(lambda x: x.replace("b'", "").replace(']', "").replace('b"','').replace('"', '').replace("'", '').replace("\n", '').replace("\t", ''))
		return required_df

		#print (os.path.join(data_folder,convert.make_csv(business_file)))
	business_df = make_clean_df('new_business.csv', BUSINESS_COLUMNS)
	review_df = make_clean_df('new_review.csv', REVIEW_COLUMNS)
	return business_df,review_df


'''
## read datasets and filter on conditions
def read_corpus_json_filter(data_folder):
	business_file = os.path.join(data_folder, 'business.json') 
	review_file = os.path.join(data_folder, 'review.json') 
	def make_clean_df (filename,column_lst):
		filename = os.path.join(data_folder,filename)
		df = pd.read_csv(filename, encoding='ISO-8859-1')
		required_df = pd.DataFrame(df, columns=column_lst)
		for col in column_lst:
			if not isinstance(required_df[col][0], numbers.Number) :
				required_df[col] = required_df[col].map(lambda x: x.replace("b'", "").replace(']', "").replace('b"','').replace('"', '').replace("'", '').replace("\n", '').replace("\t", ''))
		return required_df
	
	#print (os.path.join(data_folder,convert.make_csv(business_file)))
	business_df = make_clean_df(convert.make_csv(business_file), BUSINESS_COLUMNS)
	review_df = make_clean_df(convert.make_csv(review_file), REVIEW_COLUMNS)

	#print (os.path.join(data_folder,convert.make_csv(business_file)))
	business_df = make_clean_df(convert.make_csv(business_file), BUSINESS_COLUMNS)
	review_df = make_clean_df(convert.make_csv(review_file), REVIEW_COLUMNS)

	new_business_df = business_df.loc[business_df['city'] == 'Madison']
	new_review_df = review_df.loc[review_df['business_id'].isin(new_business_df['business_id'].tolist())]
	new_business_df.to_csv('Z:/Yelp_Data_Challenge/data/new_business.csv', sep='\t', encoding='utf-8')
	new_review_df.to_csv('Z:/Yelp_Data_Challenge/data/new_review.csv', sep='\t', encoding='utf-8')
'''



if __name__ == "__main__":
    read_corpus_json_filter()





