# import libraries
import pandas as pd
import numpy as np

# set the names for each column
cols = ['sentiment','id','date','query_string','user','text']
def main():
	# read training data with ISO-8859-1 encoding and column names set above
	df = pd.read_csv('temp/training.1600000.processed.noemoticon.csv', encoding = 'ISO-8859-1',names=cols)
	# shuffle the data
	df = df.sample(frac=1).reset_index(drop=True)
	# set the random seed and split train and test with 99 to 1 ratio
	np.random.seed(777)
	msk = np.random.rand(len(df)) < 0.99
	train = df[msk].reset_index(drop=True)
	test = df[~msk].reset_index(drop=True)
	# save both train and test as CSV files
	train.to_csv('pyspark_sa_train_data.csv')
	test.to_csv('pyspark_sa_test_data.csv')

if __name__=="__main__":
	main()
