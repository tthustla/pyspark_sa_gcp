#import libraries
import sys
import pyspark as ps
import warnings
import re
from pyspark.sql import functions as f
from pyspark.sql import types as t
from pyspark.sql.types import StringType
from pyspark.ml.feature import Tokenizer, NGram, CountVectorizer, IDF, StringIndexer, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import PipelineModel

# retrieve command line arguments and store them as variables
inputdir = sys.argv[1]
outputfile = sys.argv[2]
modeldir = sys.argv[3]

#define regex pattern for preprocessing
pat1 = r'@[A-Za-z0-9_]+'
pat2 = r'https?://[^ ]+'
combined_pat = r'|'.join((pat1,pat2))
www_pat = r'www.[^ ]+'
negations_dic = {"isn't":"is not", "aren't":"are not", "wasn't":"was not", "weren't":"were not",
                "haven't":"have not","hasn't":"has not","hadn't":"had not","won't":"will not",
                "wouldn't":"would not", "don't":"do not", "doesn't":"does not","didn't":"did not",
                "can't":"can not","couldn't":"could not","shouldn't":"should not","mightn't":"might not",
                "mustn't":"must not"}
neg_pattern = re.compile(r'\b(' + '|'.join(negations_dic.keys()) + r')\b')

# preprocessing for
# first_process: to remove Twitter handle and URL
# second_process: to remove URL pattern starting with www.
# third_process: to lower characters
# fourth_process: to replace contracted negation with proper forms
# result: remove numbers and special characters
def pre_processing(column):
    first_process = re.sub(combined_pat, '', column)
    second_process = re.sub(www_pat, '', first_process)
    third_process = second_process.lower()
    fourth_process = neg_pattern.sub(lambda x: negations_dic[x.group()], third_process)
    result = re.sub(r'[^A-Za-z ]','',fourth_process)
    return result.strip()

# build a pipeline following below order
# tokenizer: split a tweet into individual words
# ngrams: create n-gram representation of words. Here it's set to trigram
# cv: turn n-gram representaion to a sparse representaion of the token counts. Below 5,460 is used as vocabulary size
# idf: calculate inverse document frequency from the result of the previous step 
#      to diminishes the weight of terms that occur very frequently in the document set 
#      and increases the weight of terms that occur rarely.
# assembler: transform the result of previous step to a single feature vector
# label_stringIdx: encode target labels to a column of label indices. 
#                  The indices are ordered by label frequencies, so the most frequent label gets index 0.
# lr: fit logistic regression with 'features' and 'label'
def build_pipeline():
	tokenizer = [Tokenizer(inputCol='tweet',outputCol='words')]
	ngrams = [NGram(n=i, inputCol='words', outputCol='{0}_grams'.format(i)) for i in range(1,4)]
	cv = [CountVectorizer(vocabSize=5460, inputCol='{0}_grams'.format(i), outputCol='{0}_tf'.format(i)) for i in range(1,4)]
	idf = [IDF(inputCol='{0}_tf'.format(i), outputCol='{0}_tfidf'.format(i), minDocFreq=5) for i in range(1,4)]
	assembler = [VectorAssembler(inputCols=['{0}_tfidf'.format(i) for i in range(1,4)], outputCol='features')]
	label_stringIdx = [StringIndexer(inputCol='sentiment', outputCol='label')]
	lr = [LogisticRegression(maxIter=100)]
	pipeline = Pipeline(stages=tokenizer+ngrams+cv+idf+assembler+label_stringIdx+lr)
	return pipeline

# below main function can be use for either first training or getting predictions with a loaded model
# first retrieve data
# apply pre-processing by making the above defined pre_processing function to a user defined function
# either build the pipeline from the above build_pipeline function and train or use a loaded pipeline model
# make predictions on the test set
# output the pipeline model, Spark dataframe of the predictions, and the prediction accuracy on the test set
def main(sqlc,input_dir,loaded_model=None):
	print('retrieving data from {}'.format(input_dir))
	if not loaded_model:
		train_set = sqlContext.read.format('com.databricks.spark.csv').options(header='true', inferschema='true').load(input_dir+'training_data.csv')
	test_set = sqlContext.read.format('com.databricks.spark.csv').options(header='true', inferschema='true').load(input_dir+'test_data.csv')
	print('preprocessing data...')
	reg_replaceUdf = f.udf(pre_processing, t.StringType())
	if not loaded_model:
		train_set = train_set.withColumn('tweet', reg_replaceUdf(f.col('text')))
	test_set = test_set.withColumn('tweet', reg_replaceUdf(f.col('text')))
	if not loaded_model:
		pipeline = build_pipeline()
		print('training...')
		model = pipeline.fit(train_set)
	else:
		model = loaded_model
	print('making predictions on test data...')
	predictions = model.transform(test_set)
	accuracy = predictions.filter(predictions.label == predictions.prediction).count() / float(test_set.count())
	return model, predictions, accuracy


if __name__=="__main__":
	# create a SparkContext while checking if there is already SparkContext created
	try:
	    sc = ps.SparkContext()
	    sc.setLogLevel("ERROR")
	    sqlContext = ps.sql.SQLContext(sc)
	    print('Created a SparkContext')
	except ValueError:
	    warnings.warn('SparkContext already exists in this scope')
	# build pipeline, fit the model and retrieve the outputs by running main() function
	pipelineFit, predictions, accuracy = main(sqlContext,inputdir)
	print('predictions finished!')
	print('accuracy on test data is {}'.format(accuracy))
	# select the original target label 'sentiment', 'text' and 'label' created by label_stringIdx in the pipeline
	# model predictions. Save it as a single CSV file to a destination specified by the second command line argument
	print('saving predictions to {}'.format(outputfile))
	predictions.select(predictions['sentiment'],predictions['text'],predictions['label'],predictions['prediction']).coalesce(1).write.mode("overwrite").format("com.databricks.spark.csv").option("header", "true").csv(outputfile)
	# save the trained model to destination specified by the third command line argument
	print('saving model to {}'.format(modeldir))
	pipelineFit.save(modeldir)
	# Load the saved model and make another predictions on the same test set
	# to check if the model was properly saved
	loadedModel = PipelineModel.load(modeldir)
	_, _, loaded_accuracy = main(sqlContext,inputdir,loadedModel)
	print('accuracy with saved model on test data is {}'.format(loaded_accuracy))
	sc.stop()
    