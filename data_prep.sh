#!/bin/bash
# download wget with homebrew
brew install wget
# make temporary directory to download the data
# cd into the directory and download, unzip
mkdir temp                         
cd temp                            
wget http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip
unzip trainingandtestdata.zip   
# delete the zip file and the test data
# the reason why I'm not using this test data is because
# the training data only has two target labels: negative, positive
# but the test data has three target labels: negative, neutral, positive
# to simplify I won't do any neutral class classification with predicted probabilities                               
rm trainingandtestdata.zip test*.csv
# go back to the parent directory and run train_test_split.py
cd ..
python train_test_split.py
# delete temporary directory and the files in it
rm -rf temp
# upload training_data and test_data created by train_test_split.py to Google Cloud Storage
gsutil cp pyspark_sa_train_data.csv gs://${PROJECT_ID}/pyspark_nlp/data/training_data.csv
gsutil cp pyspark_sa_test_data.csv gs://${PROJECT_ID}/pyspark_nlp/data/test_data.csv
# remove the training_data and test_data after uploading
rm pyspark_sa_train_data.csv pyspark_sa_test_data.csv