#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time
from pyspark.sql import SparkSession
from lightfm import LightFM
from lightfm.evaluation import precision_at_k, recall_at_k, auc_score
from scipy.sparse import coo_matrix
import numpy as np
from sklearn import preprocessing

def main(spark, training_data_file, validation_data_file, loss_functions, num_components, epoch_range):
    
    #Read in our training and validation data files
    train_df_pre = spark.read.parquet('./'+training_data_file)
    train_df_pre.createOrReplaceTempView('train_df_pre')
    validation_df_pre_pre = spark.read.parquet('./'+validation_data_file)
    validation_df_pre_pre.createOrReplaceTempView('validation_df_pre_pre')
    
    #Make sure validation only contains membership that training contains
    validation_df_pre = spark.sql('SELECT v.user_id, v.book_id, v.rating FROM validation_df_pre_pre v JOIN train_df_pre t on v.user_id = t.user_id and v.book_id = t.book_id ') 
    validation_df_pre.createOrReplaceTempView('validation_df_pre')
    
    #Convert to Pandas
    train_df = train_df_pre.toPandas()
    validation_df = validation_df_pre.toPandas()
    
    #Prepare column 'user_id' for the COO matrix for train and validation  
    le_user_id = preprocessing.LabelEncoder()
    train_user_id = le_user_id.fit_transform(train_df['user_id'].values)
    validation_user_id = le_user_id.transform(validation_df['user_id'].values)
    
    #Prepare column 'book_id' for the COO matrix for train and validation
    le_book_id = preprocessing.LabelEncoder()
    train_book_id = le_book_id.fit_transform(train_df['book_id'].values)
    validation_book_id = le_book_id.transform(validation_df['book_id'].values)
        
    #Prepare column 'rating' for the COO matrix for train and validation
    le_rating = preprocessing.LabelEncoder()
    train_rating = le_rating.fit_transform(train_df['rating'].values)
    validation_rating = le_rating.transform(validation_df['rating'].values)
    
    #Ensure a standard COO matrix shape
    M = len(np.unique(train_user_id))
    N = len(np.unique(train_book_id))
    
    #Create the COO matrices for train and validation, as required by the LightFM process
    train = coo_matrix((train_rating,(train_user_id, train_book_id)), shape=(M, N))
    validation = coo_matrix((validation_rating,(validation_user_id, validation_book_id)), shape=(M, N))
    
    for lf in loss_functions:
        for nc in num_components:
            for e in epoch_range:
                
                #Instantiate and fit the LightFM model
                fit_start_time = time.time()
                model = LightFM(no_components=nc, loss=lf, random_state=42)
                model.fit(train, epochs=e) 
                fit_end_time = time.time()
                
                #Evaluate the model via Precision at 500
                precision_start_time = time.time()
                precision = precision_at_k(model, validation, k=500).mean()
                precision_end_time = time.time()
                
                #Evaluate the model via Recall at 500
                recall_start_time = time.time()
                recall = recall_at_k(model, validation, k=500).mean()
                recall_end_time = time.time()
                
                #Evaluate the model via AUC
                auc_start_time = time.time()
                auc = auc_score(model, validation).mean()
                auc_end_time = time.time()
                   
                #Print fitting configurations and results
                print('LightFM fit loss function:{}'.format(str(lf)))
                print('LightFM fit no_components:{}'.format(str(nc)))
                print('LightFM fit epochs:{}'.format(str(e)))
                print('LightFM fit time:{}'.format(str(round(fit_end_time - fit_start_time,0))))   
                #Print evaluation configurations and results
                print('Precision at 500:{}'.format(str(precision)))
                print('Precision at 500 time:{}'.format(str(round(precision_end_time - precision_start_time,0))))
                print('Recall at 500:{}'.format(str(recall)))
                print('Recall at 500 time:{}'.format(str(round(recall_end_time - recall_start_time,0))))
                print('AUC:{}'.format(str(auc)))
                print('AUC time:{}'.format(str(round(auc_end_time - auc_start_time,0))))
                print('###################################')
    
#Only enter this block if we're in main
if __name__ == "__main__":
    
    #Create the spark session object
    spark = SparkSession.builder.appName('LightFM').getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    
    #Set the training dataset's file name 
    training_data_file = 'train_df.parquet'
    
    #Set the validation dataset's file name 
    validation_data_file = 'validation_df.parquet'
    
    #Set the loss functions to consider
    loss_functions = ['warp','bpr', 'logistic']
    
    #Set the no_components to search over
    num_components = [5, 10, 15, 20, 25, 30]
    
    #Set the epochs to search over
    epoch_range = [5, 10, 15, 20, 25, 30]
    
    #Call our main routine
    main(spark, training_data_file, validation_data_file, loss_functions, num_components, epoch_range)
    
    #Stop Spark
    spark.stop()