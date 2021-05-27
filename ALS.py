#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pyspark.sql import SparkSession
import time
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.mllib.evaluation import RegressionMetrics, RankingMetrics

def main(spark, training_data_file, validation_data_file, ranks, regParams, metrics):
    
    #Read in our training and validation data files
    train_df = spark.read.parquet('./'+training_data_file)
    train_df.createOrReplaceTempView('train_df')
    validation_df = spark.read.parquet('./'+validation_data_file)
    validation_df.createOrReplaceTempView('validation_df')
    
    #Instantiate the ALS model
    als_model = ALS(userCol = 'user_id', itemCol = 'book_id', ratingCol = 'rating', coldStartStrategy = 'drop')
    
    for rank in ranks:
        for regParam in regParams:     
            training_start_time = time.time()
            #Set the hyperparameters of the model
            als_model.setParams(rank = rank, regParam = regParam)
            #Fit the model
            als_fit = als_model.fit(train_df)
            training_end_time = time.time()
            
            for metric in metrics:
                #Evaluate the model via RMSE from ml.evaluation
                if metric == "rmse":
                    evaluation_start_time = time.time()
                    predictions = als_fit.transform(validation_df)
                    evaluator = RegressionEvaluator(metricName = metric, labelCol = 'rating', predictionCol = 'prediction')
                    metric_result = evaluator.evaluate(predictions)
                    evaluation_end_time = time.time()
                    
                    print('ALS with rank {} and  regularization {} gave a(n) {} of {}, and took {} seconds to train, {} seconds to evaluate.'.\
                          format(str(rank), str(regParam), str(metric), str(round(metric_result,2)), \
                          str(round(training_end_time - training_start_time,0)), \
                          str(round(evaluation_end_time - evaluation_start_time,0))))
                #Evaluate the model via RMSE from mllib.evaluation
                elif metric == "rmse_mllib":
                    evaluation_start_time = time.time()
                    predictions = als_fit.transform(validation_df)  
                    predictions2 = predictions.selectExpr("user_id", "book_id", "cast(rating as double) rating", "cast(prediction as double) as prediction")
                    
                    results = predictions2.select(['user_id','book_id', 'rating', 'prediction']).rdd
                    ratingsTuple = results.map(lambda r: ((r.user_id, r.book_id), r.rating))
                    predictionstuple = results.map(lambda r: ((r.user_id, r.book_id), r.prediction))
                    labels_and_predictions = predictionstuple.join(ratingsTuple).map(lambda tup: tup[1])
                    metric_holder = RegressionMetrics(labels_and_predictions)
                    metric_result = metric_holder.rootMeanSquaredError
                    evaluation_end_time = time.time()
                    
                    print('ALS with rank {} and  regularization {} gave a(n) {} of {}, and took {} seconds to train, {} seconds to evaluate.'.\
                          format(str(rank), str(regParam), str(metric), str(round(metric_result,2)), \
                          str(round(training_end_time - training_start_time,0)), \
                          str(round(evaluation_end_time - evaluation_start_time,0))))
                #If evaluation is Precision at k, MAP, or NDCG at k, we start the process the same way here
                else:
                    evaluation_start_time = time.time()
                    predictions = als_fit.transform(validation_df)  
                    predictions2 = predictions.selectExpr("user_id", "book_id", "cast(rating as double) rating", "cast(prediction as double) as prediction")
                    predictions2.createOrReplaceTempView('predictions2')
                    predicted_ranks = spark.sql('SELECT user_id, collect_list(book_id) as items FROM (SELECT user_id, book_id, prediction, RANK() OVER(PARTITION BY user_id ORDER BY prediction DESC) as rnk FROM predictions2) i WHERE rnk <= 500 GROUP BY user_id')
                    actual_ranks = spark.sql('SELECT user_id, collect_list(book_id) as items FROM (SELECT user_id, book_id, rating, RANK() OVER(PARTITION BY user_id ORDER BY rating DESC) as rnk FROM predictions2) i WHERE rnk <= 500 GROUP BY user_id')
                    predictionAndLabels = predicted_ranks.join(actual_ranks, 'user_id').rdd.map(lambda x: (x[1], x[2]))
                    metric_holder = RankingMetrics(predictionAndLabels)
                    if metric == "Precision":
                        metric_result = metric_holder.precisionAt(500)
                    elif metric == "MAP":
                        metric_result = metric_holder.meanAveragePrecision
                    else:
                        metric_result = metric_holder.ndcgAt(500)
                    evaluation_end_time = time.time()
                    
                    print('ALS with rank {} and  regularization {} gave a(n) {} of {}, and took {} seconds to train, {} seconds to evaluate.'.\
                          format(str(rank), str(regParam), str(metric), str(round(metric_result,2)), \
                          str(round(training_end_time - training_start_time,0)), \
                          str(round(evaluation_end_time - evaluation_start_time,0))))
                    
#Only enter this block if we're in main
if __name__ == "__main__":
    
    #Create the spark session object
    spark = SparkSession.builder.appName('ALS').getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    
    #Set the training dataset's file name 
    training_data_file = 'train_df.parquet'
    
    #Set the validation dataset's file name 
    validation_data_file = 'validation_df.parquet'
    
    #Set the ranks to search over
    ranks = [1, 10, 20, 30, 40, 50]
    
    #Set the regParams to search over
    regParams = [0.001, 0.01, 0.1, 1, 10]
    
    #Set the metrics to consider
    metrics = ["rmse","rmse_mllib", "Precision", "MAP", "NDCG"]
    
    #Call our main routine
    main(spark, training_data_file, validation_data_file, ranks, regParams, metrics)
    
    #Stop Spark
    spark.stop()