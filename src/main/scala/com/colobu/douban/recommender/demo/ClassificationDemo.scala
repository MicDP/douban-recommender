package com.colobu.douban.recommender.demo

import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.{SparkConf, SparkContext}

/**
  * @author: eric.wang
  * @date: 2018/8/3 15:05
  * @description:
  */
object ClassificationDemo {

  def main(args: Array[String]): Unit = {

    val input = "data/mllib/sample_libsvm_data.txt"
    val output = "E:/temp/ml/tmp/LogisticRegressionWithLBFGSModel"


    val conf = new SparkConf().setAppName("classification").setMaster("local[*]")
    val sc = new SparkContext(conf)

    val data = MLUtils.loadLibSVMFile(sc, input)
//    data.foreach(println)

    val splitData = data.randomSplit(Array(0.6, 0.4), 11L)
    val trainData = splitData(0)
    val testData = splitData(1)

    // Run training algorithm to build the model
    val model = new LogisticRegressionWithLBFGS()
      .setNumClasses(10)
      .run(trainData)
    val predictionAndLabels = testData.map {
      case LabeledPoint(label, features) =>
        val prediction = model.predict(features)
      (prediction, label)
    }

    // Get evaluation metrics.
    val metrics = new MulticlassMetrics(predictionAndLabels)
    println(s"Accuracy = ${metrics.accuracy}")

    // Save and load model
    model.save(sc, output)
//    val sampleModel = LogisticRegressionModel.load(output)



  }

}
