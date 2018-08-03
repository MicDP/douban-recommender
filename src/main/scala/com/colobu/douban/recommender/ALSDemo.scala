package com.colobu.douban.recommender

import org.apache.spark.mllib.recommendation.{ALS, Rating}
import org.apache.spark.{SparkConf, SparkContext}

/**
  * @author: eric.wang
  * @date: 2018/8/3 17:58
  * @description:
  */
object ALSDemo {

  def main(args: Array[String]): Unit = {
    val input = ""
    val sc = new SparkContext(
      new SparkConf()
      .setAppName("ALS").setMaster("local[*]")
    )

    val data = sc.textFile("data/mllib/als/test.data")

    val ratings = data.map(_.split(",") match {
      case Array(user, item, rate) =>
        Rating(user.toInt, item.toInt, rate.toDouble)
    })
    ratings.foreach(println)

    // Build the recommendation model using ALS
    val rank = 10
    val numIterations = 10
    val model = ALS.train(ratings, rank, numIterations, 0.1)

    // Evaluate the model on rating data
    val userProducts = ratings.map {
      case Rating(user, product, rate) =>
        (user, product)
    }
    val predictions = model.predict(userProducts).map {
      case Rating(user, product, rate) =>
        ((user, product), rate)
    }

    val rateAndPreds = ratings.map {
      case Rating(user, product, rate) =>
        ((user, product), rate)
    }.join(predictions)

    rateAndPreds.foreach(println)

    val MSE = rateAndPreds.map {
      case ((user, product), (r1, r2)) =>
        val r = r1 - r2
        r * r
    }.mean()
    println(s"MSE: ${MSE}")

  }

}
