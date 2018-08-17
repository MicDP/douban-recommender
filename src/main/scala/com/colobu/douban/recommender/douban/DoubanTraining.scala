package com.colobu.douban.recommender.douban

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.mllib.recommendation.{ALS, MatrixFactorizationModel, Rating}
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
import org.jblas.DoubleMatrix

import scala.collection.Map

/**
  * @author: eric.wang
  * @date: 2018/8/15 15:53
  * @description:
  */
object DoubanTraining {

  def main(args: Array[String]): Unit = {
    val base = if (args.length > 0) args(0) else "/opt/douban/"
    val modelSavePath = "ALS-Model"

    val conf = new SparkConf().setMaster("local[*]").setAppName("Douban-Training")
    val sc = new SparkContext(conf)

    // 获取原料数据集
    val userMovieData = sc.textFile(base + "user_movies.csv")
    val hotUserMovieData = sc.textFile(base + "hot_movies.csv")
    // 数据清洗、预处理
    // 获取 User 名称与 ID 映射，并广播
    val userName2Id = buildUserName2Id(userMovieData)
    val userId2Name = reverse(userName2Id)
    val bUserName2Id = sc.broadcast(userName2Id)
    val bUserId2Name = sc.broadcast(userId2Name)
    // 获取 Movie 名称与 ID 映射，并广播
    val movieId2Name = buildMovieId2Name(hotUserMovieData)
    val movieName2Id = reverse(movieId2Name)
    val bMovieId2Name = sc.broadcast(movieId2Name)
    val bMovieName2Id = sc.broadcast(movieName2Id)
    // 转成模型结构数据
    val data = buildRating(userMovieData)
    // 获取训练数据
    val trainingData = buildTrainingData(data, bUserName2Id)
    // 训练数据并存储模型
    val model = training(trainingData, 50, 10, 0.01)
    // 模型保存
    model.save(sc, base + modelSavePath)
    // 测试推荐效果
    // 计算 MSE、RMSE
    evaluate(sc, trainingData, model)
    bUserName2Id.unpersist()
    bUserId2Name.unpersist()
    bMovieName2Id.unpersist()
    bMovieId2Name.unpersist()
    unpersist(model)
  }

  /**
    * 生成 ALS 指定模型
    * @param data 原始数据集
    * @return 返回指定模型数据
    */
  def buildRating(data: RDD[String]): RDD[UserMovieRating] = {
    data.map(_.split(",").map(_.trim)).map {
      case Array(userID, movieID, ratingStr) =>
        var rating = ratingStr.toInt
        rating = if (-1==rating) 3 else rating
        UserMovieRating(userID, movieID.toInt, rating.toDouble)
    }
  }

  /**
    * 获取电影名称与ID之间的映射关系
    * @param data 原始数据集
    * @return 返回电影与ID之间的映射关系
    */
  def buildMovieId2Name(data: RDD[String]): Map[Int, String] = {
    data.map(_.split(",").map(_.trim)).flatMap {
      case Array(id, ratingStr, name) =>
        Some((id.toInt, name))
    }.collectAsMap()
  }

  /**
    * 获取 userID 的 ID 化值
    * @param data 原始数据集
    * @return userID 的 ID 化值
    */
  def buildUserName2Id(data: RDD[String]): Map[String, Int] = {
    data.map(_.split(",")).map(_(0).trim).distinct().zipWithUniqueId().map {
      case (userName, id) =>
        (userName, id.toInt)
    }.collectAsMap()
  }

  /**
    * 获取模型所需模型数据，并持久化
    * @param data 原始数据集
    * @param userIDMap 用户 ID 序列化映射
    * @return 获取模型所需模型数据
    */
  def buildTrainingData(data: RDD[UserMovieRating], userIDMap: Broadcast[Map[String, Int]]): RDD[Rating] = {
    data.map {
      case UserMovieRating(userID, movieId, rating) =>
          Rating(userIDMap.value(userID), movieId, rating)
    }.cache()
  }

  /**
    * 训练模型，并释放模型数据集
    * @param rating 模型数据集
    * @param rank K 维特征
    * @param iterations 迭代次数
    * @param lambda 正规化因子
    * @return 返回模型
    */
  def training(rating: RDD[Rating],
               rank: Int,
               iterations: Int,
               lambda: Double): MatrixFactorizationModel = {
    // 协同过滤算法训练模型
    val model = ALS.train(rating, rank, iterations, lambda)
    rating.unpersist()
    model
  }

  /**
    * k-v 转换， 即 v-k
    * @param data 原始数据集，类型：Map
    * @tparam A
    * @tparam B
    * @return v-k 数据集，类型：Map
    */
  def reverse[A, B](data: Map[A, B]):Map[B, A]= {
    data.map {
      case (a, b) => (b, a)
    }
  }

  /**
    * 计算 MSE、RMSE
    * @param ratings 原始数据集
    * @param model 模型
    */
  def evaluate(sc: SparkContext,
               ratings: RDD[Rating],
               model: MatrixFactorizationModel): Unit = {
    val userMovie = ratings.map {
      case Rating(userId, movieId, rate) =>
        (userId, movieId)
    }
    val predictions = model.predict(userMovie).map {
      case Rating(userId, movieId, rate) =>
        ((userId, movieId), rate)
    }
    val actualAndPredictions = ratings.map {
      case Rating(userId, movieId, rate) =>
        ((userId, movieId), rate)
    }.join(predictions)

    val predictedAndTrue = actualAndPredictions.map {
      case ((actual, prediction), (r1, r2)) =>
        (r1, r2)
    }

    // 计算 MSE, RMSE
//    val regressionMetrics= new RegressionMetrics(predictedAndTrue)
//    println(s"MSE: ${regressionMetrics.meanSquaredError}, " +
//      s"RMSE: ${regressionMetrics.rootMeanSquaredError}")
    calMSE(predictedAndTrue)
    // 计算 MAP
    calMAP(sc, ratings, model)
  }

  def calMSE(predictedAndTrue: RDD[(Double, Double)]): Unit = {
    // 计算 MSE, RMSE
    val MSE = predictedAndTrue.map {
      case (r1, r2) =>
        val r = r1 - r2
        r * r
    }.mean()
    println(s"MSE: ${MSE}, ${Math.sqrt(MSE)}")
  }
  def calMAP(sc: SparkContext, ratings: RDD[Rating], model: MatrixFactorizationModel): Unit = {
    // 得到电影线性表出向量
    val itemFactors = model.productFeatures.map { case (id, factor) => factor }.collect()
    val itemMatrix = new DoubleMatrix(itemFactors)
    println(itemMatrix.rows, itemMatrix.columns)
    val imBroadcast = sc.broadcast(itemMatrix)
    val allRecs = model.userFeatures.map { case (userId, array) =>
      val userVector = new DoubleMatrix(array)
      val scores = imBroadcast.value.mmul(userVector)
      // 根据评分降序
      val sortedWithId = scores.data.zipWithIndex.sortBy(-_._1)
      val recommendedIds = sortedWithId.map(_._2 + 1).toSeq
      (userId, recommendedIds)
    }

    val userMovies = ratings.map {
      case Rating(user, product, rating) =>
        (user, product)
    }.groupBy(_._1)

//    allRecs.take(2).foreach(println)
//    userMovies.take(2).foreach(println)
//    allRecs.join(userMovies).foreach(println)
    import org.apache.spark.mllib.evaluation.RankingMetrics
    val predictedAndTrue = allRecs.join(userMovies)
    val predictedAndTrueForRanking = predictedAndTrue.map {
      case (userId, (predicted, actualWithIds)) =>
        val actual = actualWithIds.map(_._2)
        (predicted.toArray, actual.toArray)
    }
    val rankingMetrics = new RankingMetrics(predictedAndTrueForRanking)
    // 计算 MAP
    println(s"Mean Average Precision = ${rankingMetrics.meanAveragePrecision}")
  }

  /**
    * 释放模型缓存
    * @param model 模型
    */
  def unpersist(model: MatrixFactorizationModel): Unit = {
    model.userFeatures.unpersist()
    model.productFeatures.unpersist()
  }

}
