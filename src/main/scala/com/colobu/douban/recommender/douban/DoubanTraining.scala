package com.colobu.douban.recommender.douban

import org.apache.spark.mllib.recommendation.{MatrixFactorizationModel, Rating}
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

import scala.collection.Map

/**
  * @author: eric.wang
  * @date: 2018/8/15 15:53
  * @description:
  */
object DoubanTraining {

  def main(args: Array[String]): Unit = {
    val base = ""

    val conf = new SparkConf().setMaster("local[*]").setAppName("Douban-Training")
    val sc = new SparkContext(conf)

    // 获取原料数据集
    val userMovieData = sc.textFile(base + "user_movies.csv")
    val hotUserMovieData = sc.textFile(base + "hot_movies.csv")
    // 数据清洗

    // 转成模型结构数据
    val data = buildRating(userMovieData)
    // 获取 UserID 的 ID 化值
    val userID2Int = buildUserID2Int(userMovieData)
    sc.broadcast(userID2Int)
    // 获取训练数据
    val trainingData = buildTrainingData(data, userID2Int)
    // 训练数据并存储模型
    val model = trainingData(trainingData, 50, 10, 0.01)
    // 测试推荐效果

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
      case Array(name, ratingStr, id) =>
        Some((id.toInt, name))
    }.collectAsMap()
  }

  /**
    * 获取 userID 的 ID 化值
    * @param data 原始数据集
    * @return userID 的 ID 化值
    */
  def buildUserID2Int(data: RDD[String]): Map[String, Int] = {
    data.map(_.split(",").map(_.trim)).map(_(0)).distinct().zipWithUniqueId().map {
      case (id, userID) =>
        (userID.toString, id.toInt)
    }.collectAsMap()
  }

  def buildTrainingData(data: RDD[UserMovieRating], trainingUserId2Int: Map[String, Int]): RDD[Rating] = {
    data.map {
      case UserMovieRating(userID, movieId, rating) =>
        if (trainingUserId2Int.contains(userID)) {
          Rating(trainingUserId2Int.get(userID).get, movieId, rating)
        } else {
          None
        }
    }
  }

  def training(data: RDD[UserMovieRating],
               rank: Int,
               iterations: Int,
               lambda: Double): MatrixFactorizationModel = {

  }

}
