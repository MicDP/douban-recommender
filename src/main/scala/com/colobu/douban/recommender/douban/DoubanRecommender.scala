package com.colobu.douban.recommender.douban

import com.colobu.douban.recommender.douban.DoubanTraining.{buildMovieId2Name, buildUserName2Id, reverse}
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.recommendation.{MatrixFactorizationModel, Rating}

import scala.collection.Map

/**
  * @author: eric.wang
  * @date: 2018/8/16 14:46
  * @description:
  */
object DoubanRecommender {

  def main(args: Array[String]): Unit = {
    val base = if (args.length > 0) args(0) else "/opt/douban/"
    val modelSavePath = "ALS-Model"
    val conf = new SparkConf().setAppName("Douban-Recommender").setMaster("local[*]")
    val sc = new SparkContext(conf)
    val model = MatrixFactorizationModel.load(sc, base + modelSavePath)
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
    val bMovieName2Id = sc.broadcast(movieName2Id)
    val bMovieId2Name = sc.broadcast(movieId2Name)
    // 获取某个用户对某个物品的评分
    val userName = "adamwzw"
    val movieName = "小王子"
    val rating = predict(userName, movieName, model, sc, userName2Id, movieName2Id)
    // 向用户推荐 Top-N 产品
    recommendProducts(userName, 10, model, sc, bUserName2Id, bUserId2Name, bMovieId2Name)
    // 向产品推荐 Top-n 用户

  }

  /**
    * 预测用户对电影的评分
    * @param userName 用户名
    * @param movieName 电影名
    * @param model 模型
    * @param sc Spark 程序上下文
    * @param userName2Id 用户名称与 ID 映射关系
    * @param movieName2Id 电影名称与 ID 映射关系
    * @return 评分
    */
  def predict(userName: String, movieName: String, model: MatrixFactorizationModel,
              sc: SparkContext, userName2Id: Map[String, Int], movieName2Id: Map[String, Int]): Double = {
    val bUserName2Id = sc.broadcast(userName2Id)
    val bMovieName2Id = sc.broadcast(movieName2Id)
    val rating = predict(bUserName2Id.value(userName), bMovieName2Id.value(movieName), model)
    println(s"用户：${userName} 对电影：${movieName} 的评分是：${rating}")
    rating
  }
  /**
    * 预测用户对电影的评分
    * @param userId 用户 Id
    * @param movieId 电影 Id
    * @param model 模型
    * @return 评分
    */
  def predict(userId: Int, movieId: Int, model: MatrixFactorizationModel): Double = {
    model.predict(userId, movieId)
  }

  /**
    * 向指定用户推荐 Top-N 产品
    * @param userName 推荐用户
    * @param num Top-N
    * @param model 模型
    * @param sc spark 上下文
    * @param bUserName2Id 用户名与 ID 映射关系
    * @param bUserId2Name 用户 ID 与名称映射关系
    * @param bMovieId2Name 电影 ID 与名称映射关系
    */
  def recommendProducts(userName: String,
                        num: Int,
                        model: MatrixFactorizationModel,
                        sc: SparkContext,
                        bUserName2Id: Broadcast[Map[String, Int]],
                        bUserId2Name: Broadcast[Map[Int, String]],
                        bMovieId2Name: Broadcast[Map[Int, String]]) {
    val userId = bUserName2Id.value(userName)
    val rp: Array[Rating] = model.recommendProducts(userId, num)
    rp.map {
      case Rating(userId, movieId, rating) =>
        (bUserId2Name.value(userId), bMovieId2Name.value(movieId), rating)
    }.foreach(println)
  }






}
