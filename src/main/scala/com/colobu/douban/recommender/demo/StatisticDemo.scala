package com.colobu.douban.recommender.demo

import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.stat.Statistics
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

/**
  * @author: eric.wang
  * @date: 2018/8/3 11:00
  * @description:
  */
object StatisticDemo {

  def main(args: Array[String]): Unit = {

    val conf = new SparkConf()
      .setMaster("local[*]")
      .setAppName("statistic")
    val sc = new SparkContext(conf)

    val v1 : RDD[Vector] = sc.makeRDD(
      Seq(
        Vectors.dense(1, 2, 3),
        Vectors.dense(2, 3, 4),
        Vectors.dense(3, 4, 5))
    )
    val v2 : RDD[Vector] = sc.parallelize(
      Seq(
        Vectors.dense(1, 2, 3),
        Vectors.dense(2, 3, 4),
        Vectors.dense(3, 4, 5)
      )
    )
    v1.foreach(println)
    v2.foreach(println)

    // Compute column summary statistics.
    val s1 = Statistics.colStats(v1)
    val s2 = Statistics.colStats(v2)
    println(s"${s1.count}, ${s1.max}, ${s1.min}, ${s1.mean}, " +
      s"${s1.normL1}, ${s1.normL2}, ${s1.numNonzeros}, ${s1.variance}")


    // correlation
    val sx : RDD[Double] = sc.parallelize(Array(1, 2, 3, 4, 5))
    val sy : RDD[Double] = sc.parallelize(Array(11, 22, 33, 44, 55))
    val corxy = Statistics.corr(sx, sy, "pearson")
    println(s"cor: ${corxy}")

    val data = sc.parallelize(
      Seq(
        Vectors.dense(1.0, 10.0, 100.0),
        Vectors.dense(2.0, 20.0, 200.0),
        Vectors.dense(5.0, 33.0, 366.0)
      )
    )

    println(s"${Statistics.corr(data, "pearson")}")


    // Stratified sampling

    val data2 = sc.parallelize(
      Seq(
        (1, 'a'),
        (1, 'b'),
        (2, 'c'),
        (2, 'd'),
        (2, 'e'),
        (3, 'f')
      )
    )

    val fractions = Map(1 -> 0.1, 2 -> 0.6, 3 -> 0.3)
    // Get an approximate sample from each stratum
    val approxSample = data2.sampleByKey(withReplacement = false, fractions = fractions)
    // Get an exact sample from each stratum
    val exactSample = data2.sampleByKeyExact(withReplacement = false, fractions = fractions)

    approxSample.foreach(println)
    exactSample.foreach(println)

  }

}
