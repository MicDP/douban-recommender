package com.colobu.douban.recommender

import org.apache.spark.ml.linalg.Matrices
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.linalg.distributed._
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

/**
  * @author: eric.wang
  * @date: 2018/8/2 14:07
  * @description:
  */
object MLDataType {

  def main(args: Array[String]): Unit = {
    println("hello, ml")
    val conf = new SparkConf().setAppName("hello").setMaster("local[*]")
    val sc = new SparkContext(conf)

    // Vectors
    val v1 = Vectors.dense(1, 2, 3)
    val v2 = Vectors.sparse(3, Array(1,2), Array(-1, 4))
    val v3 = Vectors.sparse(3, Seq((1, -2.0), (2, 5.0)))
    println("v1: " + v1)
    println("v2: " + v2)
    println("v3: " + v3)

    // LabelPoint
    val lp1 = LabeledPoint(1.0, v1)
    val lp2 = LabeledPoint(0, v2)
    println("lp1: " + lp1)
    println("lp2: " + lp2)

    // Matrix
    val dm1 = Matrices.dense(2,2, Array(1, 2, 3, 4))
    val dm2 = Matrices.sparse(2, 2, Array(0, 2, 4), Array(0, 1, 0, 1), Array(1.0, 2, 3.0, 4))
//    val dm2 = Matrices.sparse(3, 2, Array(0, 1, 3), Array(0, 2, 1), Array(9, 6, 8))

    println("dm1: " + dm1)
    println("dm2: " + dm2)

    // Distributed matrix: row / indexed
    val rows :RDD[Vector] = sc.makeRDD(Seq(v1, v2, v3))
    val mat1 = new RowMatrix(rows)
    val qrResult = mat1.tallSkinnyQR(true)
    println("mat1: " + mat1.numRows() + ", " + mat1.numCols() + ", QR: " + qrResult)
    mat1.rows.foreach(println)
    // an RDD of indexed rows
    val rows2: RDD[IndexedRow] = sc.makeRDD(Seq(IndexedRow(1,v1), IndexedRow(2,v2), IndexedRow(3,v3)))
    // Create an IndexedRowMatrix from an RDD[IndexedRow].
    val mat2: IndexedRowMatrix = new IndexedRowMatrix(rows2)
    // Get its size.
    val m = mat2.numRows()
    val n = mat2.numCols()

    // Drop its row indices.
    val rowMat: RowMatrix = mat2.toRowMatrix()
//    println("mat2: " + rowMat.tallSkinnyQR(true))
    rowMat.rows.foreach(println)

    // coordinate matrix
    val entries : RDD[MatrixEntry] = sc.makeRDD(
      Seq(
        MatrixEntry(0, 0, 2),
        MatrixEntry(1, 1, 2),
        MatrixEntry(2, 2, 2)))
    val corMat : CoordinateMatrix = new CoordinateMatrix(entries)
    println("corMat: ")
    corMat.toRowMatrix().rows.foreach(println)

    // block matrix
    val matA : BlockMatrix = corMat.toBlockMatrix().cache()
    matA.validate()

    val result = matA.transpose.multiply(matA)

    println("A^T * A")
    result.toCoordinateMatrix().toRowMatrix().rows.foreach(println)

//    val training = sc.(Seq(
//      (1.0, Vectors.dense(0.0, 1.1, 0.1)),
//      (0.0, Vectors.dense(2.0, 1.0, -1.0)),
//      (0.0, Vectors.dense(2.0, 1.3, 1.0)),
//      (1.0, Vectors.dense(0.0, 1.2, -0.5))
//    )).toDF("label", "features")

  }

}
