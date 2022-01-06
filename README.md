# Evaluate the performance of BigDL (Distributed Deep Learning on Apache Spark) in big data analysis problems.

## Introduction

BigDL is a distributed deep learning library for Apache Spark; with BigDL, users can write their
deep learning applications as standard Spark programs, which can directly run on top of existing
Spark or Hadoop clusters.

![BigDL](https://github.com/BigDL/images/3.png)

## Installation

- Please download BigDL Packages or pip install BigDL (conda)

## How to run Program on Spark

Usage: spark-submit-with-bigdl.sh + [options] + file.py

Options:
- master MASTER URL: spark, yarn, k8s, local.
- local[k]: Run Spark locally with k worker threads as logical cores on your machine.
- File.py: File for executing program.

## System configuration

Program run on system includes:
- System/Host Processor: Intel(R) Xeon(R) CPU E5-2680 v3 @ 2.50GHz
- CPU(s): 48
- Core(s) per socket: 12
- Socket(s): 2
- Memory: 183 G (free)

## Data Description and Run Model

It is a dataset of 60,000 small square 28×28 pixel grayscale images of handwritten single digits
between 0 and 9. The MNIST data is split into three parts: 60,000 data points of training data,
10,000 points of test data.

![BigDL](https://github.com/BigDL/images/1.png)

With this BigDL Problem, We use LSTM model for MNIST digit classification problem.

## BigDL Performance Evaluation 

### Execution running time

![BigDL](https://github.com/BigDL/images/8.png)

![BigDL](https://github.com/BigDL/images/5.png)

### Computation Evaluation (SPEED UP)

![BigDL](https://github.com/BigDL/images/9.png)

![BigDL](https://github.com/BigDL/images/6.png)























