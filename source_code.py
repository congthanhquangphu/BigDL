
import time
import numpy as np



#################################
import matplotlib


import pandas
import datetime as dt

from bigdl.nn.layer import *
from bigdl.nn.criterion import *
from bigdl.optim.optimizer import *
from bigdl.util.common import *
from bigdl.dataset.transformer import *
from bigdl.dataset import mnist
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt




sc = SparkContext(conf=create_spark_conf())
init_engine() # prepare the bigdl environment




def get_mnist(sc, mnist_path):
    # target is start from 0,
    (train_images, train_labels) = mnist.read_data_sets(mnist_path, "train")
    (test_images, test_labels) = mnist.read_data_sets(mnist_path, "test")
    training_mean = np.mean(train_images)
    training_std = np.std(train_images)
    rdd_train_images = sc.parallelize(train_images[:10000])
    rdd_train_labels = sc.parallelize(train_labels[:10000])
    rdd_test_images = sc.parallelize(test_images[:1000])
    rdd_test_labels = sc.parallelize(test_labels[:1000])

    rdd_train_sample = rdd_train_images.zip(rdd_train_labels).map(lambda x:
                    Sample.from_ndarray((x[0] - training_mean)/training_std, x[1] + 1))
    rdd_test_sample = rdd_test_images.zip(rdd_test_labels).map(lambda x:
                    Sample.from_ndarray((x[0] - training_mean)/training_std, x[1] + 1))
    return (rdd_train_sample, rdd_test_sample)


##################################
mnist_path = "datasets/mnist"
(train_data, test_data) = get_mnist(sc, mnist_path)

train_data = train_data.map(lambda s: Sample.from_ndarray(np.resize(s.features[0].to_ndarray(), (28, 28)), s.label.to_ndarray()))
test_data = test_data.map(lambda s: Sample.from_ndarray(np.resize(s.features[0].to_ndarray(), (28, 28)), s.label.to_ndarray()))

print(train_data.count())
print(test_data.count())



####################################
# Parameters
batch_size = 128

# Network Parameters
n_input = 28 # MNIST data input (img shape: 28*28)
n_hidden = 128 # hidden layer num of features
n_classes = 10 # MNIST total classes (0-9 digits)

###########################################
def build_model(input_size, hidden_size, output_size):
    model = Sequential()
    recurrent = Recurrent()
    recurrent.add(LSTM(input_size, hidden_size))
    model.add(InferReshape([-1, input_size], True))
    model.add(recurrent)
    model.add(Select(2, -1))
    model.add(Linear(hidden_size, output_size))
    return model
rnn_model = build_model(n_input, n_hidden, n_classes)

###############################################
# Create an Optimizer

#criterion = TimeDistributedCriterion(CrossEntropyCriterion())
criterion = CrossEntropyCriterion()
optimizer = Optimizer(
    model=rnn_model,
    training_rdd=train_data,
    criterion=criterion,
    optim_method=Adam(),
    end_trigger=MaxEpoch(5),
    batch_size=batch_size)

# Set the validation logic
optimizer.set_validation(
    batch_size=batch_size,
    val_rdd=test_data,
    trigger=EveryEpoch(),
    val_method=[Top1Accuracy()]
)


app_name='rnn-'+dt.datetime.now().strftime("%Y%m%d-%H%M%S")
train_summary = TrainSummary(log_dir='/tmp/bigdl_summaries',
                                     app_name=app_name)
train_summary.set_summary_trigger("Parameters", SeveralIteration(50))
val_summary = ValidationSummary(log_dir='/tmp/bigdl_summaries',
                                        app_name=app_name)
optimizer.set_train_summary(train_summary)
optimizer.set_val_summary(val_summary)
print("saving logs to ",app_name)

################################################
def map_predict_label(l):
    return np.array(l).argmax()
def map_groundtruth_label(l):
    return int(l[0] - 1)


