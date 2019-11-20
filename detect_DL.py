import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

# import pathlib
# emailed_link = 'https://urldefense.proofpoint.com/v2/url?u=https-3A__apple.box.com_shared_static_uqafgvmz8d1s2uexbg74fmqlyal6jzov.zip&d=DwMFaQ&c=clK7kQUTWtAVEOVIgvi0NU5BOUHhpN0H8p7CSfnc_gI&r=d4ucKxSoqaeRpi9SjIi_Xw&m=IHt8J6CHK1Tjq2MH1BtaRBmLfhvQSeK6Q2HMzE5fabI&s=JChTdl1eP1yofrrOOlOn7Q4BYl0gSNxMm6DUQxlP5l4&e='
# data_dir = tf.keras.utils.get_file(fname= "F:/Acad/lc/apple_DC/iphone/iphone",
#                                    origin=emailed_link,
#                                    extract=True)
# data_dir = pathlib.Path(data_dir)
# data_dir = pathlib.Path("F:/Acad/lc/apple_DC/splitted")


# manual labeling
DATASET_PATH = "F:/Acad/lc/apple_DC/splitted-all" # the dataset file or root folder path.

# Image Parameters
N_CLASSES = 2  # total number of classes (11 defected , 89 undefected)
IMG_HEIGHT = 64  # the image height to be resized to
IMG_WIDTH = 64  # the image width to be resized to
CHANNELS = 1  # The 3 color channels, can change to 1 if want grayscale

training_prec = 0.8 # trian set percentage
eval_prec = .1  # validation set percentage

def augmentation(image):
    result = []
    angels = np.random.random(5)*50-25  # random -25, 25 degree rotate
    for angel in angels:
        result.append(cv2.warpAffine(image, cv2.getRotationMatrix2D((IMG_WIDTH / 2, IMG_HEIGHT / 2), angel, 1.0),dsize=(IMG_WIDTH,IMG_HEIGHT)))
    result.append(cv2.flip(image, flipCode=-1)) # vertical and horizontal flip
    result.append(cv2.flip(image, flipCode=0)) # vertical flip
    result.append(cv2.flip(image, flipCode=1)) # horizontal flip
    invGamma = 1
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    result.append(cv2.LUT(image, table))

    return result


# Reading the dataset
def read_images(dataset_path):
    # imagepaths, labels = list(), list()
    images, labels = list(), list()

    # An ID will be affected to each sub-folders by alphabetical order
    label = 0
    # List the directory
    classes = sorted(os.walk(dataset_path).__next__()[1])

    # grayscale or RGB
    if CHANNELS==1:
        read_img = lambda x: cv2.imread(x, cv2.IMREAD_GRAYSCALE)
    elif CHANNELS==3:
        read_img = lambda x: cv2.imread(x, cv2.IMREAD_UNCHANGED)

    # List each sub-directory (the classes)
    for c in classes:
        c_dir = os.path.join(dataset_path, c)
        walk = os.walk(c_dir).__next__()
        # Add each image to the set
        for sample in walk[2]:
            image = read_img(os.path.join(c_dir, sample))
            image = cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH), interpolation=cv2.INTER_AREA)
            if label == 0:
                res = augmentation(image)
                images += res
                labels += [label]*len(res)
            images.append(image * 1.0 / 127.5 - 1.0)
            labels.append(label)
        label += 1

    # shuffle
    num_data = len(labels)
    np.random.seed(110)
    idx = np.random.permutation(len(labels))
    images, labels = np.array(images)[idx,:], np.array(labels)[idx]

    num_training = int(num_data * .8)
    num_eval = int(num_data * .1)

    tr_X = images[:num_training]
    eval_X = images[num_training : num_eval+num_training]
    tes_X = images[num_eval+num_training:]

    tr_y = labels[:num_training]
    eval_y = labels[num_training : num_training+num_eval]
    tes_y = labels[num_training+num_eval:]

    return tr_X, eval_X, tes_X, tr_y, eval_y, tes_y, num_training, num_eval

X_train, X_val, X_test, Y_train, Y_val, Y_test, num_training, num_eval = read_images(DATASET_PATH)
defection_prec = lambda x: (len(x), 1-sum(x)/len(x))

print('Train set has %d sample and defection proportion = %.3f' % defection_prec(Y_train))
print('Validation set has %d sample and defection proportion = %.3f' % defection_prec(Y_val))
print('Test set has %d sample and defection proportion = %.3f' % defection_prec(Y_test))

# an imaginary channel in the gray case
if CHANNELS == 1:
    X_train = np.expand_dims(X_train, axis=-1)
    X_val = np.expand_dims(X_val, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)


# # definec some wrappers
def conv2d(input, kernel_size, stride, num_filter):
    stride_shape = [1, stride, stride, 1]
    filter_shape = [kernel_size, kernel_size, input.get_shape()[3], num_filter]

    W = tf.get_variable('w', filter_shape, tf.float32, tf.random_normal_initializer(0.0, 0.02))
    b = tf.get_variable('b', [1, 1, 1, num_filter], initializer=tf.constant_initializer(0.0))
    return tf.nn.conv2d(input, W, stride_shape, padding='SAME') + b

def max_pool(input, kernel_size, stride):
    ksize = [1, kernel_size, kernel_size, 1]
    strides = [1, stride, stride, 1]
    return tf.nn.max_pool(input, ksize=ksize, strides=strides, padding='SAME')

class BaseModel(object):
    def __init__(self):
        self.num_epoch = 20
        self.batch_size = 5
        self.log_step = 5
        self._build_model()

    def _model(self):
        with tf.variable_scope('conv1'):
            self.conv1 = conv2d(self.X, 7, 1, 32)
            self.relu1 = tf.nn.relu(self.conv1)
            self.pool1 = max_pool(self.relu1, 3, 2)

        with tf.variable_scope('conv2'):
            self.conv2 = conv2d(self.pool1, 5, 1, 64)
            self.relu2 = tf.nn.relu(self.conv2)
            self.pool2 = max_pool(self.relu2, 3, 2)
        self.flat = tf.layers.flatten(self.pool2)

        with tf.variable_scope('fc3'):
            self.fc3 = tf.layers.dense(self.flat, 384)
            self.relu3 = tf.nn.relu(self.fc3)

        with tf.variable_scope('fc4'):
            self.fc4 = tf.layers.dense(self.relu3, N_CLASSES)

        # Return the last layer
        return self.fc4

    def _input_ops(self):
        # Placeholders
        self.X = tf.placeholder(tf.float32, [None, IMG_HEIGHT, IMG_WIDTH, CHANNELS])
        self.Y = tf.placeholder(tf.int64, [None])
        self.is_training = tf.placeholder(tf.bool, None)

    def _build_optimizer(self):
        # Adam optimizer 'self.train_op' that minimizes 'self.loss_op'
        gs = tf.Variable(0, trainable=False)
        l_r = tf.train.exponential_decay(learning_rate=1e-3, global_step=gs,
                                         decay_steps=50, decay_rate=0.9, staircase=True)
        optimer = tf.train.AdamOptimizer(l_r)
        self.train_op = optimer.minimize(self.loss_op, global_step=gs)
        print()

    def _loss(self, labels, logits):
        # Softmax cross entropy loss
        ls = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        self.loss_op = tf.reduce_mean(ls)

    def _build_model(self):
        # Define input variables
        self._input_ops()
        # Convert Y to one-hot vector
        labels = tf.one_hot(self.Y, N_CLASSES)

        # Build a model and get logits
        logits = self._model()

        # Compute loss
        self._loss(labels, logits)

        # Build optimizer
        self._build_optimizer()

        # Compute accuracy
        predict = tf.argmax(logits, 1)
        correct = tf.equal(predict, self.Y)
        self.accuracy_op = tf.reduce_mean(tf.cast(correct, tf.float32))
        TP = tf.count_nonzero(predict * self.Y)
        TN = tf.count_nonzero((predict - 1) * (self.Y - 1))
        FP = tf.count_nonzero(predict * (self.Y - 1))
        FN = tf.count_nonzero((predict - 1) * self.Y)
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        self.f1_score = 2 * precision * recall / (precision + recall)

    def train(self, sess, X_train, Y_train, X_val, Y_val):
        sess.run(tf.global_variables_initializer())

        step = 0
        losses = []
        accuracies = []
        print('-' * 5 + '  Start training  ' + '-' * 5)
        for epoch in range(self.num_epoch):
            print('train for epoch %d' % epoch)
            for i in range(num_training // self.batch_size):
                X_ = X_train[i * self.batch_size:(i + 1) * self.batch_size][:]
                Y_ = Y_train[i * self.batch_size:(i + 1) * self.batch_size]

                feed_dict = {self.X: X_, self.Y: Y_}
                fetches = [self.train_op, self.loss_op, self.accuracy_op]
                # import IPython
                # IPython.embed()

                _, loss, accuracy = sess.run(fetches, feed_dict=feed_dict)
                losses.append(loss)
                accuracies.append(accuracy)

                if step % self.log_step == 0:
                    print('iteration (%d): loss = %.3f, accuracy = %.3f' %
                          (step, loss, accuracy))
                step += 1

            # Print validation results
            print('validation for epoch %d' % epoch)
            val_accuracy, f1 = self.evaluate(sess, X_val, Y_val)
            print('-  epoch %d: validation accuracy = %.3f, f1 = %.3f' % (epoch, val_accuracy, f1))

        ###################################
        # Plot training curve             #
        ###################################
        # Graph 1. X: epoch, Y: training loss

        plt.title('Training loss')
        loss_hist_ = losses[1::len(losses)//10]  # sparse the curve a bit
        plt.plot(loss_hist_, '-o')
        plt.xlabel('epoch')
        plt.gcf().set_size_inches(15, 12)
        plt.show()

    def evaluate(self, sess, X_eval, Y_eval):
        eval_accuracy = 0.0
        eval_f1 = 0.0
        eval_iter = 0
        for i in range(X_eval.shape[0] // self.batch_size):
            X_ = X_eval[i * self.batch_size:(i + 1) * self.batch_size][:]
            Y_ = Y_eval[i * self.batch_size:(i + 1) * self.batch_size]
            feed_dict = {self.X: X_, self.Y: Y_}
            accuracy, f1 = sess.run([self.accuracy_op, self.f1_score], feed_dict=feed_dict)
            eval_accuracy += accuracy
            eval_f1 += f1
            eval_iter += 1
        return eval_accuracy / eval_iter, eval_f1/ eval_iter

tf.reset_default_graph()

# Train our sample model
with tf.Session() as sess:
    with tf.device('/cpu:0'):
        model = BaseModel()
        model.train(sess, X_train, Y_train, X_val, Y_val)
        accuracy, f1 = model.evaluate(sess, X_test, Y_test)
        print('***** test accuracy: %.3f, f1: %.3f' % (accuracy, f1))
