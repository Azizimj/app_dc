import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
np.random.seed(110)

# manual labeling
DATASET_PATH = "./splitted-all" # the dataset file or root folder path.

# Image Parameters
N_CLASSES = 2 # total number of classes (11 defected , 89 undefected)
IMG_HEIGHT = 64  # the image height to be resized to
IMG_WIDTH = 64 # the image width to be resized to
CHANNELS = 1 # The 3 color channels, can change to 1 if want grayscale
training_prec = 0.8 # trian set percentage
eval_prec = .1  # validation set percentage, rest is test


def augmentation(image, gammas, rotations):
    '''
    input: image tensor
    return: a list of augmented images
    
    By a bried look at the data and also the assumption that we are in a semi-controlled environment,
    we just did flips, shifts and add light. 
    '''
    result = []
    angels = np.random.random(rotations)*50-25  # random -25, 25 degree rotate
    for angel in angels:
        result.append(cv2.warpAffine(image, cv2.getRotationMatrix2D((IMG_WIDTH / 2, IMG_HEIGHT / 2), angel, 1.0),dsize=(IMG_WIDTH,IMG_HEIGHT)))
    result.append(cv2.flip(image, flipCode=-1)) # vertical and horizontal flip
    result.append(cv2.flip(image, flipCode=0)) # vertical flip
    result.append(cv2.flip(image, flipCode=1)) # horizontal flip
    for gamma in gammas:
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
                          for i in np.arange(0, 256)]).astype("uint8")
        result.append(cv2.LUT(image, table))

    return result


def read_images(dataset_path):
    '''
    input: str folder path 
    output: train, validation and test data sets
    '''
    images, labels = list(), list()
    images_dir = list()

    classes = sorted(os.walk(dataset_path).__next__()[1])  # List the directory

    # grayscale or RGB
    if CHANNELS==1:
        read_img = lambda x: cv2.imread(x, cv2.IMREAD_GRAYSCALE)
    elif CHANNELS==3:
        read_img = lambda x: cv2.imread(x, cv2.IMREAD_UNCHANGED)

    # List each sub-directory (the classes)
    for c in classes:
        if c =="defected":
            label = 1
        elif c=="undefected":
            label= 0
            
        c_dir = os.path.join(dataset_path, c)
        walk = os.walk(c_dir).__next__()
        # Add each image to the set
        for sample in walk[2]:
            img_path = os.path.join(c_dir, sample)
            image = read_img(img_path)
            image = cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH), interpolation=cv2.INTER_AREA)
            images.append(image * 1.0 / 127.5 - 1.0)  # normalize
            images_dir.append(img_path)
            labels.append(label)
            
            if label == 1:  # augmentation just on defected
                res = augmentation(image, [.1, .4, .6, 1], 5)
                images += res
                images_dir += [img_path]*len(res)
                labels += [label]*len(res)
#             if label == 0:
#                 res = augmentation(image, [.1, 1], 0)
#                 images += res
#                 images_dir += [img_path]*len(res)
#                 labels += [label]*len(res)
        

    # shuffle and split train, validation and test
    num_data = len(labels)
#     np.random.seed(110) # set the seed
    idx = np.random.permutation(len(labels))
    images, images_dir, labels = np.array(images)[idx,:], np.array(images_dir)[idx] , np.array(labels)[idx], 
    
    num_training = int(num_data * .8)
    num_eval = int(num_data * .1)

    tr_X = images[:num_training]
    eval_X = images[num_training : num_eval+num_training]
    tes_X = images[num_eval+num_training:]
    tes_X_dir = images_dir[num_eval+num_training:]

    tr_y = labels[:num_training]
    eval_y = labels[num_training : num_training+num_eval]
    tes_y = labels[num_training+num_eval:]

    return tr_X, eval_X, tes_X, tr_y, eval_y, tes_y, num_training, num_eval, tes_X_dir


X_train, X_val, X_test, Y_train, Y_val, Y_test, num_training, num_eval, tes_X_dir = read_images(DATASET_PATH)
defection_prec = lambda x: (len(x), sum(x)/len(x))

print('Train set has %d samples and defection proportion = %.3f' % defection_prec(Y_train))
print('Validation set has %d samples and defection proportion = %.3f' % defection_prec(Y_val))
print('Test set has %d samples and defection proportion = %.3f' % defection_prec(Y_test))



# an imaginary channel in the gray case
if CHANNELS == 1:
    X_train = np.expand_dims(X_train, axis=-1)
    X_val = np.expand_dims(X_val, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)
    
    
    
# # definec some wrappers with almost tunned parameters
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


num_filter_1 = 100
num_filter_2 = 300
num_filter_3 = 400
class BaseModel(object):
    def __init__(self):
        self.batch_size = 19
        self.num_epoch = 10
        self.log_step = 5
        self._build_model()

    def _model(self):
        with tf.variable_scope('conv1'):
            self.conv1 = conv2d(self.X, 7, 1, num_filter_1)
            self.relu1 = tf.nn.relu(self.conv1)
            self.pool1 = max_pool(self.relu1, 5, 2)
            self.bn1 = tf.layers.batch_normalization(self.pool1, training=self.is_training)
        with tf.variable_scope('conv2'):
            self.conv2 = conv2d(self.bn1, 5, 1, num_filter_2)
            self.relu2 = tf.nn.relu(self.conv2)
            self.pool2 = max_pool(self.relu2, 3, 2)
            self.bn2 = tf.layers.batch_normalization(self.pool2, training=self.is_training)
        with tf.variable_scope('conv3'):
            self.conv3 = conv2d(self.bn2, 5, 1, num_filter_3)
            self.relu3 = tf.nn.relu(self.conv3)
            self.pool3 = max_pool(self.relu3, 3, 2)
            self.bn3 = tf.layers.batch_normalization(self.pool3, training=self.is_training)
        self.flat = tf.layers.flatten(self.bn3)
        
        with tf.variable_scope('fc1'):
            self.fc1 = tf.layers.dense(self.flat, 384)
            self.fc1 = tf.nn.dropout(x=self.fc1, keep_prob=self.keep_prob)
            self.relu_fc1 = tf.nn.relu(self.fc1)
        with tf.variable_scope('fc2'):
            self.fc2 = tf.layers.dense(self.relu_fc1, 200)
            self.fc2 = tf.nn.dropout(x=self.fc2, keep_prob=self.keep_prob)
            self.relu_fc2 = tf.nn.relu(self.fc2)
        with tf.variable_scope('fc3'):
            self.fc3 = tf.layers.dense(self.relu_fc2, N_CLASSES)
        return self.fc3

    def _input_ops(self):
        # Placeholders
        self.X = tf.placeholder(tf.float32, [None, IMG_HEIGHT, IMG_WIDTH, CHANNELS])
        self.Y = tf.placeholder(tf.int64, [None])
        self.is_training = tf.placeholder(tf.bool, None)
        self.lr = tf.placeholder(tf.float32, None)
        self.keep_prob = tf.placeholder(tf.float32 , None)
        

    def _build_optimizer(self):
        # Adam optimizer 'self.train_op' that minimizes 'self.loss_op'
        gs = tf.Variable(0, trainable=False)
        l_r = tf.train.exponential_decay(learning_rate=self.lr, global_step=gs,decay_steps=10, decay_rate=0.9, staircase=True)
        optimer = tf.train.AdamOptimizer(l_r)
        self.train_op = optimer.minimize(self.loss_op, global_step=gs)

    def _loss(self, labels, logits):
        # Softmax cross entropy loss
        ls = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        self.loss_op = tf.reduce_mean(ls)

    def _build_model(self):
        self._input_ops() # Define input variables
        labels = tf.one_hot(self.Y, N_CLASSES) # Convert Y to one-hot vector
        self.logits = self._model() # Build a model and get logits
        self._loss(labels, self.logits) # Compute loss
        self._build_optimizer()  # Build optimizer

        # Compute accuracy and f1 score
        self.predict = tf.argmax(self.logits, 1)
        correct = tf.equal(self.predict, self.Y)
        self.accuracy_op = tf.reduce_mean(tf.cast(correct, tf.float32))
        TP = tf.count_nonzero(self.predict * self.Y)
        TN = tf.count_nonzero((self.predict - 1) * (self.Y - 1))
        FP = tf.count_nonzero(self.predict * (self.Y - 1))
        FN = tf.count_nonzero((self.predict - 1) * self.Y)
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        self.f1_score = 2 * precision * recall / (precision + recall)

    def train(self, sess, X_train, Y_train, X_val, Y_val, lr):
        sess.run(tf.global_variables_initializer())
        step = 0
        losses = []
        accuracies = []
        val_accuracies = []
        print('-' * 5 + '  Start training  ' + '-' * 5)
        for epoch in range(self.num_epoch):
            print('train for epoch %d' % epoch)
            for i in range(num_training // self.batch_size):
                X_ = X_train[i * self.batch_size:(i + 1) * self.batch_size][:]
                Y_ = Y_train[i * self.batch_size:(i + 1) * self.batch_size]

                feed_dict = {self.X: X_, self.Y: Y_, self.lr:lr, self.is_training : True, self.keep_prob: .7}
                fetches = [self.train_op, self.loss_op, self.accuracy_op]
                # import IPython  # used for online debugging (we cannot do this using something like Pycharm, useful note for me) 
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
            val_accuracy, _, _ = self.evaluate(sess, X_val, Y_val)
            val_accuracies.append(val_accuracy)
            print('-  epoch %d: validation accuracy = %.3f' % (epoch, val_accuracy))

#         ###################################
#         # Plot training curve             #
#         ###################################
#         plt.title('Training loss')
#         loss_hist_ = losses[1::len(losses)//20]  # sparse the curve a bit
#         acc_hist_ = accuracies[1::len(accuracies)//20]
#         plt.plot(loss_hist_, '-o')
#         plt.plot(accuracies*100, '-s')
    
#         plt.xlabel('epoch')
#         plt.gcf().set_size_inches(15, 12)
#         plt.show()
        return losses, accuracies, val_accuracies

    def evaluate(self, sess, X_eval, Y_eval):
        eval_accuracy = 0.0
        eval_f1 = 0.0
        eval_iter = 0
        predicts = []
        for i in range(X_eval.shape[0] // self.batch_size):
            X_ = X_eval[i * self.batch_size:(i + 1) * self.batch_size][:]
            Y_ = Y_eval[i * self.batch_size:(i + 1) * self.batch_size]
            feed_dict = {self.X: X_, self.Y: Y_, self.is_training : False, self.keep_prob: 1}
            accuracy, f1, preds= sess.run([self.accuracy_op, self.f1_score, self.predict], feed_dict=feed_dict)
            eval_accuracy += accuracy
            eval_f1 += f1
            eval_iter += 1
            predicts += list(preds)
            
        X_ = X_eval[(i+1) * self.batch_size:][:]
        Y_ = Y_eval[(i+1) * self.batch_size:]
        feed_dict = {self.X: X_, self.Y: Y_,  self.is_training : False, self.keep_prob: 1}
        accuracy, f1, preds= sess.run([self.accuracy_op, self.f1_score, self.predict], feed_dict=feed_dict)
        eval_accuracy += accuracy
        eval_f1 += f1
        eval_iter += 1
        predicts += list(preds)
        
        return eval_accuracy / eval_iter, eval_f1/ eval_iter, predicts
        
    def _predict(self, sess, X):
        feed_dict = {self.X: X, self.is_training : False, self.keep_prob: 1}
        pred = sess.run([self.predict], feed_dict=feed_dict)
        return pred
        
            
tf.reset_default_graph()

with tf.Session() as sess:
    with tf.device('/cpu:0'):
        model = BaseModel()
        lr = 1e-4
        losses, accuracies, val_accuracies = model.train(sess, X_train, Y_train, X_val, Y_val, lr)
        accuracy, f1, preds = model.evaluate(sess, X_test, Y_test)
        
#         pred = model._predict(sess, image)
#         print("pred", pred)
        
        print('***** test accuracy: %.3f, f1: %.3f' % (accuracy, f1))
        print("preds", preds)
        print("Y_test", Y_test)
        
        saver = tf.train.Saver()
        model_path = saver.save(sess, "./app_dc.ckpt")
        print("Model saved in %s" % model_path)