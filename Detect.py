from __future__ import print_function
import tensorflow as tf
import os

# import pathlib
# emailed_link = 'https://urldefense.proofpoint.com/v2/url?u=https-3A__apple.box.com_shared_static_uqafgvmz8d1s2uexbg74fmqlyal6jzov.zip&d=DwMFaQ&c=clK7kQUTWtAVEOVIgvi0NU5BOUHhpN0H8p7CSfnc_gI&r=d4ucKxSoqaeRpi9SjIi_Xw&m=IHt8J6CHK1Tjq2MH1BtaRBmLfhvQSeK6Q2HMzE5fabI&s=JChTdl1eP1yofrrOOlOn7Q4BYl0gSNxMm6DUQxlP5l4&e='
# data_dir = tf.keras.utils.get_file(fname= "F:/Acad/lc/apple_DC/iphone/iphone",
#                                    origin=emailed_link,
#                                    extract=True)
# data_dir = pathlib.Path(data_dir)
# data_dir = pathlib.Path("F:/Acad/lc/apple_DC/splitted")

# manual labeling
DATASET_PATH = "F:/Acad/lc/apple_DC/splitted" # the dataset file or root folder path.

# Image Parameters
N_CLASSES = 2 # total number of classes (11 defected , 89 undefected)
IMG_HEIGHT = 64  # the image height to be resized to
IMG_WIDTH = 64 # the image width to be resized to
CHANNELS = 3 # The 3 color channels, can change to 1 if want grayscale


# Reading the dataset
def read_images(dataset_path, batch_size):
    imagepaths, labels = list(), list()

    # An ID will be affected to each sub-folders by alphabetical order
    label = 0
    # List the directory
    classes = sorted(os.walk(dataset_path).__next__()[1])
    # List each sub-directory (the classes)
    for c in classes:
        c_dir = os.path.join(dataset_path, c)
        walk = os.walk(c_dir).__next__()
        # Add each image to the set
        for sample in walk[2]:
            # Only keeps jpeg images
            if sample.endswith('.jpg'):
                imagepaths.append(os.path.join(c_dir, sample))
                labels.append(label)
        label += 1

    # Convert to Tensor
    imagepaths = tf.convert_to_tensor(imagepaths, dtype=tf.string)
    labels = tf.convert_to_tensor(labels, dtype=tf.int32)
    # Build a TF Queue, shuffle data
    image, label = tf.train.slice_input_producer([imagepaths, labels],
                                                 shuffle=True)
    # Read images from disk
    image = tf.read_file(image)
    image = tf.image.decode_jpeg(image, channels=CHANNELS)

    # Resize images to a common size
    image = tf.image.resize_images(image, [IMG_HEIGHT, IMG_WIDTH])

    # Normalize
    image = image * 1.0/127.5 - 1.0

    # Create batches
    X, Y = tf.train.batch([image, label], batch_size=batch_size,
                          capacity= batch_size * 8,
                          num_threads=4)

    #split to training, evaluation and test sets
    trainig_size =  int(batch_size*.8)
    eval_size = int(batch_size * .1)
    tr_X = tf.slice(X, [0,0,0,0], [trainig_size,-1,-1,-1])
    eval_X = tf.slice(X, [trainig_size, 0, 0, 0], [eval_size, -1, -1, -1])
    tes_X = tf.slice(X, [trainig_size + eval_size, 0, 0, 0], [-1, -1, -1, -1])

    tr_y = tf.slice(Y, [0], [trainig_size])
    eval_y = tf.slice(Y, [trainig_size], [eval_size])
    tes_y = tf.slice(Y, [trainig_size+eval_size], [-1])

    return tr_X, eval_X, tes_X, tr_y, eval_y, tes_y

# -----------------------------------------------
# THIS IS A CLASSIC CNN (see examples, section 3)
# -----------------------------------------------

# Parameters
learning_rate = 0.001
num_steps = 1000
batch_size = 10
display_step = 100

# Network Parameters
dropout = 0.75 # Dropout, probability to keep units

# Build the data input
tr_X, eval_X, tes_X, tr_y, eval_y, tes_y = read_images(DATASET_PATH, batch_size)



# Create model
def conv_net(x, n_classes, dropout, reuse, is_training):
    # Define a scope for reusing the variables
    with tf.variable_scope('ConvNet', reuse=reuse):

        # Convolution Layer with 32 filters and a kernel size of 5
        conv1 = tf.layers.conv2d(x, 32, 5, activation=tf.nn.relu)
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        conv1 = tf.layers.max_pooling2d(conv1, 2, 2)

        # Convolution Layer with 32 filters and a kernel size of 5
        conv2 = tf.layers.conv2d(conv1, 64, 3, activation=tf.nn.relu)
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        conv2 = tf.layers.max_pooling2d(conv2, 2, 2)

        # Flatten the data to a 1-D vector for the fully connected layer
        fc1 = tf.contrib.layers.flatten(conv2)

        # Fully connected layer (in contrib folder for now)
        fc1 = tf.layers.dense(fc1, 1024)
        # Apply Dropout (if is_training is False, dropout is not applied)
        fc1 = tf.layers.dropout(fc1, rate=dropout, training=is_training)

        # Output layer, class prediction
        out = tf.layers.dense(fc1, n_classes)
        # Because 'softmax_cross_entropy_with_logits' already apply softmax,
        # we only apply softmax to testing network
        out = tf.nn.softmax(out) if not is_training else out

    return out


# Because Dropout have different behavior at training and prediction time, we
# need to create 2 distinct computation graphs that share the same weights.

# Create a graph for training
logits_train = conv_net(tr_X, N_CLASSES, dropout, reuse=False, is_training=True)
# Create another graph for evaluating on training set that reuse the same weights
logits_train_2 = conv_net(tr_X, N_CLASSES, dropout, reuse=True, is_training=False)

# Define loss and optimizer (with train logits, for dropout to take effect)
loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
    logits=logits_train, labels=tr_y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model (with test logits, for dropout to be disabled)
correct_pred = tf.equal(tf.argmax(logits_train_2, 1), tf.cast(tr_y, tf.int64))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# evaluate on eval set
logits_eval = conv_net(eval_X, N_CLASSES, dropout, reuse=True, is_training=False)
loss_op_eval = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
    logits=logits_train, labels=tr_y))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Saver object
saver = tf.train.Saver()

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    # Start the data queue
    tf.train.start_queue_runners()

    # Training cycle
    for step in range(1, num_steps+1):

        if step % display_step == 0:
            # Run optimization and calculate batch loss and accuracy
            _, loss, acc = sess.run([train_op, loss_op, accuracy])
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))
        else:
            # Only run the optimization op (backprop)
            sess.run(train_op)

    print("Optimization Finished!")

    # Save your model
    saver.save(sess, 'my_tf_model')
