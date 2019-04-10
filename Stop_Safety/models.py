import tensorflow as tf
import numpy as np
from data import shuffle
import math
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

class Model(object):
    def __init__(self):

        tf.reset_default_graph()

        self.X = tf.placeholder(tf.float32, [None, 88, 200, 3])
        self.Y = tf.placeholder(tf.float32, [None, 1])
        self.keep_prob = tf.placeholder_with_default(1.0, shape=())
        self.training = tf.placeholder(tf.bool, name='is_training')

        self.initModel()
        self.buildModel()

        self.saver = tf.train.Saver()

    def initModel(self):
        raise NotImplementedError

    def buildModel(self):
        self.logits = self.net(self.X)
        self.prediction = tf.nn.sigmoid(self.logits)

        self.correct_pred = tf.equal(tf.round(self.prediction), self.Y)
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=self.Y))
        self.regularizer = tf.add_n([tf.nn.l2_loss(w) for w in list(self.weights.values())])

    def net(self):
        raise NotImplementedError

    def conv2d(self, x, W, b, strides, batch_norm=True):
        x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
        x = tf.nn.bias_add(x, b)
        if batch_norm:
            x = tf.layers.batch_normalization(x, training=self.training)
        x = tf.nn.relu(x)
        return x

    def maxpool2d(self, x, k):
        return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

    def fc(self, x, W, b, batch_norm=True, activation=True):
        x = tf.matmul(x, W) + b
        if batch_norm:
            x = tf.layers.batch_normalization(x, training=self.training)
        if activation:
            x = tf.nn.relu(x)
        return x

    def resblock(self, x, W1, b1, W2, b2, strides, batch_norm=True):
        
        f = tf.nn.conv2d(x, W1, strides=[1, strides, strides, 1], padding='SAME')
        f = tf.nn.bias_add(f, b1)
        if batch_norm:
            f = tf.layers.batch_normalization(f, training=self.training)
        f = tf.nn.relu(f)
        
        f = tf.nn.conv2d(f, W2, strides=[1, strides, strides, 1], padding='SAME')
        f = tf.nn.bias_add(f, b2)
        if batch_norm:
            f = tf.layers.batch_normalization(f, training=self.training)

        x = f + x
        f = tf.nn.relu(f)

        return x

    def train(self, trainInput, testInput, trainTarget, testTarget, \
        reg_lambda=0.0, learning_rate=1e-4, dropout=0.0, batch_size=32, epochs=50, \
        restore_model=False, save_model=True, save_freq=5):
        
        self.loss_op = self.loss + reg_lambda*self.regularizer

        #self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        self.train_op = self.optimizer.minimize(self.loss_op)
        self.train_op = tf.group([self.train_op, self.update_ops])

        print("Training...")

        total_num_input = trainInput.shape[0]
        steps = math.ceil(total_num_input / batch_size)

        train_loss_history, test_loss_history, \
        train_accuracy_history, test_accuracy_history = [], [], [], []

        with tf.Session() as sess:
            tf.gfile.MakeDirs(self.save_folder)
            if restore_model:
                lastckpt = tf.train.latest_checkpoint(self.save_folder)
                print("Restoring from {}".format(lastckpt))
                self.saver = tf.train.import_meta_graph(lastckpt+'.meta')
                self.saver.restore(sess, lastckpt)
            else:
                sess.run(tf.global_variables_initializer())
            
            for e in range(1, epochs+1):
                X, Y = shuffle(trainInput, trainTarget)

                print("Training phase:")
                for i in tqdm(range(0, steps)):
                    X_batch = X[i * batch_size:(i + 1) * batch_size, :]
                    Y_batch = Y[i * batch_size:(i + 1) * batch_size, :]
                    n_batch = X_batch.shape[0]
                    _, = sess.run([self.train_op], feed_dict={
                            self.X: X_batch,
                            self.Y: Y_batch,
                            self.keep_prob: 1-dropout,
                            self.training: True})

                # Need to run in non-training mode for batch norm and dropout
                print("Test phase:")
                train_loss, train_accuracy, train_prediction = self._test(sess, trainInput, trainTarget, batch_size=batch_size)
                test_loss, test_accuracy, test_prediction = self._test(sess, testInput, testTarget, batch_size=batch_size)
                
                train_auc = roc_auc_score(trainTarget, train_prediction)
                test_auc = roc_auc_score(testTarget, test_prediction)

                train_loss_history.append(train_loss)
                train_accuracy_history.append(train_accuracy)
                test_loss_history.append(test_loss)
                test_accuracy_history.append(test_accuracy)

                print('Epoch %3d ==> Train Loss: %.4f, Train AUC: %.4f, Test Loss: %.4f, Test AUC: %.4f' % \
                    (e, train_loss_history[-1], train_auc, test_loss_history[-1], test_auc))

                if save_model and (e % save_freq == 0):
                    self.saver.save(sess, self.save_folder+'model.ckpt', global_step=e)

            if save_model:
                self.saver.save(sess, self.save_folder+'model.ckpt', global_step=epochs)

        loss_history = {
            "train": train_loss_history,
            "test": test_loss_history
        }
        accuracy_history = {
            "train": train_accuracy_history,
            "test": test_accuracy_history
        }
        return loss_history, accuracy_history

    def test(self, Input, Target, batch_size=32):

        with tf.Session() as sess:         
            lastckpt = tf.train.latest_checkpoint(self.save_folder)
            print("Restoring from {}".format(lastckpt))
            self.saver = tf.train.import_meta_graph(lastckpt+'.meta')
            self.saver.restore(sess, lastckpt)

            test_loss, test_accuracy, test_prediction = self._test(sess, Input, Target, batch_size=batch_size)

        return test_loss, test_accuracy, test_prediction

    def _test(self, sess, Input, Target, batch_size=32):
        total_num_input = Input.shape[0]
        steps = math.ceil(total_num_input / batch_size)

        test_loss = 0
        test_accuracy = 0

        predictions = []
        for i in tqdm(range(0, steps)):
            X = Input[i * batch_size:(i + 1) * batch_size, :]
            Y = Target[i * batch_size:(i + 1) * batch_size, :]
            n = X.shape[0]
            _loss, _accuracy, _prediction = sess.run(
                [self.loss, self.accuracy, self.prediction], feed_dict={
                    self.X: X,
                    self.Y: Y,
                    self.keep_prob: 1.0,
                    self.training: False})
            test_loss += _loss*n
            test_accuracy += _accuracy*n
            predictions.append(_prediction)

        test_loss = test_loss/total_num_input
        test_accuracy = test_accuracy/total_num_input
        test_prediction = np.concatenate(predictions, axis=0)
        
        return test_loss, test_accuracy, test_prediction

#########
# Multilayered Perception #
#########

class MLP_Model(Model):
    def __init__(self):
        super().__init__()
            
        self.save_folder = "./models/mlp/"

    def initModel(self):
        self.weights = {
            'w_hidden1' : tf.get_variable(name="WH1", shape=[88*200*3, 4096], initializer=tf.contrib.layers.xavier_initializer()),
            'w_hidden2' : tf.get_variable(name="WH2", shape=[4096, 1024], initializer=tf.contrib.layers.xavier_initializer()),
            'w_hidden3' : tf.get_variable(name="WH3", shape=[1024, 256], initializer=tf.contrib.layers.xavier_initializer()),
            'out' : tf.get_variable(name="WOUT", shape=[256, 1], initializer=tf.contrib.layers.xavier_initializer())
        }

        self.biases = {
            'b_hidden1': tf.get_variable(name="BH1", shape=[4096], initializer=tf.contrib.layers.xavier_initializer()),
            'b_hidden2': tf.get_variable(name="BH2", shape=[1024], initializer=tf.contrib.layers.xavier_initializer()),
            'b_hidden3': tf.get_variable(name="BH3", shape=[256], initializer=tf.contrib.layers.xavier_initializer()),
            'out': tf.get_variable(name="BOUT", shape=[1], initializer=tf.contrib.layers.xavier_initializer())
        }

    def net(self, x):
        weights = self.weights
        biases = self.biases

        input_layer = tf.reshape(x, [-1, 88*200*3])

        fc1 = self.fc(input_layer, weights['w_hidden1'], biases['b_hidden1'])
        fc2 = self.fc(fc1, weights['w_hidden2'], biases['b_hidden2'])
        fc3 = self.fc(fc2, weights['w_hidden3'], biases['b_hidden3'])
        fc3 = tf.nn.dropout(fc3, self.keep_prob)
        out = self.fc(fc3, weights['out'], biases['out'], batch_norm=False, activation=False)

        return out



#########
# Convolutional Neural Network #
#########

class CNN_Model(Model):
    def __init__(self):
        super().__init__()
            
        self.save_folder = "./models/cnn/"

        # set graph-level random seed
        # tf.set_random_seed(421)

    def initModel(self):
        self.weights = {
            'w_conv1' : tf.get_variable(name="WC1", shape=[5, 5, 3, 32], initializer=tf.contrib.layers.xavier_initializer()),
            'w_conv2' : tf.get_variable(name="WC2", shape=[5, 5, 32, 64], initializer=tf.contrib.layers.xavier_initializer()),
            'w_fc1' : tf.get_variable(name="WD1", shape=[22*50*64, 1024], initializer=tf.contrib.layers.xavier_initializer()),
            'w_fc2' : tf.get_variable(name="WD2", shape=[1024, 256], initializer=tf.contrib.layers.xavier_initializer()),
            'out' : tf.get_variable(name="WOUT", shape=[256, 1], initializer=tf.contrib.layers.xavier_initializer())
        }

        self.biases = {
            'b_conv1': tf.get_variable(name="BC1", shape=[32], initializer=tf.contrib.layers.xavier_initializer()),
            'b_conv2': tf.get_variable(name="BC2", shape=[64], initializer=tf.contrib.layers.xavier_initializer()),
            'b_fc1': tf.get_variable(name="BD1", shape=[1024], initializer=tf.contrib.layers.xavier_initializer()),
            'b_fc2': tf.get_variable(name="BD2", shape=[256], initializer=tf.contrib.layers.xavier_initializer()),
            'out': tf.get_variable(name="BOUT", shape=[1], initializer=tf.contrib.layers.xavier_initializer())
        }

    def net(self, x):
        weights = self.weights
        biases = self.biases

        input_layer = tf.reshape(x, [-1, 88, 200, 3])

        conv1 = self.conv2d(input_layer, weights['w_conv1'], biases['b_conv1'], strides=1)
        conv1_pool = self.maxpool2d(conv1, k=2)

        conv2 = self.conv2d(conv1_pool, weights['w_conv2'], biases['b_conv2'], strides=1)
        conv2_pool = self.maxpool2d(conv2, k=2)

        flattened = tf.reshape(conv2_pool, [-1, 22*50*64])
        fc1 = self.fc(flattened, weights['w_fc1'], biases['b_fc1'])
        fc2 = self.fc(fc1, weights['w_fc2'], biases['b_fc2'])
        fc2 = tf.nn.dropout(fc2, self.keep_prob)

        out = self.fc(fc2, weights['out'], biases['out'], batch_norm=False, activation=False)
        return out


#########
# Residual Neural Network #
#########

class ResNet_Model(Model):
    def __init__(self):
        super().__init__()
            
        self.save_folder = "./models/resnet/"

        # set graph-level random seed
        # tf.set_random_seed(421)

    def initModel(self):
        self.weights = {
            'w_res1_1' : tf.get_variable(name="WR1_1", shape=[5, 5, 3, 32], initializer=tf.contrib.layers.xavier_initializer()),
            'w_res1_2' : tf.get_variable(name="WR1_2", shape=[5, 5, 32, 3], initializer=tf.contrib.layers.xavier_initializer()),
            'w_conv1' : tf.get_variable(name="WC1", shape=[5, 5, 3, 32], initializer=tf.contrib.layers.xavier_initializer()),
            'w_res2_1' : tf.get_variable(name="WR2_1", shape=[5, 5, 32, 64], initializer=tf.contrib.layers.xavier_initializer()),
            'w_res2_2' : tf.get_variable(name="WR2_2", shape=[5, 5, 64, 32], initializer=tf.contrib.layers.xavier_initializer()),
            'w_conv2' : tf.get_variable(name="WC2", shape=[5, 5, 32, 64], initializer=tf.contrib.layers.xavier_initializer()),
            'w_fc1' : tf.get_variable(name="WD1", shape=[22*50*64, 1024], initializer=tf.contrib.layers.xavier_initializer()),
            'w_fc2' : tf.get_variable(name="WD2", shape=[1024, 256], initializer=tf.contrib.layers.xavier_initializer()),
            'out' : tf.get_variable(name="WOUT", shape=[256, 1], initializer=tf.contrib.layers.xavier_initializer())
        }

        self.biases = {
            'b_res1_1': tf.get_variable(name="BR1_1", shape=[32], initializer=tf.contrib.layers.xavier_initializer()),
            'b_res1_2': tf.get_variable(name="BR1_2", shape=[3], initializer=tf.contrib.layers.xavier_initializer()),
            'b_conv1': tf.get_variable(name="BC1", shape=[32], initializer=tf.contrib.layers.xavier_initializer()),            
            'b_res2_1': tf.get_variable(name="BR2_1", shape=[64], initializer=tf.contrib.layers.xavier_initializer()),
            'b_res2_2': tf.get_variable(name="BR2_2", shape=[32], initializer=tf.contrib.layers.xavier_initializer()),
            'b_conv2': tf.get_variable(name="BC2", shape=[64], initializer=tf.contrib.layers.xavier_initializer()),
            'b_fc1': tf.get_variable(name="BD1", shape=[1024], initializer=tf.contrib.layers.xavier_initializer()),
            'b_fc2': tf.get_variable(name="BD2", shape=[256], initializer=tf.contrib.layers.xavier_initializer()),
            'out': tf.get_variable(name="BOUT", shape=[1], initializer=tf.contrib.layers.xavier_initializer())
        }

    def net(self, x):
        weights = self.weights
        biases = self.biases

        input_layer = tf.reshape(x, [-1, 88, 200, 3])

        res1 = self.resblock(input_layer, weights['w_res1_1'], biases['b_res1_1'], \
                            weights['w_res1_2'], biases['b_res1_2'], strides=1)
        conv1 = self.conv2d(res1, weights['w_conv1'], biases['b_conv1'], strides=1)
        conv1_pool = self.maxpool2d(conv1, k=2)

        res2 = self.resblock(conv1_pool, weights['w_res2_1'], biases['b_res2_1'], \
                            weights['w_res2_2'], biases['b_res2_2'], strides=1)
        conv2 = self.conv2d(res2, weights['w_conv2'], biases['b_conv2'], strides=1)
        conv2_pool = self.maxpool2d(conv2, k=2)

        flattened = tf.reshape(conv2_pool, [-1, 22*50*64])
        fc1 = self.fc(flattened, weights['w_fc1'], biases['b_fc1'])
        fc2 = self.fc(fc1, weights['w_fc2'], biases['b_fc2'])
        fc2 = tf.nn.dropout(fc2, self.keep_prob)

        out = self.fc(fc2, weights['out'], biases['out'], batch_norm=False, activation=False)
        return out
