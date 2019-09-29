from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np
import pandas as pd
import Data_Processor

class LSTM_net():
    def __init__(self,
                 sess,
                 stock_count,
                 lstm_shape,
                 num_steps,
                 input_size,
                 train_ratio,
                 logs_dir='',
                 plots_dir=''):
        self.sess = sess
        self.stock_count = stock_count
        self.lstm_shape = lstm_shape
        self.num_steps = num_steps
        self.input_size = input_size
        self.logs_dir = logs_dir
        self.plots_dir = plots_dir
        self.train_ratio=train_ratio
        # 在类的构造函数里就画好图
        self.graph()

    def graph(self):
        '''构建数据流图'''
        '''定义需要用到的占位符'''
        with tf.name_scope('inputs'):
            self.inputs = tf.placeholder(tf.float32, [None,self.num_steps, 2], name='inputs') # 2 is for the change and vol
            self.targets = tf.placeholder(tf.float32, [None, 1], name='targets')
            with tf.name_scope('parameter'):
                self.learning_rate = tf.placeholder(tf.float32, None, name='LearningRate')
                self.keep_prob = tf.placeholder(tf.float32, None, name='KeepProb')   # 表示在 dropout 中每个神经元被保留的几率

        '''根据lstm_size、keep_prob、num_layers构建含Dropout包装器的LSTM神经网络'''
        def _one_lstm_layer(lstm_size):
            lstm_layer = tf.nn.rnn_cell.DropoutWrapper(
                tf.nn.rnn_cell.LSTMCell(self.lstm_size, state_is_tuple=True),
                output_keep_prob=self.keep_prob
            )
            return lstm_layer

        with tf.name_scope('hidden_layer'):
            if len(self.lstm_shape) > 1 :
                lstm_layers = tf.nn.rnn_cell.MultiRNNCell([_one_lstm_layer(self.lstm_shape[i]) for i in range(len(self.lstm_shape))], state_is_tuple=True)
            else :
                lstm_layers = one_lstm_layer(self.lstm_shape[0]) #网络共num_layers层，每层的神经元结构相同

            '''获得LSTM网络的输出和状态'''
            lstm_output,_ = tf.nn.dynamic_rnn(lstm_layers, self.inputs, dtype=tf.float32)

            '''根据LSTM网络的输出计算出与target维度匹配的输出'''
            # Before transpose, lstm_output.get_shape() = (batch_size, num_steps, lstm_size)
            # After transpose, lstm_output.get_shape() = (num_steps, batch_size, lstm_size)
            lstm_output = tf.transpose(lstm_output, [1, 0, 2], name='hidden_layer_output')

        '''构建神经网络的输出层'''
        with tf.name_scope('linear_layer'):
            last = tf.gather(lstm_output,int(lstm_output.get_shape()[0])-1, name='last_lstm_output')
            weight = tf.Variable(tf.truncated_normal([self.lstm_size, 1]), name='weight')
            bias = tf.Variable(tf.constant(0.1,shape=[1]), name='bias')
            self.prediction = tf.matmul(last, weight) + bias
            tf.summary.histogram('prediction', self.prediction)

        '''模型的代价函数和优化器'''
        with tf.name_scope('loss'):
            self.loss=tf.reduce_mean(tf.square(self.prediction-self.targets))
        tf.summary.scalar('loss', self.loss)#均方差函数

        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

        '''tensorboard记录的参数mergerd, 以及文件接口'''
        self.merged = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter(self.logs_dir + '/train', self.sess.graph)
        self.test_writer = tf.summary.FileWriter(self.logs_dir + '/test')



    def train(self,max_epoch,init_learning_rate,decay_rate,decay_epoch,batch_ratio,keep_prob,interval,future):
        '''获取数据，设定好batch_size与训练epoch,将数据输入进图开始训练和测试'''

        '''获取数据，初始化模型参数'''
        stock_name, stock_data, mean_fluctuation, merge_test_x, merger_test_y = Data_Processor.get_stocks(
            input_size=self.input_size,
            num_steps=self.num_steps,
            train_ratio=self.train_ratio,
            interval=interval,
            future=future)

        tf.global_variables_initializer().run()

        def _feed_dic(train):
            if train:
                feed_dic = {
                    self.learning_rate: learning_rate,
                    self.keep_prob: keep_prob,
                    self.inputs: batch_x,
                    self.targets: batch_y,
                }
            else:
                feed_dic = {
                    self.learning_rate: 0.0,
                    self.keep_prob: 1.0,
                    self.inputs: merge_test_x,
                    self.targets: merger_test_y,
                }
            return feed_dic

        '''开始训练'''
        for epoch in range(max_epoch):
            '''每轮更新一次学习率'''
            learning_rate = init_learning_rate * (
                decay_rate ** max(float(epoch + 1 - decay_epoch), 0.0)
            )

            '''按股票训练'''
            for data in stock_data:
                train_x = tf.placeholder(data.train_x.dtype,data.train_x.shape)
                train_y = tf.placeholder(data.train_y.dtype,data.train_y.shape)

                # 取batch_size个样本的训练集
                batch_size = int(len(data.train_x)*batch_ratio)
                dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y))
                dataset = dataset.batch(batch_size)
                iterator = dataset.make_initializable_iterator()
                # 初始化参数
                self.sess.run(iterator.initializer, feed_dict={train_x: data.train_x, train_y: data.train_y})
                next_batch = iterator.get_next()
                # 获取下一个batch的数据
                batch_x, batch_y = self.sess.run(next_batch)

                # 训练
                summary, train_optimizer = self.sess.run(
                    [self.merged, self.optimizer], _feed_dic(True)
                )
            self.train_writer.add_summary(summary, epoch)

            # 测试
            test_pred, test_loss = self.sess.run([self.prediction, self.loss], _feed_dic(False))
            print('After epoch',epoch,'the test_loss: ',test_loss)

        # 最终再测试一次
        final_pred, final_loss=self.sess.run([self.prediction, self.loss], _feed_dic(False))
        print('Final,the test_loss: ',final_loss)

        # 计算所有股票的平均预测误差
        sum_error=0
        for i in range(final_pred.shape[0]):
            print('final_pred:',final_pred[i][0],' and the target:',merge_test_y[i][0])
            sum_error += (final_pred[i][0] - merge_test_y[i][0])
        mean_error=sum_error/final_pred.shape[0]

        print('所有股票涨幅的平均波动为：', mean_fluctuation)
        print('在测试集上对于涨跌趋势的预测的平均误差为：', mean_error)
        # 这里借助所有股票涨幅的平均波动来帮助评估模型的预测误差

        self.train_writer.close()
        self.test_writer.close()


def main():
    with tf.Session() as sess:
        lstm_model=LSTM_net(
            sess,
            stock_count=5,
            lstm_shape=[128],
            num_steps=250, # 一年250个交易日
            input_size=10,
            train_ratio=0.9,
            logs_dir='./logs',
            plots_dir='./plots'
        )
        lstm_model.train(max_epoch=30,
                         init_learning_rate=0.001,
                         decay_rate=0.98,
                         decay_epoch=10,
                         batch_ratio=0.8,
                         keep_prob=0.8,
                         interval=30,
                         future=30
                         )

main()
