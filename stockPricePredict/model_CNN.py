import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split

from sklearn import preprocessing     #数据预处理


import pandas as pd
import pandas as pd
data2015=pd.read_excel("./data_set/data_set_2015.xlsx")
data2016=pd.read_excel("./data_set/data_set_2016.xlsx")
data2017=pd.read_excel("./data_set/data_set_2017.xlsx")
data2018=pd.read_excel("./data_set/data_set_2018.xlsx")
data=data2015.append(data2016)
data=data.append(data2017)
data=data.append(data2018)


#剔除银行
ex_list = list(data.销售毛利率)
#print(ex_list)
tt=ex_list[0]
for i in range(90):
    ex_list.remove(tt)
data=data[data.销售毛利率.isin(ex_list)]
print(data.shape)



data1=data.iloc[:,3:29]                                     #26个特征输入
data2=data.iloc[:,29]                                       #1个输出结果（所要预测的股价）
X_train,X_test,y_train,y_test=train_test_split(data1,data2,test_size=0.1,random_state=0)
x_test=X_test.iloc[:,-1]
x_test=np.array(x_test).reshape(-1,1)
print(x_test.shape[0])
#print(x_test[0,0])

X_train=X_train.iloc[:,0:25]
X_test=X_test.iloc[:,0:25]


X_train=scale(X_train)
X_test=scale(X_test)
y_train=np.array([y_train]).reshape(-1,1)
y_test=np.array([y_test]).reshape(-1,1)
print(y_test.shape)

#变厚矩阵
def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)     #################
    return tf.Variable(initial)

#偏置
def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

#卷积处理  变厚过程
def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

#pool 长宽缩小一倍
def max_poll_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')


#define placeholder for inputs to network
xs=tf.placeholder(tf.float32,[None,25])   #原始数据的维度：25
ys=tf.placeholder(tf.float32,[None,1])    #输出数据维度：1

keep_prob=tf.placeholder(tf.float32)    #dropout的比例

x_image=tf.reshape(xs,[-1,5,5,1])   #原始数据16列变成二维图片5*5


#conv1 layer   第一卷积层
W_conv1 = weight_variable([2,2,1,32])
b_conv1= bias_variable([32])
h_conv1=tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1)


#conv2 layer   第二卷积层
W_conv2 = weight_variable([2,2,32,64])
b_conv2= bias_variable([64])
h_conv2=tf.nn.relu(conv2d(h_conv1,W_conv2)+b_conv2)

#fcl layer  全连接层
W_fcl=weight_variable([5*5*64,512])
b_fcl=bias_variable([512])

h_pool2_flat=tf.reshape(h_conv2,[-1,5*5*64])
h_fcl=tf.nn.relu(tf.matmul(h_pool2_flat,W_fcl)+b_fcl)
h_fcl_drop=tf.nn.dropout(h_fcl,keep_prob)

W_fc2=weight_variable([512,1])
b_fc2=bias_variable([1])
prediction=tf.matmul(h_fcl_drop,W_fc2)+b_fc2

cross_entropy=tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction),reduction_indices=[1]))
train_step=tf.train.AdamOptimizer(0.01).minimize(cross_entropy)


ITER=3000

#draw pics
fig=plt.figure()
ax=fig.add_subplot(1,1,1)
l1=ax.plot(range(66),y_train[0:66],'b',label='train_real_price')
ax.set_ylim([0,50])
plt.ion()
plt.show()

def fit(X,y,ax,n,keep_prob):
    init=tf.global_variables_initializer()
    feed_dict_train={ys:y,xs:X,keep_prob:1}       ##################################
    feed_dict_test={ys:y_test,xs:X_test,keep_prob:1}
    with tf.Session() as sess:
        saver=tf.train.Saver(tf.global_variables(),max_to_keep=15)
        merged=tf.summary.merge_all()
        writer=tf.summary.FileWriter(logdir="bp_stock_log",graph=sess.graph)
        sess.run(init)
        for i in range(n):
            _loss,_=sess.run([cross_entropy,train_step],feed_dict=feed_dict_train)
            '''if i%100==0:
                print("epoch:%d\tloss:%.5f" %(i,_loss))
                y_pred=sess.run(pred,feed_dict=feed_dict_train)
                rs=sess.run(merged,feed_dict=feed_dict_train)
                writer.add_summary(summary=rs,global_step=i)
                saver.save(sess=sess,save_path="bp_stock_model/bp_stock.model",global_step=i)
                try:
                    ax.lines.remove(lines[0])
                except:
                    pass
                lines=ax.plot(range(66),y_pred[0:66],'r--')
                path="./picture/"+str(i)+'.jpg'
                plt.savefig(path)
            if i==(n-1):
                y_pred=sess.run(pred,feed_dict={xs:X_test,ys:y_test,keep_prob_s:keep_prob})
                try:
                    ax.lines.remove(lines[0])
                except:
                    pass
                ax.plot(range(40), y_test, 'yellow')
                lines=ax.plot(range(40),y_pred[0:40],'r--')
                path = "./picture/test.jpg"
                plt.savefig(path)'''

            if i%100==0:
                print("epoch:%d\tloss:%.5f" %(i,_loss))
                y_pred=sess.run(prediction,feed_dict=feed_dict_train)
                # rs=sess.run(merged,feed_dict=feed_dict_test)
                # writer.add_summary(summary=rs,global_step=i)
                # saver.save(sess=sess,save_path="bp_stock_model/bp_stock.model",global_step=i)
                try:
                    ax.lines.remove(lines[0])
                except:
                    pass
                lines=ax.plot(range(66),y_pred[0:66],'g--',lw=2,label='train_predict_price')
                plt.legend()
                plt.pause(1)


                y_pred = sess.run(prediction, feed_dict=feed_dict_test)
                # 计算准确率
                count = 0
                index = []
                for j in range(x_test.shape[0]):
                    if (y_test[j:j + 1] - x_test[j:j + 1]) * (y_pred[j:j + 1] - x_test[j:j + 1]) >= 0:
                        count += 1.0
                        index.append(j)
                print('准确率：',count / x_test.shape[0])


            if i==(n-1):
                y_pred=sess.run(prediction,feed_dict=feed_dict_test)
                try:
                    ax.lines.remove(lines[0])
                except:
                    pass
                ax.lines.remove(l1[0])
                ax.plot(range(x_test.shape[0]),x_test,'black',lw=2,label='test_present_price')
                ax.plot(range(x_test.shape[0]), y_test, 'yellow',lw=3,label='test_real_price')
                lines=ax.plot(range(x_test.shape[0]),y_pred[0:x_test.shape[0]],'r--',label='test_predict_price')
                plt.legend()

                #计算准确率
                count=0
                index=[]
                for j in range(x_test.shape[0]):
                    print(j,y_test[j:j+1])
                    print(j,y_pred[j:j + 1])
                    print(j,x_test[j:j + 1])

                    if (y_test[j:j+1]-x_test[j:j + 1])*(y_pred[j:j+1]-x_test[j:j + 1])>=0:
                        count+=1.0
                        index.append(j)
                print(count/x_test.shape[0])
                print(index)
                plt.pause(100)

            #saver.save(sess=sess, save_path="nn_boston_model/nn_boston.model", global_step=n)  # 保存模型

fit(X=X_train,y=y_train,n=ITER,keep_prob=keep_prob,ax=ax)