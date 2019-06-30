import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
import numpy as np

import openpyxl
path='./accuracy_log.xlsx'
wb=openpyxl.load_workbook(path)
ws=wb['Sheet1']
value=['当前训练轮次','预测股价上涨','预测上涨且正确','预测股票下跌','预测下跌且正确','准确率']
ws.append(value)
wb.save(path)


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
x_test=X_test.iloc[:,-1]     #当前股价
x_test=np.array([x_test]).reshape(-1,1)
print(x_test.shape)
print(x_test.shape[0])
#print(x_test[0,0])

#砍掉一个当前股价特征
X_train=X_train.iloc[:,0:25]
X_test=X_test.iloc[:,0:25]


X_train=scale(X_train)
X_test=scale(X_test)
y_train=np.array([y_train]).reshape(-1,1)
y_test=np.array([y_test]).reshape(-1,1)    #目标股价
print(y_test.shape)
#print(y_test[0:1])

def add_layer(inputs,input_size,output_size,activation_function=None):
    with tf.variable_scope("Weights"):
        Weights=tf.Variable(tf.random_normal(shape=[input_size,output_size]),name='weights')
    with tf.variable_scope("biases"):
        biases=tf.Variable(tf.zeros(shape=[1,output_size])+0.1,name="biases")
    with tf.name_scope("Wx_plus_b"):
        Wx_plus_b=tf.matmul(inputs,Weights)+biases
    with tf.name_scope("dropout"):
        Wx_plus_b=tf.nn.dropout(Wx_plus_b,keep_prob=keep_prob_s)
    if activation_function is None:
        return Wx_plus_b
    else:
        with tf.name_scope("activation_function"):
            return activation_function(Wx_plus_b)


xs=tf.placeholder(shape=[None,X_train.shape[1]],dtype=tf.float32,name="inputs")
ys=tf.placeholder(shape=[None,1],dtype=tf.float32,name="y_true")
keep_prob_s=tf.placeholder(dtype=tf.float32)

with tf.name_scope("layer_1"):
    l1=add_layer(xs,25,30,activation_function=tf.nn.tanh)
    l2=add_layer(l1,30,30,activation_function=tf.nn.tanh)
    l3=add_layer(l2,30,10,activation_function=tf.nn.tanh)
    #l4 = add_layer(l3, 30, 10, activation_function=tf.nn.tanh)

with tf.name_scope("y_pred"):
    pred=add_layer(l3,10,1)
pred=tf.add(pred,0,name="pred")

with tf.name_scope("loss"):
    loss=tf.reduce_mean(tf.reduce_sum(tf.square(ys-pred),reduction_indices=[1]))
    tf.summary.scalar("loss",tensor=loss)
with tf.name_scope("train"):
    train_op=tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)

#draw pics
fig=plt.figure()
ax=fig.add_subplot(1,1,1)
l1=ax.plot(range(100),y_train[0:100],'b',label='train_real_price')
ax.set_ylim([0,50])
plt.ion()
plt.show()

keep_prob=0.9
ITER=3000


def fit(X,y,ax,n,keep_prob):
    init=tf.global_variables_initializer()
    feed_dict_train={ys:y,xs:X,keep_prob_s:keep_prob}       ##################################
    with tf.Session() as sess:
        saver=tf.train.Saver(tf.global_variables(),max_to_keep=15)
        merged=tf.summary.merge_all()
        writer=tf.summary.FileWriter(logdir="bp_stock_log",graph=sess.graph)
        sess.run(init)
        for i in range(n):
            _loss,_=sess.run([loss,train_op],feed_dict=feed_dict_train)
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
                value=[]
                value.append(i)

                print("epoch:%d\tloss:%.5f" %(i,_loss))
                y_pred=sess.run(pred,feed_dict={xs:X_train,ys:y_train,keep_prob_s:1})
                rs=sess.run(merged,feed_dict=feed_dict_train)
                writer.add_summary(summary=rs,global_step=i)

                #保存模型
                #saver.save(sess=sess,save_path="bp_stock_model/bp_stock.model",global_step=i)

                try:
                    ax.lines.remove(lines[0])
                except:
                    pass
                lines=ax.plot(range(100),y_pred[0:100],'g--',lw=2,label='train_predict_price')
                plt.legend()
                plt.pause(1)


                y_pred = sess.run(pred, feed_dict={xs: X_test, ys: y_test, keep_prob_s: 1})
                # 计算准确率
                count = 0     #用来记录预测涨跌正确的股票数
                index = []
                for j in range(x_test.shape[0]):
                    if (y_test[j:j + 1] - x_test[j:j + 1]) * (y_pred[j:j + 1] - x_test[j:j + 1]) >= 0:
                        count += 1.0
                        index.append(j)
                accuracy=count / x_test.shape[0]
                print('准确率：',accuracy)

                #保存模型
                if (count / x_test.shape[0])>0.6:
                    saver.save(sess=sess, save_path="bp_stock_model/bp_stock.model", global_step=i)

                #计算收益
                money=0
                predict_zhang_and_correct=0
                predict_zhang=0
                predict_die=0
                predict_die_and_correct=0
                for j in range(x_test.shape[0]):
                    if y_pred[j:j+1]>=x_test[j:j+1]:
                        predict_zhang+=1
                        money += 10000 * (y_test[j:j + 1] - x_test[j:j + 1])
                    if y_pred[j:j + 1] < x_test[j:j + 1]:
                        predict_die += 1
                    if (y_pred[j:j+1]>=x_test[j:j+1]) and ((y_test[j:j + 1] - x_test[j:j + 1]) * (y_pred[j:j + 1] - x_test[j:j + 1]) >= 0):
                        predict_zhang_and_correct+=1
                    if (y_pred[j:j+1]<x_test[j:j+1]) and ((y_test[j:j + 1] - x_test[j:j + 1]) * (y_pred[j:j + 1] - x_test[j:j + 1]) >= 0):
                        predict_die_and_correct+=1
                print("投资总额：",predict_zhang*10000," 预期收益：",money,"预测涨：",predict_zhang," 预测要涨且正确：",predict_zhang_and_correct)
                print("预测跌：",predict_die,"预测跌且正确：",predict_die_and_correct,'\n')

                value.append(predict_zhang)
                value.append(predict_zhang_and_correct)
                value.append(predict_die)
                value.append(predict_die_and_correct)
                value.append(accuracy)
                print(value)
                ws.append(value)
                wb.save(path)

            if i==(n-1):
                y_pred=sess.run(pred,feed_dict={xs:X_test,ys:y_test,keep_prob_s:1})
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
                    if (y_test[j:j+1]-x_test[j:j + 1])*(y_pred[j:j+1]-x_test[j:j + 1])>=0:
                        count+=1.0
                        index.append(j)
                print(count/x_test.shape[0])
                print(index)
                plt.pause(100)


    wb.save(path)

            #saver.save(sess=sess, save_path="nn_boston_model/nn_boston.model", global_step=n)  # 保存模型

fit(X=X_train,y=y_train,n=ITER,keep_prob=keep_prob,ax=ax)


#setting->Tools->Python Scientific  关闭Show plots in tool window才能用matplotlib画出动态更新的图

#cd C:\Users\music\PycharmProjects\stockPricePredict
#tensorboard --logdir "bp_stock_log"