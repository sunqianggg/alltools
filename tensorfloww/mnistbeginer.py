#encoding:utf8
'''
来自tensorflow官方文档：http://tensorfly.cn/tfdoc/tutorials/mnist_beginners.html
该程序先是介绍了mnist数据集，mnist数据集是手写识别数据集，它包含6000个训练数据
1000个测试数据，每一个数据包含28X28个像素，label为[1,0,0...]这样的形式.

然后介绍了softmax函数如何对多分类结果进行表达。

最后介绍了用tensorflow实现多分类.
'''

import input_data
import tensorflow as tf

mnist=input_data.read_data_sets("MNIST_data/",one_hot=True)

#对于x这个待输入的自变量使用占位函数来表示其shape和数据类型
x=tf.placeholder("float",[None,784])
#对于w,b这样的参数使用variable并且使用０来初始化
W=tf.Variable(tf.zeros([784,10]))
b=tf.Variable(tf.zeros([10]))
#对与y直接使用将x,w,b的线性组合带入softmax中
y=tf.nn.softmax(tf.matmul(x,W)+b)
"""
到这里一个一次性的的tensorflow计算图就完成了，注意输入只有x,w,b都是先定义后修正的。
"""

#为了修正w,b参数和真实的数据y对比，将待输入的真实数据y用占位符表示
y_=tf.placeholder("float",[None,10])
#计算交叉熵，此处就是损失函数
cross_entropy=-tf.reduce_sum(y_*tf.log(y))
#此处用梯度下降法来最小化损失函数
train_step=tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

"""
到这里完整的计算图定义完毕，包括模型定义，迭代定义。
"""

#进行启动计算图之前的预热阶段，初始化变量，也就是W和b，此处已经赋值为何还要初始化?
init=tf.initialize_all_variables()
sess=tf.Session()
sess.run(init)

#开始真正的训练，1000次，每次随机取100个数据，1000*100=100000,而训练数据有60000个，说明多取了，有重复的，可能也有没取到的。
for i in range(1000):
    batch_xs,batch_ys=mnist.train.next_batch(100)
    sess.run(train_step,feed_dict={x:batch_xs,y_:batch_ys})

#开始评估模型的准确性
correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,"float"))

#为什么不能打印模型
print(sess.run(accuracy,feed_dict={x:mnist.test.images,y_:mnist.test.labels}))


#问题：
#数据保存问题
#为什么多次进行sess.run，比如计算accuracy时为什么还要run,它和sklearn里面的train,fit有何区别?
#该模型还能打印出更多东西吗？
#这里面没哟激活函数，损失函数和迭代方法选别的是怎样的情景，反向传播是怎样做的?