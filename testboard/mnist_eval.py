# _*_ coding:utf-8 _*_
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_inference
from mnist_train import MOVING_AVERAGE_DECAY,REGULARAZTION_RATE,LEARNING_RATE_BASE,BATCH_SIZE,LEARNING_RATE_DECAY,TRAINING_STEPS,MODEL_SAVE_PATH,MODEL_NAME
INPUT_NODE = 784
OUTPUT_NODE = 10
LAYER1_NODE = 500


def get_weight_variable(shape,regularizer):
    weights = tf.get_variable(
        "weights",shape,
        initializer=tf.truncated_normal_initializer(stddev=0.1)
    )
    if regularizer != None:
        tf.add_to_collection('losses',regularizer(weights))
    return weights
def inference(input_tensor,regularizer):
    with tf.variable_scope('layer1'):
        weights = get_weight_variable([INPUT_NODE,LAYER1_NODE],regularizer)
        biases = tf.get_variable("biases",[LAYER1_NODE],initializer=tf.constant_initializer(0.0))
        layer1 = tf.nn.relu(tf.matmul(input_tensor,weights)+biases)
    with tf.variable_scope("layer2"):
        weights = get_weight_variable([LAYER1_NODE,OUTPUT_NODE], regularizer)
        biases = tf.get_variable("biases", [OUTPUT_NODE], initializer=tf.constant_initializer(0.0))
        layer2 = tf.nn.relu(tf.matmul(layer1, weights) + biases)
    return layer2
def train(mnist):
    with tf.name_scope("input"):
        x = tf.placeholder(tf.float32,[None,mnist_inference.INPUT_NODE],name="x-input")
        y_ = tf.placeholder(tf.float32,[None,mnist_inference.OUTPUT_NODE],name="y-cinput")
    regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)
    y = inference(x,regularizer)
    global_step = tf.Variable(0,trainable=False)
    with tf.name_scope("moving_average"):
        variable_averages = tf.train.ExponentialMovingAverage(
            MOVING_AVERAGE_DECAY,global_step
        )
        variable_averages_op = variable_averages.apply(tf.trainable_variables())
    with tf.name_scope("loss_funcation"):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=tf.argmax(y_,1))
        cross_entropy_mean = tf.reduce_mean(cross_entropy)
        loss = cross_entropy_mean+tf.add_n(tf.get_collection('losses'))

    with tf.name_scope("train_step"):
        learning_rate = tf.train.exponential_decay(
            LEARNING_RATE_BASE,
            global_step,
            mnist.train.num_examples/BATCH_SIZE,
            LEARNING_RATE_DECAY,
            staircase=True
        )
        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)
    with tf.control_dependencies([train_step, variable_averages_op]):
        train_op = tf.no_op(name='train')

        # 初始化Tensorflow持久化类
        #saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        # 在训练过程中不再测试模型在验证数据上的表现，验证和测试的过程将会有一个独立的程序来完成
        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: xs, y_: ys})

            # 每1000轮保存一次模型
            if i % 1000 == 0:
                # 输出当前训练情况。这里只输出了模型在当前训练batch上的损失函数大小
                # 通过损失函数的大小可以大概了解训练的情况。在验证数据集上的正确率信息
                # 会有一个单独的程序来生成
                print("After %d training step(s),loss on training batch is %g" % (step, loss_value))

                # 保存当前的模型。注意这里给出了global_step参数，这样可以让每个被保存模型的文件末尾加上训练的轮数
                # 比如"model.ckpt-1000"表示训练1000轮之后得到的模型
                #saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)
    writer = tf.summary.FileWriter("/Users/wqq/tensorlog",tf.get_default_graph())
    writer.close()

def main(argv=None):
    mnist = input_data.read_data_sets('/tmp/data/',one_hot=True)
    train(mnist)

if __name__ == '__main__':
    tf.app.run()