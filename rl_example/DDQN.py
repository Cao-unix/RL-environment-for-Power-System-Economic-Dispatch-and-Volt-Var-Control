import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def calc_loss(q_eval, n_actions, lr):
    q_target_feed = tf.placeholder(
        tf.float32, [None, n_actions], name='Q_target')    # for calculating loss
    with tf.variable_scope('loss'):
        loss = tf.reduce_mean(
            tf.squared_difference(q_target_feed, q_eval))
    with tf.variable_scope('train'):
        train_op = tf.train.RMSPropOptimizer(lr).minimize(loss)
    return loss, train_op, q_target_feed


# 建立两个结构完全相同的神经网络，q_target和q_eval，神经网络的输出为本状态下各个动作的Q值
def MLP_Q(n_features, n_actions, name):
    # ------------------ build evaluate_net ------------------
    s = tf.placeholder(tf.float32, [None, n_features], name='s')  # input
    with tf.variable_scope(f'{name}_net'):
        # c_names(collections_names) are the collections to store variables
        c_names, n_l1, n_l2, w_initializer, b_initializer = \
            [f'{name}_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 256, 256, \
            tf.random_normal_initializer(0., 0.3), tf.constant_initializer(
                0.1)            # config of layers

        # first layer. collections is used later when assign to target net
        with tf.variable_scope('l1'):
            w1 = tf.get_variable('w1', [n_features, n_l1],
                                 initializer=w_initializer, collections=c_names)
            b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer,
                                 collections=c_names)
            l1 = tf.nn.relu(tf.matmul(s, w1) + b1)

        # second layer. collections is used later when assign to target net
        with tf.variable_scope('l2'):
            w2 = tf.get_variable(
                'w2', [n_l1, n_l2], initializer=w_initializer, collections=c_names)
            b2 = tf.get_variable(
                'b2', [1, n_l2], initializer=b_initializer, collections=c_names)
            l2 = tf.matmul(l1, w2) + b2

        # third layer. collections is used later when assign to target net
        with tf.variable_scope('l3'):
            w3 = tf.get_variable(
                'w3', [n_l2, n_actions], initializer=w_initializer, collections=c_names)
            b3 = tf.get_variable(
                'b3', [1, n_actions], initializer=b_initializer, collections=c_names)
            q = tf.matmul(l2, w3) + b3
    return q, s

# Deep Q Network off-policy


class DDQN:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.2,
            e_greedy=0.9,
            greedy_start=0.9,
            e_greedy_increment=None,
            replace_target_iter=300,
            memory_size=500,
            batch_size=256,
            output_graph=False,
            flag=1
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.epsilon_increment = e_greedy_increment
        self.epsilon = greedy_start if e_greedy_increment is not None else self.epsilon_max
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.restore_flag = flag

        # total learning step，学习次数
        self.learn_step_counter = 0

        # consist of [target_net, evaluate_net]
        self.q_eval, self.s = MLP_Q(n_features, n_actions, 'eval')

        self.loss, self._train_op, self.q_target_feed = calc_loss(
            self.q_eval, n_actions, learning_rate)

        self.sess = tf.Session()

        self.sess.run(tf.global_variables_initializer())
        self.cost_his = []

        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

    # epsilon-贪婪选择动作
    def choose_action(self, observation):
        # to have batch dimension when feed into tf placeholder
        observation = observation[np.newaxis, :]

        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions
            actions_value = self.sess.run(
                self.q_eval, feed_dict={self.s: observation})
            action = np.argmax(actions_value)
        else:
            # 函数的作用是，返回一个随机整型数，范围从低（包括）到高（不包括），即[low, high)
            action = np.random.randint(0, self.n_actions)
        return action

    # 神经网络学习
    def learn(self, batch_memory):
        q_target = np.zeros((1, self.n_actions))
        # 计算q_next, q_eval的值,q_next是用下一时刻状态输入q_target网络得到的，
        # 用来计算q_target值，q_eval是用本时刻状态输入q_eval网络得到的
        q_next = self.sess.run(
            [self.q_eval],
            feed_dict={
                self.s: batch_memory[:, :self.n_features],
            })

        # 提取每个样本中动作及奖励，以备更新神经网络参数
        eval_act_index = int(batch_memory[:, self.n_features])
        reward = batch_memory[0, self.n_features + 1]

        # 目标Q值:r+y*maxQ
        q_target[0, eval_act_index] = reward + \
            self.gamma * np.max(q_next[0], axis=1)

        """
        For example in this batch I have 2 samples and 3 actions:
        q_eval =
        [[1, 2, 3],
         [4, 5, 6]]

        q_target = q_eval =
        [[1, 2, 3],
         [4, 5, 6]]

        Then change q_target with the real q_target value w.r.t the q_eval's action.
        For example in:
            sample 0, I took action 0, and the max q_target value is -1;
            sample 1, I took action 2, and the max q_target value is -2:
        q_target =
        [[-1, 2, 3],
         [4, 5, -2]]

        So the (q_target - q_eval) becomes:
        [[(-1)-(1), 0, 0],
         [0, 0, (-2)-(6)]]

        We then backpropagate this error w.r.t the corresponding action to network,
        leave other action as error=0 cause we didn't choose it.
        """
        # train eval network，用样本数组前面的观测值对应神经网络的featrues和q_target
        # 训练q_eval神经网络，将损失提取为cost
        _, self.cost = self.sess.run([self._train_op, self.loss],
                                     feed_dict={self.s: batch_memory[:, :self.n_features],
                                                self.q_target_feed: q_target})

        # 将cost附到cost_his中
        self.cost_his.append(self.cost)

    # 绘制cost曲线
    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()
