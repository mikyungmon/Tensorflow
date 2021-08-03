import tensorflow as tf
import numpy as np
import random
from collections import deque


class DQN:
    REPLAY_MEMORY = 10000  # 학습에 사용할 플레이 결과를 얼마나 많이 저장해서 사용할지를 정함
    BATCH_SIZE = 32  # 한번 학습할 때 몇 개의 기억을 사용할지 정함
    GAMMA = 0.99  # 오래된 상태의 가중치를 줄이기 위한 파라미터
    STATE_LEN = 4  # 한 번에 볼 프레임 총 수

    def __init__(self, session, width, height, n_action):
        self.session = session
        self.n_action = n_action
        self.width = width
        self.height = height
        self.memory = deque()  # 게임 플레이 결과를 저장할 메모리를 만드는 코드
        self.state = None

        self.input_X = tf.placeholder(tf.float32, [None, width, height, self.STATE_LEN])  # 게임의 상태를 입력 받음
        self.input_A = tf.placeholder(tf.int64, [None])  # 각 상태를 만들어낸 액션의 값을 입력 받음
        self.input_Y = tf.placeholder(tf.float32, [None])  # 손실값 계산에 사용할 값을 입력 받음

        self.Q = self._build_network('main')
        self.cost, self.train_op = self._build_op()

        self.target_Q = self._build_network('target')  # 목표 신경망은 단순히 Q값을 구하는 데만 사용하므로 손실값과 최적화 함수를 사용하지 않음

    def _build_network(self, name):  # 학습 신경망과 목표 신경망을 구성하는 함수
        with tf.variable_scope(name):
            model = tf.layers.conv2d(self.input_X, 32, [4, 4], padding='same', activation=tf.nn.relu)
            model = tf.layers.conv2d(model, 64, [2, 2], padding='same', activation=tf.nn.relu)
            model = tf.contrib.layers.flatten(model)
            model = tf.layers.dense(model, 512, activation=tf.nn.relu)
            Q = tf.layers.dense(model, self.n_action, activation=None)

        return Q

    def _build_op(self):  # DQN의 손실 함수를 구하는 부분
        one_hot = tf.one_hot(self.input_A, self.n_action, 1.0, 0.0)
        Q_value = tf.reduce_sum(tf.multiply(self.Q, one_hot),
                                axis=1)  # one_hot에는 현재 행동의 인덱스에 해당하는 값에만 1이, 나머지에는 0이 들어있으므로 Q값과 one_hot값을 곱하면 현재 행동의 인덱스에 해당하는 값만 남고 나머지는 전부 0
        cost = tf.reduce_mean(tf.square(
            self.input_Y - Q_value))  # 현재 상태를 이용해 학습 신경망으로 구한 Q_value와 다음 상태를 이용해 목표 신경망으로 구현 Q_value(input_Y)를 이용하여 손실값 구함
        train_op = tf.train.AdamOptimizer(1e-6).minimize(cost)

        return cost, train_op

    # 학습 신경망의 변수들의 값을 목표 신경망으로 복사해서 목표 신경망의 변수들을 최신 값으로 갱신
    def update_target_network(self):  # 목표 신경망을 갱신하는 함수
        copy_op = []
        main_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='main')
        target_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='target')

        for main_var, target_var in zip(main_vars, target_vars):
            copy_op.append(target_var.assign(main_var.value()))

        self.session.run(copy_op)

    def get_action(self):  # 현재 상태를 이용해 다음에 취해야할 행동 찾는 함수
        Q_value = self.session.run(self.Q, feed_dict={self.input_X: [self.state]})
        action = np.argmax(Q_value[0])

        return action

    def init_state(self, state):  # 현재의 상태를 초기화하는 함수
        # DQN에서 입력값으로 사용할 상태는 게임판의 현재 상태 + 앞의 상태 몇개를 합친 것
        state = [state for _ in range(self.STATE_LEN)]
        self.state = np.stack(state, axis=2)  # STATE_LEN크기만큼의 스택으로

    def remember(self, state, action, reward, terminal):  # 게임 플레이 결과를 받아 메모리에 기억하는 기능
        next_state = np.reshape(state, (self.width, self.height, 1))
        next_state = np.append(self.state[:, :, 1:], next_state, axis=2)

        self.memory.append((self.state, next_state, action, reward, terminal))

        if len(self.memory) > self.REPLAY_MEMORY:
            self.memory.popleft()

        self.state = next_state

    def _sample_memory(self):  # 기억해둔 게임 플레이에서 임의의 메로리를 배치 크기만 가져오는 함수
        sample_memory = random.sample(self.memory, self.BATCH_SIZE)

        state = [memory[0] for memory in sample_memory]
        next_state = [memory[1] for memory in sample_memory]
        action = [memory[2] for memory in sample_memory]
        reward = [memory[3] for memory in sample_memory]
        terminal = [memory[4] for memory in sample_memory]

        return state, next_state, action, reward, terminal

    def train(self):
        state, next_state, action, reward, terminal = self._sample_memory()
        target_Q_value = self.session.run(self.target_Q, feed_dict={self.input_X: next_state})

        Y = []
        for i in range(self.BATCH_SIZE):
            if terminal[i]:
                Y.append(reward[i])

            else:
                Y.append(reward[i] + self.GAMMA * np.max(target_Q_value[i]))

        self.session.run(self.train_op, feed_dict={self.input_X: state, self.input_A: action, self.input_Y: Y})