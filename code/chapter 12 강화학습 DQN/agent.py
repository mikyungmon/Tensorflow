import tensorflow as tf
import numpy as np
import random
import time

from game import Game
from model import DQN

# 에이전트는 학습 모드와 게임 실행 모드로 나뉨
tf.app.flags.DEFINE_boolean("train", False,
                            "학습모드. 게임을 화면에 보여주지 않습니다.")  # 에이전트 실행 시 모드를 나누어 실행할 수 있도록 tf.app.flags를 이용해 실행 시 받을 옵션들을 설정

FLAGS = tf.app.flags.FLAGS

# 하이퍼파라미터 설정
MAX_EPISODE = 10000  # 최대로 학습할 게임 횟수
TARGET_UPDATE_INTERVAL = 1000  # 학습을 일정 횟수만큼 진행할 때마다 한번씩 목표 신경망을 갱신하라는 옵션
TRAIN_INTERVAL = 4  # 게임 4프레임(상태)마다 한 번씩 학습하라는 이야기
OBSERVE = 100  # 일정 수준의 학습 데이터가 쌓이기 전에는 학습하지 않고 지켜보기만 하라는 의미. 100번의 프레임이 지난 뒤부터 학습을 진행

NUM_ACTION = 3  # 행동 - 0: 좌, 1: 유지, 2: 우
SCREEN_WIDTH = 6
SCREEN_HEIGHT = 10


def train():
    print('뇌세포 깨우는 중..')
    sess = tf.Session()

    game = Game(SCREEN_WIDTH, SCREEN_HEIGHT, show_game=False)  # 게임 객체에는 화면 크기 넣어주고 학습 속도를 위해 게임 화면은 출력 안함
    brain = DQN(sess, SCREEN_WIDTH, SCREEN_HEIGHT, NUM_ACTION)  # DQN 모델 객체를 생성

    rewards = tf.placeholder(tf.float32, [None])  # 에피소드(한 판)마다 얻는 점수를 저장하고 확인하기 위한 텐서
    tf.summary.scalar('avg.reward/ep.', tf.reduce_mean(rewards))

    # 학습 결과 저장 및 학습 상태 확인
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())

    writer = tf.summary.FileWriter('logs', sess.graph)  # 로그 저장위한 객체
    summary_merged = tf.summary.merge_all()  # 학습 상태를 확인하기 위한 값들을 모아서 저장하기 위한 텐서

    # 목표 신경망 한 번 초기화
    brain.update_target_network()

    epsilon = 1.0
    time_step = 0  # 학습 진행 조절 위해 진행된 프레임(상태)횟수
    total_reward_list = []  # 학습 결과를 확인하기 위한 점수들을 저장할 배열을 초기화

    for episode in range(MAX_EPISODE):
        terminal = False  # terminal은 게임의 종료 상태
        total_reward = 0  # total_reward는 한 게임당 얻은 총 점수

        state = game.reset()  # 게임의 상태 초기화
        brain.init_state(state)  # 그 상태를 DQN에 초기 상태값으로 넣어줌

        while not terminal:
            if np.random.rand() < epsilon:  # 행동을 무작위로 선택
                action = random.randrange(NUM_ACTION)
            else:
                action = brain.get_action()

            if episode > OBSERVE:  # 일정 시간(에피소드 100번)이 지난 뒤 epsilon값을 조금씩 줄임
                epsilon -= 1 / 1000

            state, reward, terminal = game.step(action)
            total_reward += reward

            brain.remember(state, action, reward, terminal)  # 현재 상태를 신경망 객체에 기억

            if time_step > OBSERVE and time_step % TRAIN_INTERVAL == 0:  # 현재 프레임이 100번(OBSERVE)이 넘으면 4프레임(TRAIN_INTERVAL)마다 한 번씩 학습을 진행
                brain.train()

            if time_step % TARGET_UPDATE_INTERVAL == 0:  # 1,000프레임(TARGET_UPDATE_INTERVAL)마다 한 번씩 목표 신경망을 갱신
                brain.update_target_network()

            time_step += 1

        print('게임 횟수: %d 점수: %d' % (episode + 1, total_reward))

        total_reward_list.append(total_reward)

        if episode % 10 == 0:  # 에피소드 10번마다 받은 점수를 로그에 저장
            summary = sess.run(summary_merged, feed_dict={rewards: total_reward_list})
            writer.add_summary(summary, time_step)
            total_reward_list = []

        if episode % 100 == 0:  # 100번마다 학습된 모델을 저장
            saver.save(sess, 'model/dqn.ckpt', global_step=time_step)


## 이제 학습 결과를 실행하는 함수를 작성
def replay():
    print('뇌세포 깨우는 중..')
    sess = tf.Session()

    game = Game(SCREEN_WIDTH, SCREEN_HEIGHT, show_game=True)
    brain = DQN(sess, SCREEN_WIDTH, SCREEN_HEIGHT, NUM_ACTION)

    # 텐서플로 세션을 새로 생성하지 않고 tf.train.Saver()로 저장해둔 모델을 읽어와서 생성
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state('model')
    saver.restore(sess, ckpt.model_checkpoint_path)

    for episode in range(MAX_EPISODE):
        terminal = False
        total_reward = 0

        state = game.reset()
        brain.init_state(state)

        while not terminal:
            action = brain.get_action()

            state, reward, terminal = game.step(action)
            total_reward += reward

            brain.remember(state, action, reward, terminal)

            time.sleep(0.3)

        print('게임 횟수: %d 점수: %d' % (episode + 1, total_reward))


# 스크립트 학습용으로 실행할지 학습된 결과로 게임을 진행할지 선택하는 부분
def main(_):
    if FLAGS.train:
        train()
    else:
        replay()


if __name__ == "__main__":
    tf.app.run()