# Chapter 12

**DQN**은 딥마인드에서 나온 신경망이다. DQN은 게임 화면만 보고 게임을 학습하는 신경망으로 2014년에 공개했다.

딥마인드에서 이 DQN으로 아타리 2600용 비디오 게임 49개를 학습시킨 결과 모두 잘 학습하여 플레이했고 그중 29개에서는 사람의 평균 기록보다 높은 점수를 보였다.

이번 장에서는 DQN의 개념을 간단히 살펴보고 직접 구현한 뒤 간단한 게임을 학습시켜본다.

## 12.1 DQN 개념

**DQN은 Deep Q-network의 줄임말인데 강화학습 알고리즘으로 유명한 Q-러닝을 딥러닝으로 구현했다는 의미이다.**

💡 강화학습이란? 

  - 어떤 환경에서 인공지능 에이전트가 현재 상태(환경)를 판단하여 가장 이로운 행동을 만드는 학습 방법이다.
  
  - 학습 시 이로운 행동을 하면 보상을 해주고 해로운 행동을 하면 페널티를 줘서 학습이 진행될 수 있도록 이로운 행동을 점점 많이 하도록 유도한다.
  
  - 즉 누적된 이득이 최대가 되게 행동하도록 학습이 진행된다.

**Q-러닝**은 어떠한 상태에서 특정 행동을 했을 때의 가치를 나타내는 함수인 **Q 함수**를 학습하는 알고리즘이다.

즉, 특정 상태에서 이 함수의 값이 최대가 되는 행동을 찾도록 학습하면 그 상태에서 어떤 행동을 취해야 할지 알 수 있게 된다.

그리고 이 Q함수를 신경망을 활용해 학습하게 한 것이 바로 DQN이다.

하지만 Q-러닝을 신경망으로 구현하면 학습이 상당히 불안정해진다.

이에 딥마인드는 다음의 두 가지 방법을 사용하여 이 문제를 해결하였다.

1) 먼저 과거의 상태를 기억한 뒤 그중에서 임의의 상태를 뽑아 학습시키는 방법을 사용한다. 이렇게 하면 특수한 상황에 치우치지 않도록 학습시킬 수 있어서 더 좋은 결과를 내는데 도움이 된다.

2) 두 번째로는 손실값 계산을 위해 학습을 진행하면서 최적의 행동을 얻어내는 **기본 신경망**과 얻어낸 값이 좋은 선택인지 비교하는 **목표(target) 신경망**을 분리하는 방법을 사용한다. 목표 신경망은 계속 갱신하는 것이 아니라 기본 신경망의 학습된 결과값을 일정 주기마다 목표 신경망에 갱신해준다.

그리고 DQN은 화면의 상태, 즉 화면 영상만으로 게임을 학습한다. 따라서 이미지 인식에 뛰어난 CNN을 사용하여 신경망 모델을 구성하였다.

(사진) DQN기본개념

## 12.2 게임 소개

OpenAI는 인공지능의 발전을 위해 다양한 실험을 할 수 있는 Gym이라는 강화학습 알고리즘 개발 도구를 제공한다.

이 도구를 이용하면 아타리 게임들을 쉽게 구동할 수 있다. 다만, 아타리 게임을 학습시키려면 매우 성능 좋은 컴퓨터가 필요하고 시간도 아주 오래걸린다.

시간이 너무 오래 걸리는 환경은 공부용으로 적절하지 않아서 학습을 빠르게 시켜볼 수 있는 다음 그림과 같은 간단한 게임을 사용한다.

(사진)

이 게임은 아래로 떨어지는 물체를 피하는 간단한 게임이다. 

소스는 깃허브 저장소에서 내려받을 수 있다(https://goo.gl/VQ9JDT).

인터페이스는 일부러 OpenAI Gym과 거의 같게 만들었다.

## 12.3 에이전트 구현하기

**에이전트**는 게임의 상태를 입력받아 신경망으로 전달하고 신경망에서 판단한 행동을 게임에 적용해서 다음 단계로 진행한다.

그러므로 에이전트가 어떤 식으로 작동하는지 알아야 신경망 구현 시 이해가 더 수월할 것이기 때문에 신경망 모델을 구현하기 전에 에이전트부터 구현해본다.

1️⃣ 먼저 필요한 모듈을 임포트한다. Game모듈을 앞서 말한 저장소에서 내려받은 game.py 파일에 정의된 게임 클래스이다. 이 모듈은 게임을 진행하고 필요 시 matplotlib을 써서 게임 상태를 화면에 출력해준다. DQN 모듈은 다음 절에서 구현할 신경망 모델이다.

    import tensorflow as tf
    import numpy as np
    import random
    import time
    
    from game import Game
    from model import DQN
    
2️⃣ 에이전트는 학습 모드(train)와 게임 실행 모드(replay)로 나뉜다. 학습 모드 때는 게임을 화면에 보여주지 않은 채 빠르게 실행하여 학습 속도를 높이고 게임 실행 모드에서는 학습한 결과를 이용해 게임을 진행하면서 화면에도 출력해준다.

이를 위해 에이전트 실행 시 모드를 나누어 실행할 수 있도록 tf.app.flags를 이용해 실행 시 받을 옵션들을 설정한다.

    tf.app.flags.DEFINE_boolean("train", False, "학습모드. 게임을 화면에 보여주지 않습니다.")
    
    FLAGS = tf.app.flags.FLAGS
    
3️⃣ 하이퍼파라미터들을 설정한다.

    MAX_EPISODE = 10000    # 최대로 학습할 게임 횟수
    TARGET_UPDATA_INTERVAL = 1000   # 학습을 일정 횟수만큼 진행할 때마다 한번씩 목표 신경망을 갱신하라는 옵션
    TRAIN_INTERVAL = 4    # 게임 4프레임(상태)마다 한 번씩 학습하라는 이야기
    OBSERVE = 100   # 일정 수준의 학습 데이터가 쌓이기 전에는 학습하지 않고 지켜보기만 하라는 의미. 100번의 프레임이 지난 뒤부터 학습을 진행

   - DQN은 안정된 학습을 위해 학습을 진행하면서 최적의 행동을 얻어내는 기본 신경망과 얻어낸 값이 좋은 선택인지 비교하는 목표 신경망이 분리되어 있다. 가장 최근의 학습 결과나 아주 오래된 학습 결과로 현재의 선택을 비교한다면 적절한 비교가 되지 않을 것이다.

   ➡ 목표 신경망 갱신할 때, 아주 오래된 학습 결과로 비교한다면 차이가 많이 나게되고 최근의 학습 결과로 비교하면 차이가 너무 적게되어서 적절한 비교를 하려면 1000이 적당하다고 설명하는 듯?
   
   - 따라서 적당한 시점에 최신의 학습 결과로 목표 신경망을 갱신해줘야 한다.

4️⃣ 다음은 게임 자체에 필요한 설정이다. 떨어지는 물건을 좌우로 움직이면서 피하는 게임이므로 취할 수 있는 행동은 좌, 우, 상태유지 이렇게 세 가지이다. 또한 게임 화면은 가로 6칸, 세로 10칸으로 설정하였다. 원래 이 설정은 Game 모듈에 들어있는 것이 맞지만 이해를 위해 에이전트에 넣었다.

    NUM_ACTION = 3   # 행동 - 0: 좌, 1: 유지, 2: 우
    SCREEN_WIDTH = 6
    SCREEN_HEIGHT = 10

앞서 말한 거처럼 에이전트는 학습시키는 부분과 학습된 결과로 게임을 실행해보는 두 부분으로 나뉘어 있다. 학습 부분부터 만들어본다.

5️⃣ 먼저 텐서플로 세션과 게임 객체, DQN 모델 객체를 생성한다. 게임 객체에는 화면 크기를 넣어주고 학습을 빠르게 진행하기 위해 게임을 화면에 출력하지 않을 것이므로 show_game옵션을 False로 준다. DQN객체에는 신경망을 학습시키기 위해 텐서플로 세션을 넣어주고, 화면을 입력받아 CNN을 구성할 것이므로 화면 크기를 넣어 초기 설정을 한다. 그리고 가장 중요한 신경망의 최종 결과값의 개수인 선택할 행동의 개수(NUM_ACTION)를 넣어준다.

    def train():
        print('뇌세포 깨우는 중..')
        sess = tf.Session()
        game = Game(SCREEN_WIDTH, SCREEN_HEIGHT, show_game = False)
        brain = DQN(sess, SCREEN_WIDTH, SCREEN_HEIGHT, NUM_ACTION)
        
6️⃣ 다음으로는 학습 결과를 저장하고 학습 상태를 확인하는 코드를 작성한다. 

    rewards = tf.placeholder(tf.float32, [None])    # 에피소드(한 판)마다 얻는 점수를 저장하고 확인하기 위한 텐서
    tf.summary.scalar('avg.reward/ep./', tf.reduce_mean(rewards))
    
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    
    writer = tf.summary.FileWriter('logs', sess.graph)   # 로그 저장위한 객체
    summary_merged = tf.summary.merge_all()   # 학습 상태를 확인하기 위한 값들을 모아서 저장하기 위한 텐서

 - 에피소드 10번에 한 번씩 로그를 저장할 것이고 그때 rewards의 평균을 저장할 것이다.
 
 - 학습 결과를 저장하기 위해 tf.train.Saver와 텐서플로 세션, 그리고 로그를 저장하기 위한 tf.summary.FileWriter 객체를 생성하고, 학습 상태를 확인하기 위한 값들을 모아서 저장하기 위한 텐서를 설정한다.

7️⃣ 그리고 목표 신경망을 한 번 초기화해준다. 아직 학습된 결과가 없으므로 여기서 목표 신경망의 값은 초기화된 기본 신경망의 값과 같다.

    brain.update_target_network()
    
8️⃣ 다음에 설정하는 epsilon값은 행동을 선택할 때 DQN을 이용할 시점을 정한다. 학습 초반에는 DQN이 항상 같은 값만 내놓을 가능성이 높다. 따라서 일정 시간이 지나기 전에는 행동을 무작위로 선택해야한다. 이를 위해 게임 진행 중에 epsilon값을 줄여나가는 코드를 넣을 것이다.

    epsilon = 1.0
    time_step = 0    # 학습 진행 조절 위해 진행된 프레임(상태)횟수
    total_reward_list = []    # 학습 결과를 확인하기 위한 점수들을 저장할 배열을 초기화

9️⃣ 게임을 진행하고 학습시키는 부분을 작성한다. 앞서 설정한 MAX_EPISODE 횟수만큼 게임 에피소드를 진행하며 매 게임을 시작하기 전에 초기화한다.

    for episode in range(MAX_EPISODE):
        terminal = False   # terminal은 게임의 종료 상태
        total_reward = 0    # total_reward는 한 게임당 얻은 총 점수
        state = game.reset()    # 게임의 상태 초기화
        brain.init_state(state)    # 그 상태를 DQN에 초기 상태값으로 넣어줌
        
  - 상태는 screen_width * screen_height 크기의 화면 구성이다 -> 게임 화면 자체를 말하는 듯. 캐릭터가 오른쪽으로 움직여서 화면이 바뀌면 다른 상태

❗ 원래 DQN에서는 픽셀값들을 상태값으로 받지만 여기서 사용하는 Game모듈에서는 학습 속도를 높이고자 해당 위치에 사각형이 있는지 없는지를 1과 0으로 전달한다.

🔟 이제 게임 에피소드 한 번 진행한다. 녹색 사각형이 다른 사각형에 충돌할 때까지이다. 게임에서 처음으로 해야할 일은 다음에 취할 행동을 선택하는 일이다. 학습 초반에는 행동을 무작위로 선택하고 일정 시간(에피소드 100번)이 지난 뒤 epsilon값을 조금씩 줄여간다. 그러면 초반에는 대부분 무작위 값을 사용하다가 무작위 값을 사용하는 비율이 점점 줄어들어 나중에는 거의 사용하지 않게 된다. 이 값들도 하이퍼파라미터이므로 잘 조절해야 한다.

    with not terminal:
        if np.random.rand() < epsilon:   # np.random.rand()하면 처음에는 0~1값이 나오니까 무조건 epsilon값인 1보다 작게 되어 무조건 무작위로 뽑게 되는거고 epsilon값이 줄면 무작위로 뽑게 되는 횟수가 줄어드는 것
            action = random.randrange(NUM_ACTION)
        else : 
            action = brain.get_action()
            
        if episode > OBSERVE :
            epsilon -=1/1000
        
1️⃣1️⃣ 그런 다음 앞서 결정한 행동을 이용해 게임을 진행하고 보상과 게임의 종료 여부를 받아온다.

    state, reward, terminal = game.step(action)
    total_reward += reward
    
1️⃣2️⃣ 그리고 현재 상태를 신경망 객체에 기억시킨다. 이 기억한 현재 상태를 이용해 다음 상태에서 취할 행동을 결정한다. 또한 여기서 저장한 상태, 행동, 보상들을 이용하여 신경망을 학습시킨다.

    brain.remember(state, action, reward, terminal)

1️⃣3️⃣ 현재 프레임이 100번(OBSERVE)이 넘으면 4프레임(TRAIN_INTERVAL)마다 한 번씩 학습을 진행한다. 또한 1,000프레임(TARGET_UPDATE_INTERVAL)마다 한 번씩 목표 신경망을 갱신해준다.

    if time_step > OBSERVE and time_step % TRAIN_INTERVAL == 0:
        brain.train()
    if time_step % TARGET_UPDATE_INTERVAL == 0:
        brain.update_target_network()
        
    time_stpe +=1

1️⃣4️⃣ 사각형들이 충돌해 에피소드(게임)가 종료되면 획득한 점수를 출력하고 이번 에피소드에서 받은 점수를 저장한다. 그리고 에피소드 10번마다 받은 점수를 로그에 저장하고, 100번마다 학습된 모델을 저장한다.

    print('게임횟수: %d 점수 : %d' %(episode +1, total_reward))
    
    total_reward_list.append(total_reward)
    
    if episode % 10 == 0 :
        summary = sess.run(summary_merged, feed_dict = {rewards : total_reward_list})
        writer.add_summary(summary,time_step)
        total_reward_list = []
        
    if episode % 100 == 0:
    saver.save(sess, 'model/dqn.ckpt', global_step = time_step) 

여기까지가 학습을 시키는 에이전트 코드이다. 

1️⃣5️⃣ 이제 학습 결과를 실행하는 함수를 작성한다. 결과를 실행하는 replay()함수는 학습 부분만 빠져있을 뿐 train()함수와 거의 같다. 한 가지 주의할 점은 텐서플로 세션을 새로 생성하지 않고 tf.train.Saver()로 저장해둔 모델을 읽어와서 생성해야 한다는 점이다.

    def replay():
        print('뇌세포 깨우는 중..')
        sess = tf.Session()
        
        game = Game(SCREEN_WIDTH, SCREEN_HEIGHT, show_game = True)
        brain = DQN(sess, SCREEN_WIDTH, SCREEN_HEIGHT, NUM_ACTION)
        
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state('model')
        saver.restore(sess, ckpt.model_checkpoint_path)
        
1️⃣6️⃣ 다음은 게임을 진행하는 부분이다. 학습 코드가 빠져있어 매우 단순하다. 게임 진행을 인간이 인지할 수 있는 속도로 보여주기 위해 마지막 부분에 time.sleep(0.3)코드를 추가했다는 것만 주의하자.

    for episode in range(MAX_EPISODE):
        terminal = False
        total_reward = 0
        
        state = game.reset()
        brain.init_state(state)
        
        while not terminal:
            action = brain.get_action()
            state, reward, terminal = game.step(action)
            total_reward +=reward
            
            brain.remember(state, action, reward, terminal)
            
            time,sleep(0.3)
            
       print('게임 횟수: %d 점수: %d' % (episode +1, total_reward))
       
1️⃣7️⃣ 마지막으로 스크립트 학습용으로 실행할지 학습된 결과로 게임을 진행할지 선택하는 부분이다. 이는 터미널이나 명령 프롬프트에서 스크립트를 실행할 때 train 옵션을 받아 정하게 했다.

    der main(_):
        if FLAGS.train:
            train()
        else :
            replay()
            
    if __name__ == '__main__' :
        tf.app.run()
        
## 12.4 신경망 모델 구현하기

게임을 진행하고 신경망을 학습할 에이전트를 구현하였으니 이제 DQN을 구현해본다.

1️⃣ DQN 구현에 필요한 모듈들을 임포트한다.

    import tensorflow as tf
    import numpy as np
    import random
    from collections import deque
    
2️⃣ 지금까지 작성해온 코드와는 다르게 DQN을 파이썬 클래스로 작성한다. 과거의 상태들을 기억하고 사용하는 긴으을 조금 더 구조적으로 만들고 기능별로 나누어 이해와 수정을 쉽게 하기 위함이다.

    class DQN :
        REPLAY_MEMORY = 10000   # 학습에 사용할 플레이 결과를 얼마나 많이 저장해서 사용할지를 정함
        BATCH_SIZE = 32    # 한번 학습할 때 몇 개의 기억을 사용할지 정함
        GAMMA = 0.99   # 오래된 상태의 가중치를 줄이기 위한 파라미터
        STATE_LEN = 4    # 한 번에 볼 프레임 총 수
        
   - 에이전트에서 DQN에 상태를 넘겨줄 때는 한 프레임의 상태만 넘겨준다. 다만 DQN이 상태를 받으면 해당 상태만이 아니라 STATE_LEN -1개의 앞 상태까지 고려해서 현재의 상태로 만들어 저장한다. 즉 , 이전 상태까지 고려하는 것이다.

3️⃣ 다음은 DQN 객체를 초기화하는 코드이다. 텐서플로 세션과 가로/세로 크기, 그리고 행동의 개수를 받아 객체를 초기화 한다.

    def __init__(self, session, width, heights, n_action):
        self.session = session
        self.n_action = n_action
        self.width = width
        self.height = height
        self.memory = deque()   # 게임 플레이 결과를 저장할 메모리를 만드는 코드
        self.state = None
        
   - collections모듈의 deque()함수로 만들어진 객체는 배열과 비슷하지만 맨 처음에 들어간 요소를 쉽게 제거해주는 popleft함수를 제공한다. 저장할 기억의 개수를 제한하는 데 사용한다.

4️⃣ 그런 다음 DQN에서 사용할 플레이스홀더들을 설정한다.

    self.input_X = tf.placeholder(tf.float32, [None,width,height, self.STATE_LEN])   # 게임의 상태를 입력 받음
    self.input_A = tf.placeholder(tf.int64, [None])   # 각 상태를 만들어낸 액션의 값을 입력 받음
    self.input_Y = tf.placeholder(tf.float32, [None])    # 손실값 계산에 사용할 값을 입력 받음
    
 - input_X의 구조는 보는 바 처럼 [게임판의 가로 크기, 게임판의 세로 크기, 게임 상태의 개수(현재+과거+과거..)]형식이다.
 
 - input_A는 원-핫 인코딩이 아닌 행동을 나타내는 숫자를 그대로 받아서 사용한다.
 
 - input_Y는 보상에 목표 신경망으로 구한 다음 상태의 Q값을 더한 값이다. 여기에서 학습 신경망에서 구한 Q값을 뺀 값을 손실값으로 하여 학습을 진행한다.
 
 - Q값은 행동에 따른 가치를 나타내는 값으로 이때 목표 신경망에서 구한 Q값을 구한 값 중에서 최대값을, 학습 신경망에서 구한 Q값은 현재 행동에 따른 값을 사용한다.
 
 - 이렇게 하면 행동을 선택할 때 가장 가치가 높은 행동을 선택하도록 학습할 것이다.
 
5️⃣ 다음으로 학습을 진행할 신경망과 목표 신경망을 구성한다. 두 신경망을 구성이 같으므로 신경망을 구성하는 함수를 같은 것을 사용하되 이름만 다르게 한다. 목표 신경망은 단순히 Q값을 구하는 데만 사용하므로 손실값과 최적화 함수를 사용하지 않는다.

    self.Q = self._build_network('main')   # 학습 신경망을 거쳐서 나온 Q-value값
    self.cost, self.train_op = self._build_op()
    
    self.target_Q = self.build_network('target')
    
6️⃣ 다음의 _ build_network는 앞서 나온 학습 신경망과 목표 신경망을 구성하는 함수이다. 상태값 input_X를 받아 행동의 가짓수만큼 출력을 만든다. 이 값들의 최대값을 취해 다음 행동을 정한다. 

    def _build_network(self,name):
        with tf.variable_scope(name):
            model = tf.layers.conv2d(self.input_X, 32, [4,4], padding = 'same', activation = tf.nn.relu)
            model = tf.layers.conv2d(model, 64, [2,2], padding = 'same', activation = tf.nn.relu)
            model = tf.contrib.layers.flatten(model)
            model = tf.layers.dense(model, 512, activation = tf.nn.relu)
            
            Q = tf.layers.dense(model, self.n_action, activation = None)
            
        return Q
    
7️⃣ 다음은 DQN의 손실 함수를 구하는 부분이다. 현재 상태를 이용해 학습 신경망으로 구한 Q_value와 다음 상태를 이용해 목표 신경망으로 구현 Q_value(input_Y)를 이용하여 손실값을 구하고 최적화한다.

    def _build_op(self):
        one_hot = tf.one_hot(self.input_A, self.n_action, 1.0, 0.0)    # 만약 액션이 3개면 크기가 3이고 input_A에 해당하는 액션만 1이고 나머지는 0이 됨
        Q_value = tf.reduce_sum(tf.multiply(self.Q, one_hot), axis =1)   # tf.multiply(self.Q, one_hot)는 self.Q로 구한 값에서 현재 행동의 인덱스에 해당하는 값만 선택하기 위해 사용
        cost = tf.reduce_mean(tf.square(self.input_Y - Q_value))    # 목표 신경망에서 나온 Q value값 - 학습 신경망에서 나온 Q value값 
        train_op = tf.train.AdamOptimizer(1e-6).minimize(cost)
        
        return cost, train_op
        
 - one_hot에는 현재 행동의 인덱스에 해당하는 값에만 1이, 나머지에는 0이 들어있으므로 Q값과 one_hot값을 곱하면 현재 행동의 인덱스에 해당하는 값만 남고 나머지는 전부 0 이 된다.

8️⃣ 다음은 목표 신경망을 갱신하는 함수이다. 학습 신경망의 변수들의 값을 목표 신경망으로 복사해서 목표 신경망의 변수들을 최신 값으로 갱신한다.

    def updata_target_network(self):
        copy_op = []
        main_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'main')
        target_varts = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'target')
        
        for main_var, target_var in zip(main_vars, target_vars):
            copy_op.accpend(target_var.assign(main_var.value()))
            
        self.session.run(copy_op)
        
9️⃣ get_action은 현재 상태를 이용해 다음에 취해야 할 행동을 찾는 함수이다. _ build_network 함수에서 계산한 Q_value를 이용한다. 출력값이 원-핫 인코딩 되어 있으므로 np.argmax함수를 이용해 최댓값이 담긴 index값을 행동값으로 취한다.

    def get_action(self):
        Q_value = self.session.run(self.Q, feed_dict = {self.input_X : [self.state]})
        action = np.argmax(Q_value[0])
        
        return action
        
  - 이것으로 학습에 필요한 텐서와 연산이 모두 준비되었다.
  
  - CNN을 사용한 신경망으로 Q-value를 구하고 이 Q-value를 이용해 학습에 필요한 손실 함수를 만들었다. 
  
  - 그리고 DQN의 핵심인 목표 신경망을 학습 신경망의 값으로 갱신하는 함수와 Q_value를 이용해 다음 행동을 판단하는 함수를 만들었다.

🔟 학습을 시키는 코드를 작성한다.

먼저 _ sample_memory 함수를 사용해 게임 플레이를 저장한 메모리에서 배치 크기만큼을 샘플링하여 가져온다.

    def train(self):
        state, next_state, action, reward, terminal = self._sample_memory()
        
그런 다음 가져온 메모리에서 다음 상태를 만들어 목표 신경망에 넣어 target_Q_value를 구한다. 현재 상태가 아닌 다음 상태를 넣는다는 점에 유의한다.

    target_Q_value = self.session.run(self.target_Q, feed_dict = {self.input_X : next_state})   # 왜 next_state넣는지?
    
그리고 앞서 만든 손실 함수에 보상값을 입력한다. 단, 게임이 종료된 상태라면 보상값을 바로 넣고 게임이 진행중이라면 보상값에 target_Q_value의 최대값을 추가하여 넣는다. 현재 상황에서의 최대 가치를 목표로 삼기 위함이다.
 
    Y = []
    for i in range(self.BATCH_SIZE):
        if terminal[i] :
            Y.append(reward[i])
            
        else:
            Y.append(reward[i] + self.GAMMA * np.max(target_Q_value[i]))
            
마지막으로 AdamOptimizer를 이용한 최적화 함수에 게임 플레이 메모리에서 가져온 현재 상태값들과 취한 행동, 그리고 앞서 구한 Y값을 넣어 학습시킨다.
 
    self.session.run(self.train_op, feed_dict = {self.input_X :state, self.input_A : action, self.input_Y: Y})
    
✔ 이제 몇 가지 헬퍼 함수들을 만들고 구현을 마무리한다. init_state, remember, _ sample_memory로 학습에 사용할 상태값을 만들고 메모리에 저장하고 추출해오는 함수들이다.

1️⃣1️⃣ init_state는 현재의 상태를 초기화하는 함수이다. DQN에서 입력값으로 사용할 상태는 게임판의 현재 상태 + 앞의 상태 몇개를 합친 것이다. 이를 입력값으로 만들기 위해 STATE_LEN크기만큼의 스택으로 만들어둔다.

    def init_state(self,state):
        state = [state for _ in range(self.STATE_LEN)]
        self.state = np.stack(state,axis=2)   # axis =2옵션은 input_X에 넣을 플레이스홀더를 [None, width, height, self.STATE_LEN]구성으로 만들었기 때문
        
1️⃣2️⃣ remember함수는 게임 플레이 결과를 받아 메모리에 기억하는 기능을 수행한다. 가장 오래된 상태를 제거하고 새로운 상태를 넣어 다음 상태로 만들어둔다. 입력받은 새로운 상태가 DQN으로 취한 행동을 통해 만들어진 상태이므로 실제론ㄴ 다음 상태라고 불 수 있기 때문이다.

메모리에는 게임의 현재 상태와 다음 상태, 취한 행동과 그 행동으로 얻어진 보상, 그리고 게임 종료 여부를 저장해둔다. 그리고 너무 오래된 과거까지 기억하려면 메모리가 부족할 수도 있고, 학습에도 효율적이지 않으므로 저장할 플레이 개수를 제한한다.

    def remember(self,state, action, reward, terminal):
        next_state = np.reshape(state,(self.width, self.height, 1)
        next_state = np.append(self.state[:,:,1:], next_state, axis =2)
        
        self.memory.append((self.state, next_state, action, reward, terminal))
        
        if len(self.memory) > self.REPLAY_MEMORY:
            self.memory.popleft()
            
        self.state= next_state
        
1️⃣3️⃣ _ sample_memory 함수는 기억해둔 게임 플레이에서 임의의 메로리를 배치 크기만큼 가져온다. random.sampe함수를 통해 임의의 메모리를 가져오고 그중 첫 번째 요소를 현재 상태값으로, 두 번째를 다음 상태값으로, 그리고 취한 행동, 보상, 게임 종료 여부를 순서대로 가져온 뒤 사용하기 쉽도록 재구성하여 넘겨준다.

    def _sample_memory(self):
        sample_memory = random.sample(self.memory, self.BATCH_SIZE)
        
        state = [memory[0] for memory in sample_memory]
        next_state= [memory[1] for memory in sample_memory]
        action = [memory[2] for memory in sample_memory]
        reward = [memory[3] for memory in sample_memory]
        terminal = [memory[4] for memory in sample_memory]
        
        return state, next_state, action, reward, terminal
        
## 12.5 학습시키기 

소스 파일이 있는 곳에서 터미널이나 명령 프롬프트를 열고 다음 명령을 입력해주면 된다.

    C:\> python agent.py --train
    
한가지 주의할 것은 코드가 있는 곳에 logs와 model디렉터리를 미리 만들어둬야 한다. 해당 디렉터리에 게임 점수에 대한 로그와 학습된 모델을 저장하기 때문이다. 
        
학습이 많이 진행되지 않았더라도 학습을 종료한 뒤 다음 명령을 이용하여 결과를 확인해볼 수 있다.

    C:\> python agent.py
    
또한 텐서보드를 사용하면 언제부터 얼마나 똑똑해졌는지 확인해볼 수 있다. 텐서보드를 띄어 게임 점수가 높아지는 모습을 그래프로 확인해보자.

    C:\> tensorboard --logdir=./logs


















































































