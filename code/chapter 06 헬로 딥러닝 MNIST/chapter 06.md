# Chapter 06

이번 장에서는 MNIST 데이터셋을 신경망으로 학습시키는 방법을 알아볼 것이다. 

MNIST는 손으로 쓴 숫자들의 이미지를 모아놓은 데이터셋으로, 0부터 9까지의 숫자를 28 * 28 픽셀 크기의 이미지로 구성해놓은 것이다.

MNIST 데이터셋은 다음 주소에서 내려받을 수 있다.

http://yann.lecun.com/exdb/mnist/

## 6.1 MNIST 학습하기

MNIST 데이터셋이 매우 잘 정리되어 있지만 사용하려면 데이터를 내려받고 읽어 들이고 나누고 신경망 학습에 적합한 형식으로 처리하는 번거로운 과정을 거쳐야한다.

1️⃣ 먼저 텐서플로를 임포트하고 추가로 텐서플로에 내장된 tensorflow.examples.tutorials.mnist.input_data 모듈을 임포트한다.

    import tensorflow as tf
    
    from tensorflow.examples.tutorials.mnist import input_data
    
    mnist = input_data.read_data_set("./mnist/data/", one_hot = True)  
   
  - 마지막 줄에서 MNIST 데이터를 내려받고 레이블은 동물 분류 예제에서 본 원-핫 인코딩 방식으로 읽어 들인다.
  
  - MNIST 데이터셋을 사용하기 위한 준비는 이게 끝이다.

2️⃣ 신경망 모델을 구성해보자. MNIST의 손글씨 이미지는 28 * 28 픽셀로 이루어져 있다. 이는 다시 말해 784개의 특징으로 이루어져 있다고 할 수 있다. 그리고 레이블은 0부터 9까지이니 10개의 분류로 나누면 된다. 따라서 입력과 출력인 X와 Y를 다음과 같이 정의한다.

    X = tf.placeholder(tf.float32, [None,784])
    Y = tf.placeholder(tf.float32, [None,10])
    
 ❗ 이미지를 하나씩 학습시키는 것보다 여러 개를 한꺼번에 학습시키는 쪽이 효과가 좋지만 많은 메모리와 높은 컴퓨팅 성능이 뒷받침 되어야하기 때문에 일반적으로 데이터를 적당한 크기로 잘라서 학습시키는데 이것을 **미니배치(minibatch)** 라고 한다.
 
  - 앞의 X와 Y코드에서 텐서의 첫 번째 차원이 None으로 되어있는 것을 볼 수 있는데 이 자리에는 **한 번에 학습시킬 MNIST이미지의 개수를 지정하는 값**이 들어간다. 즉 배치 크기를 지정하는 자리이다.

  - 원하는 배치 크기로 정확히 명시해줘도 되지만 한 번에 학습할 개수를 계속 바꿔가면서 실험해보려는 경우에는 None으로 넣어주몀ㄴ 텐서플로라 알아서 계산한다.

이제 2개의 은닉층이 다음 처럼 구성된 신경망을 만들어 볼 것이다.

    784(입력, 특징 개수) -> 256 (첫 번째 은닉층 뉴런 개수) ->  256 (두 번째 은닉층 뉴런 개수) -> 10 (결과값 0~9 분류 개수)
    
이를 코드로 구성하면 다음과 같다.

    W1 = tf.Variable(tf.random_normal([784,256], stddev = 0.01))   # 표준편차가 0.01인 정규분포를 가지는 임의의 값으로 뉴런을 초기화
    L1 = tf.nn.relu(tf.matmul(X,W1))    # tf.matmul함수를 이용하여 각 계층으로 들어오는 입력값에 각각의 가중치를 곱하고 tf.nn.relu함수를 활성화 함수로 ReLU를 사용하는 신경망 계층 만듦
    
    W2 = tf.Variable(tf.random_normal([256,256], stddev = 0.01))
    L2 = tf.nn.relu(tf.matmul(L1,W2))
    
    W3 = tf.Variable(tf.random_normal([256,10], stddev = 0.01))
    model = tf.matmul(L2,W3)   # 요소 10개짜리 배열이 출력됨. 가장 큰 값을 갖는 인덱스가 예측 결과에 가까운 숫자. 출력층에는 보통 활성화 함수를 사용하지 않음
    
3️⃣ 다음으로 tf.nn.softmax_cross_entropy_with_logits함수를 이용해 각 이미지에 대한 손실값을 구하고 tf.reduce_mean함수를 이용해 미니배치의 평균 손실값을 구한다. 그리고 tf.train.AdamOptimizer함수를 이용하여 이 손실값을 최소화하는 최적화를 수행하도록 그래프를 구성한다.

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = model, labels = Y))
    optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)
    
4️⃣ 앞에서 구성한 신경망 모델을 초기화하고 학습을 진행할 세션을 시작한다.

    init = tf.global_variables_initializer()    # 신경망 모델 초기화
    sess = tf.Session()
    sess.run(init)
    
다음은 학습을 실제로 진행하는 코드를 작성해야하지만 그에 앞서 학습용 데이터와 테스트용 데이터에 대해 알아보자.

💡 테스트용 데이터를 왜 따로 구분할까?

  - 머신러닝을 위한 학습 데이터는 항상 학습용과 테스트용으로 분리해서 사용한다.
  - 학습용 데이터는 학습을 시킬 때 사용하고 테스트 데이터는 학습이 잘 되었는지를 확인하는데 사용한다. 
  - 별도의 테스트 데이터를 사용하는 이유는 학습 데이터로 예측을 하면 예측 정확도가 매우 높게 나오지만 학습 데이터에 포함되지 않은 새로운 데이터를 예측할 때는 정확도가 매우 떨어지는 경우가 많기 때문이다. 
  - 이처럼 학습 데이터는 예측을 매우 잘하지만 실제 데이터는 그렇지 못한 상태를 **과적합(overfitting)** 이라고 한다.
  - 이러한 형상을 확인하고 방지하기 위해 학습 데이터와 테스트 데이터를 분리하고 학습이 끝나면 항상 테스트 데이터를 사용하여 학습 결과를 검증해야한다.
  - MNIST데이터셋은 학습 데이터 6만 개와 테스트 데이터 1만 개로 구성돼 있다. 텐서플로 이용하면 쉽게 사용가능하다. mnist.train을 사용하면 학습데이터를 mnist.test를 사용하면 테스트 데이터를 사용할 수 있다.

5️⃣ 학습을 진행한다.

    batch_size = 100   # 미니배치 크기를 100으로 설정
    total_batch = int(mnist.trian.num_examples / batch_size)   # 학습데이터의 총 개수인 mnist.trian.num_examples를 배치크기로 나눠 미니 배치가 총 몇 개인지를 저장해둠

그리고 MNIST데이터 전체를 학습하는 일을 총 15번 반복한다(학습 데이터 전체를 한 바퀴 도는 것을 에포크라고 한다).

    for epoch in range(15):
        total_cost = 0
        
다음 반복문에서 미니배치의 총 개수만큼 반복하여 학습한다.
    
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)    # mnist.train.next_batch(batch_size) 함수를 이용해 학습할 데이터를 배치 크기만큼 가져옴
        
        _,cost_val = sess.run([optimizer,cost], feed_dict = {X:batch_xs, Y:batch_ys})   # 가져온 뒤 입력값인 이미지 데이터는 batch_xs에, 출력값인 레이블 데이터는 batch_ys에 저장
        total_cost +=cost_val
        print('Epoch:', '$04d' %(epoch +1), 'Avg. cost = ', '{:3.3f}'.format(total_cost / total_batch))
        
    print('최적화 완료!')
    
 - sess.run을 이용하여 최적화시키고 손실값을 가져와서 저장한다. 이 때 feed_dict매개변수에 입력값 X와 예측을 평가할 실제 레이블 값 Y에 사용할 데이터를 넣어준다.
 
 - 손실값을 저장한 다음 한 세대의 학습이 끝나면 학습한 세대의 평균 손실을 구한다.

6️⃣ 학습이 잘 되었는지 결과를 출력해보자. 다음 코드는 예측 결과인 model의 값과 실제 레이블 Y의 값을 비교한다.

    is_correct = tf.equal(tf.argmax(model,1), tf.argmax(Y,1))
    
 - 예측한 결괏값은 원-핫 인코딩 형식이며 각 인덱스에 해당하는 값은 다음처럼 해당 숫자가 얼마나 해당 인덱스와 관련이 높은가를 나타낸다. 즉, 값이 가장 큰 인덱스가 가장 근접한 예측결과라는 말이다.
 
 - tf.argmax(model,1)은 두 번째 차원(1번 인덱스의 차원)의 값 중 최댓값의 인덱스를 뽑아내는 함수이다. model로 출력한 결과는 [None,10]처럼 결과값을 배치 크기만큼 가지고 있다. 따라서 두 번째 차원이 예측한 각각의 결과이다.
 
 - 그러므로 tf.argmax(Y,1)로 실제 레이블에 해당하는 숫자를 가져온다. 그런 다음 맨 바깥의 tf.equal함수를 통해 예측한 숫자와 실제 숫자가 같은지 확인한다.
 
7️⃣ 이제 tf.cast를 이용해 is_correct를 0과 1로 변환한다. 그리고 변환한 값들을 tf.reduce_mean을 이용해 평균을 내며 그게 바로 정확도(확률)이 된다.

    accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
    
그리고 다음처럼 테스트 데이터를 다루는 객체인 mnist.test를 이용해 테스트 이미지와 레이블 데이터를 넣어 accuracy를 계산한다.

    print('정확도:',sess.run(accuracy, feed_dict = {X: mnist.test.images, Y: mnist.test.labels}))
    
## 6.2 드롭아웃

앞서 설명한 내용 중 '과적합'이라는 것이 있었다. 다시 설명하자면 과적합이란 학습한 결과가 학습 데이터에는 매우 잘 맞지만, 학습 데이터에만 너무 꼭 맞춰져 있어서 그 외의 데이터에는 잘 맞지 않는 상황을 말한다.

과적합 문제는 머신러닝의 가장 중요한 과제 중 하나여서 이를 해결하기 위한 매우 다양한 방법이 있는데 그중 가장 효과가 좋은 방법 하나가 바로 **드롭아웃(dropout)** 이다.

💡 **드롭아웃의 원리는 간단한데 학습 시 전체 신경망 중 일부만 사용하도록 하는 것이다.** 즉, 학습 단계마다 일부 뉴런을 제거(사용하지 않도록)함으로써 일부 특징이 특정 뉴런들에 고정되는 것을 막아 가중치의 균형을 잡도록하여 과적합을 방지한다.

1️⃣ 앞에서 만든 손글씨 인식 모델에 드롭아웃 기법을 적용해본다. 드롭아웃 역시 텐서플로가 기본으로 지원해주기 때문에 다음과 같이 아주 간단하게 적용할 수 있다.

    W1 = tf.Variable(tf.random_normal([784,256], stddev = 0.01))
    L1 = tf.nn.relu(tf.matmul(X,W1))
    L1 = tf.nn.dropout(L1,0.8)   # 0.8은 사용할 뉴런의 비율. 이 경우 학습 시 해당 계층의 약 80% 뉴런만 사용하겠다는 의미
    
    W2 = tf.Variable(tf.random_normal([256,256],stddev = 0.01))
    L2 = tf.nn.relu(tf.matmul(L1,W2))
    L2 = tf.nn.dropout(L2,0.8)
    
❗ 드롭아웃 기법을 사용해 학습하더라도, **학습이 끝난 뒤 예측 시에는 신경망 전체를 사용하도록 해줘야한다.**

따라서 다음과 같이 keep_prob라는 플레이스홀더를 만들어, 학습 시에는 0.8을 넣어 드롭아웃을 사용하도록 하고 예측 시에는 1을 넣어 신경망 전체를 사용하도록 만들어야한다.
    
    keep_prob = tf.placeholder(tf.float32)
    
    W1 = tf.Variable(tf.random_normal([784,256], stddev = 0.01))
    L1 = tf.nn.relu(tf.matmul(X,W1))
    L1 = tf.nn.dropout(L1,keep_prob)   # 0.8은 사용할 뉴런의 비율. 이 경우 학습 시 해당 계층의 약 80% 뉴런만 사용하겠다는 의미
    
    W2 = tf.Variable(tf.random_normal([256,256],stddev = 0.01))
    L2 = tf.nn.relu(tf.matmul(L1,W2))
    L2 = tf.nn.dropout(L2,keep_prob)
    
    #학습코드 : keep_prob를 0.8로 넣어준다
    _,cost_val = sess.run([optimizer,cost]. feed_dict = {X:batch_xs, Y:batch_ys, keep_prob : 0.8})
    
    #예측코드 : keep_prob를 1로 넣어준다
    print('정확도:',sess.run(accuracy, feed_dict = {X:mnist.test.images, Y:mnist.test.labels, keep_prob : 1}))
 
 - 드롭아웃 기법을 적용한 뒤 학습을 진행해보면 아마도 드롭아웃을 적용하지 않은 때와 별 차이 없는 결과를 보게 될 것인데 이유는 드롭아웃을 사용하면 학습이 느리게 진행되기 때문이다.
 
 - 그렇다면 epoch를 30번으로 늘려 조금 더 많이 학습하도록 해본다. 그렇게 30번을 학습 시킨 결과는 드롭아웃을 사용하지 않은 결과보다 조금 더 나은 결과가 나왔다.
 
 - 드롭아웃을 사용하지 않은 모델도 30번을 학습시키면 정확도가 높아질까? 과적합으로 인해 오히려 더 낮아진다.
    
## 6.3 matplotlib

**matplotlib**은 시각화를 위해 그래프를 쉽게 그릴 수 있도록 해주는 파이썬 라이브러리이다. 

이번 절에서는 matplotlib을 이용하여 학습 결과를 손글씨 이미지로 확인해보는 간단한 예제를 만들어본다.

1️⃣ 다음 코드를 앞 절에서 작성한 코드에 추가하면 된다. 먼저 matplotlib의 pyplot모듈을 임포트한다.

    import matplotlib.pyplot as plt
    
2️⃣ 테스트 데이터를 이용해 예측 모델을 실행하고 결과값을 labels에 저장한다.

    labels = sess.run(model, feed_dict = {X:mnist.test.images, Y:mnist.test.labels, keep_prob :1})
    
3️⃣ 그런 다음 손글씨를 출력할 그래프를 준비한다.

    fig = plt.figure()
    
4️⃣ 테스트 데이터의 첫 번째부터 열 번째까지의 이미지와 예측한 값을 출력한다.

    for i in range(10):
        # 2행 5열의 그래프 만들고 i+1번째에 숫자 이미지를 출력한다
        subplot = fig.add_subplot(2,5,i+1)
        #이미지를 깨끗하게 출력하기 위해 x와 y의 눈금을 출력하지 않는다
        subplot.set_xticks([])
        subplot.set_yticks([])
        #출력한 이미지 위에 예측한 숫자를 출력한다
        #np.argmax는 tf.argmax와 같은 기능의 함수이다
        #결과값인 labels의i번째 요소가 원-핫 인코딩 형식으로 되어있으므로 해당 배열에서 가장 높은 값을 가진 인덱스를 예측한 숫자로 출력한다
        subplot.set_title('%d' % np.argmax(labels[i]))
        #1차원 배열로 되어있는 i번째 이미지 데이터를 28 * 28형식의 2차원 배열로 변형하여 이미지 형태로 출력한다. cmap파라미터를 통해 이미지를 그레이스케일로 출력한다
        subplot.imshow(mnist.test.images[i].reshape((28,28)),cmap = plt.cm.gray_r)
        
5️⃣ 마지막으로 그래프를 화면에 표시한다.

    plt.show()

- 코드를 실행하면 손글씨 이미지를 출력하고 그 위에는 예측한 숫자를 출력한다.




