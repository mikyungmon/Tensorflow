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
    total_batch = int(mnist.trian.num_examples / batch_size)   # 학습데이터의 총 개수인 mnist.trian.num_examples를 배치크기로 나눠 미니 배치가 총 














