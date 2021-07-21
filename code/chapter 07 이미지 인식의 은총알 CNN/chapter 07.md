# Chapter 07

이번 장에서 다룰 합성곱 신경망, 즉 CNN(Convolutional Neural Network)은 1998년 얀 레쿤 교수가 소개한 이래로 널리 사용되고 있는 신경망으로 이미지 인식 분야에서는 거의 은총알이라고 할 정도로 강력한 성능을 발휘하고 있다.

또한 최근에는 음성인식이나 자연어 처리 등에도 사용되며 활용성에서도 매우 뛰어난 성과를 보여주고 있다.

이미지 인식 분야의 절대 강자인 CNN을 이용하여 MNIST 데이터를 학습시켜 보고 앞서 배운 기본적인 신경망보다 성능이 얼마나 좋아지는지 확인해본다.

## 7.1 CNN 개념

CNN 모델은 기본적으로 **컨볼루션 계층**과 **풀링 계층**으로 구성된다.

이 계층들을 얼마나 많이 또 어떠한 방식으로 쌓느냐에 따라 성능 차이는 물론 풀 수 있는 문제가 달라질 수 있다.

컨볼루션 계층과 풀링 계층의 개념은 매우 간단하다. 

💡 2D 컨볼루션의 경우, 2차원 평면 행렬에서 지정한 영역의 값들을 하나의 값으로 압축하는 것이다. 

![image](https://user-images.githubusercontent.com/66320010/126479605-9ec426cd-473f-492e-bddc-4c9bb723faa0.png)

단, 하나의 값으로 압축할 때 **컨볼루션 계층**은 가중치와 편향을 적용하고 **풀링 계층**은 단순히 값들 중 하나를 선택해서 가져오는 방식을 취한다.

위 그림과 같이 지정한 크기의 영역을 윈도우라고 하며 이 윈도우의 값을 오른쪽, 그리고 아래쪽으로 한 칸씩 움직이면서 은닉층을 완성한다. 

움직이는 크기는 변경할 수 있으며  몇 칸씩 움직일지 정하는 값은 **스트라이드**라고 한다.

이렇게 입력층의 윈도우를 은닉층의 뉴런 하나로 압축할 때, 컨볼루션 계층에서는 윈도우 크기만큼의 가중치와 1개의 편향을 적용한다.

예를 들어 윈도우 크기가 3 * 3 이라면 3 * 3개의 가중치와 1개의 편향이 필요하다. 이 3 * 3개의 가중치와 1개의 편향을 **커널** 또는 **필터**라고 하며 이 커널은 해당 은닉층을 만들기 위한 모든 윈도우에 공통으로 적용된다.

➡ 이것이 바로 CNN의 가장 중요한 특징 중의 하나인데, 예를 들어 입력층이 28 * 28개라고 했을 때 기본 신경망으로 모든 뉴런을 연결한다면 784개의 가중치를 찾아야하지만 컨볼루션 계층에서는 3 * 3개인 9개의 가중치만 찾아내면 된다. 계산량이 매우 적어져 학습이 더 빠르고 효율적으로 이뤄진다.

이제 앞 장에서 하용한 MNIST 데이터를 CNN으로 학습시키는 모델을 만들어 볼 것이다.

## 7.2 모델 구현하기

앞장과 동일한 코드를 실행한다.

    import tensorflow as tf
    from tensorflow.examples.tutorial.mnist import input_data
    mnist = input_data.read_data_sets("./mnist/data/", one_hot = True)
    
1️⃣ 앞 장에서 만든 모델에서는 입력값을 28 * 28짜리 차원 하나로 구성했지만, CNN모델에서는 앞서 설명한 것처럼 2차원 평면으로 구성하므로 다음처럼 조금 더 직관적인 형태로 구성할 수 있다.

    X = tf.placeholder(tf.float32, [None,28, 28, 1])   # None : 입력 데이터의 개수 , 1 : 특징의 개수(MNIST데이터가 회색 이미지라 채널이 1이므로 1을 사용)
    Y = tf.placeholder(tf.float32, [None,10])   # 출력값이 10개의 분류
    keep_prob = tf.placeholder(tf.float32)    # 드롭아웃을 위한 플레이스홀더

2️⃣ 첫 번째 CNN 계층을 구성해본다. 우선 3 * 3크기의 채널을 가진 컨볼루션 계층을 만든다. 다음처럼 커널에 사용할 가중치 변수와 텐서플로가 제공하는 tf.nn.conv2d 함수를 사용하면 간단하게 구성할 수 있다.

    w1 = tf.Variable(tf.random_normal([3,3,1,32], stddev= 0.01))   # 입력층 X와 첫 번째 계층의 가중치 W1을 가지고 32개의 커널을 가진 컨볼루션 계층 만들기
    L1 = tf.nn.conv2d(X,W1,strides = [1,1,1,1], padding= 'SAME')   # padding = 'SAME'은 커널 슬라이딩 시 이미지의 가장 외곽에서 한 칸 밖으로 움직이는 옵션
    L1 = tf.nn.relu(L1)
    
3️⃣ 풀링 계층 만든다. 풀링 계층 역시 다음과 같이 텐서플로가 제공하는 함수로 매우 간단하게 작성할 수 있다.

    L1 = tf.nn.max_pool(L1, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')   # padding = 'SAME' 옵션하면 이미지의 크기가 변하지 않음
    
  - 앞서 만든 컨볼루션 계층을 입력층으로 사용하고 커널 크기를 2 * 2로 하는 풀링 계층을 만든다.
  
  - strides = [1,2,2,1] 값은 슬라이딩 시 두 칸씩 움직이겠다는 옵션이다.

![image](https://user-images.githubusercontent.com/66320010/126487822-70c87023-17ca-440f-bb28-f14ab45a0f95.png)

위 사진은 CNN의 첫 번째 계층의 구성을 나타낸 그림이다.

4️⃣ 두 번째 계층을 동일한 방식으로 구성해본다.

    W2 = tf.Variable(tf.random_normal([3,3,32,64], stddev = 0.01))   # [3,3,32,64]에서 32는 첫 번째 컨볼루션 계층의 커널 개수. 첫 번째 컨볼루션이 찾아낸 이미지의 특징 개수라고 할 수 있음
    L2 = tf.nn.conv2d(L1,W2, strides = [1,1,1,1], padding = 'SAME')
    L2 = tf.nn.relu(L2)
    L2 = tf.nn.max_pool(L2, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')

5️⃣ 이제 추출한 특징들을 이용해 10개의 분류를 만들어내는 계층을 구성한다.

    W3 = tf.Variable(tf.random_normal([7*7*64,256], stddev = 0.01))
    L3 = tf.reshape(L2, [-1, 7*7*64])
    L3 = tf.matmul(L2,W3)
    L3 = tf.nn.relu(L3)
    L3 = tf.nn.dropout(L3,keep_prob)
    
  - 10개의 분류는 1차원 배열이므로 차원을 줄이는 단계를 거쳐야 한다.
  
  - 직전 풀링 계층 크기가  7 * 7 * 64 이므로 먼저 tf.reshape 함수를 이용해 7 * 7 * 64 크기의 1차원 계층으로 만들고 이 배열 전체를 최종 출력값의 중간 단계인 256개의 뉴런으로 연결하는 신경망을 만들어준다.
  
  - 이처럼 인접한 계층의 모든 뉴런과 상호 연결되는 계층을 **완전 연결 계층(fully connected layer)** 이라고 한다.
  
  - 이번 계층에서는 추가로 과적합을 막아주는 드롭아웃 기법을 사용했다.

6️⃣ 모델 구성의 마지막으로 직전의 은닉층인 L3의 출력값 256개를 받아 최종 출력값인 0~9 레이블을 갖는 10개의 출력 값을 만든다.

    W4 = tf.Variable(tf.random_normal([256,10], stddev = 0.01))
    model = tf.matmul(L3,W4)
    
7️⃣ 손실함수와 AdamOptimizer를 이용한 최적화 함수를 만든다.

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = model, labels = Y))
    optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)
    
 나중에 최적화 함수를 RMSPropOptimizer로 바꿔서 결과를 비교해보는 것도 좋다.
 
    optimizer = tf.train.RMSPropOptimizer(0.001,0.9).minimize(cost)
    
8️⃣ 학습을 시키고 결과를 확인하는 코드를 작성하면 된다. 모델에 입력값을 전달하기 위해 MNIST 데이터를 28 * 28 데이터로 구성하는 부분이 조금 다르다.

    batch_xs.reshape(-1,28,28,1)
    mnist.test.images.reshape(-1,28,28,1)
    
이것을 적용한 학습 및 결과 확인 코드는 다음과 같다.

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    
    batch_size = 100
    total_batch = int(mnist.train.num_examples / batch_size)
    
    for epoch in range(15):
      total_cost = 0
      
      for i in range(total_batch):
        batch_xs , batch_ys = mnist.train.next_batch(batch_size)
        batch_xs = batch_xs.reshape(-1,28,28,1)
        
        _, cost_val = sess.run([optimizer,cost], feed_dict = {X:batch_xs, Y:batch_ys, keep_prob: 0.7})
        
        total_cost +=cost_val
      print('Epoch:', '%04d' %(epoch +1), 'Avg. cost = ', '{:.3f}'.format(total_cost / total_batch))
      
    print('최적화 완료!')
    
    is_correct = tf.equal(tf.argmax(model,1), tf.argmax(Y,1))
    accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
    print('정확도:', sess.run(accuracy, feed_dict = {X:mnist.test.images.reshape(-1,28,28,1), Y:mnist.test.labels, keep_prob:1}))

## 7.3 고수준 API 















