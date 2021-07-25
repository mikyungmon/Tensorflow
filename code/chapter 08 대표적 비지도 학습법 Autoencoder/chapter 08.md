# Chapter 08

머신러닝 학습 방법은 크게 **지도 학습**과 **비지도 학습**으로 나눌 수 있다.

  - 지도 학습 : 프로그램에게 원하는 결과를 알려주고 학습하게 하는 방법
  - 비지도 학습 : 입력값으로부터 데이터의 특징을 찾아내는 학습 방법

✔ 간단하게 말해 지도 학습은 X와 Y가 둘 다 있는 상태에서, 비지도 학습은 X만 있는 상태에서 학습하는 것이다.

이러한 비지도 학습 중 가장 널리 쓰이는 신경망으로 **오토인코더**가 있다.

## 8.1 오토인코더 개념

오토인코더는 입력값과 출력값을 같게 하는 신경망이며 가운데 계층의 노드 수가 입력값보다 적은 것이 독특한 점이다.

이런 구조로 인해 입력 데이터를 압축하는 효과를 얻게 되고, 이 과정이 노이즈 제거에 매우 효과적이라고 알려져있다.

💡 오토인코더의 핵심은 입력층으로 들어온 데이터를 인코더를 통해 은닉층으로 내보내고 은닉층의 데이터를 디코더를 통해 출력층으로 내보낸 뒤 만들어진 출력값을 입력값과 비슷해지도록 만드는 가중치를 찾아내는 것이다.

## 8.2 오토인코더 구현하기

1️⃣ 텐서플로와 numpy, 그래프 출력 위한 matplotlib, 그리고 MNIST 모듈을 임포트하고 학습시킬 데이터를 준비한다.

    import tensorflow as tf
    import numpy as np
    import matplotlib.pyplot as plt
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("./mnist/data/", one_hot = True)
    
2️⃣ 하이퍼파라미터로 사용할 옵션들을 따로 빼내어 코드를 구조화한다.

    learning_rate = 0.01   # 최적화 함수에서 사용할 학습률
    training_epoch = 20   # 전체 데이터를 학습할 총 횟수
    batch_size = 100    # 미니 배치로 한 번에 학습할 데이터
    n_hidden = 256    # 은닉층의 뉴런 개수
    n_input = 28 * 28    
    
 3️⃣ 신경망 모델 구성한다. 이 모델을 비지도 학습이므로 Y값이 없다.
 
    X = tf.placeholder(tf.float32, [None, n_input])
    
 4️⃣ 오토인코더의 핵심은 인코더와 디코더를 만드는 것이다. 인코더와 디코더를 만드는 방식에 따라 다양한 오토인코더를 만들 수 있다. 우선 인코더를 만들어본다.
 
    W_encode = tf.Variable(tf.random_normal([n_input,n_hidden]))
    b_encode = tf.Variable(tf.random_normal([n_hidden]))
    encoder = tf.nn.sigmoid(tf.add(tf.matmul(X,W_encode), b_encode))
  
  - 이렇게 하면 입력값을 압축하고 노이즈를 제거하면서 입력값의 특징을 찾아내게 된다.

5️⃣ 다음은 디코더를 만든다.

    W_decode = tf.Variable(tf.random_normal([n_hidden,n_input])
    b_decode = tf.Variable(tf.random_normal([n_input])
    decoder = tf.nn.sigmoid(tf.add(tf.matmul(encoder, W_decode), b_decode))

6️⃣ 그런 다음 가중치를 최적화하기 위한 손실 함수 만든다.

    cost = tf.reduce_mean(tf.pow(X -decoder),2)

  - 기본적인 오토인코더의 목적은 출력값과 입력값을 가장 비슷하게 만드는 것이다. 그렇게하면 압축된 은닉층의 뉴런들을 통해 입력값의 특징을 알아낼 수 있다.
  
  - 따라서 입력값인 X를 평가하기 위한 실측값으로 사용하고 디코더가 내보낸 결과값과의 차이를 손실값으로 설정한다.
  
  - 그리고 이 값의 차이는 거리 함수로 구하도록 한다.


7️⃣ 마지막으로 RMSPropOptimizer함수를 이용한 최적화 함수를 설정한다.

    optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)
    
8️⃣ 그리고 학습을 진행하는 코드를 작성한다.

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    
    total_batch = int(mnist.train.num_examples / batch_size)
    
    for epoch in range(training_epoch):
      total_cost = 0
      
      for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        _, cost_val = sess.run([optimizer,cost], feed_dict = {X:batch_xs})
        
        total_cost +=cost_val
        
      print('Epoch:', '%04d' % (epoch+1), 'Avg. cost = ', '{:.4f}'.format(total_cost / total_batch))
      
    print('최적화 완료!')
    
9️⃣ 이번에는 결과값을 정확도가 아닌 디코더로 생성해낸 결과를 직관적인 방법으로 확인해보자. 여기서는 matplotlib을 이용해 이미지를 출력해본다.

먼저 총 10개의 테스트 데이터를 가져와 디코더를 이용해 출력값으로 만든다.

    sample_size = 10
    samples = sess.run(decoder, feed_dict = {X:mnist.test.images[:sample_size]})
    fig, ax = plt.subplots(2, sample_size, figsize = (sample_size,2))
    
    for i in range(sample_size):
      ax[0][i].set_axis_off()
      ax[1][0].set_axis_off()
      ax[0][i].imshow(np.reshape(mnist.test.images[i],(28,28)))    # 입력값의 이미지 출력
      ax[1][i].imshow(np.reshape(samples[i], (28,28)))     # 신경망으로 생성한 이미지 출력



