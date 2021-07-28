# Chapter 10

이미지 인식에 CNN이 있다면 자연어 인식에는 **순환 신경망**이라고 하는 RNN(Recurrent Neural Network)가 있다.

RNN은 상태가 고정된 데이터를 처리하는 다른 신경망과는 달리 자연어 처리나 음성 인식처럼 **순서가 있는 데이터**를 처리하는 데 강점을 가진 신경망이다.

앞이나 뒤의 정보에 따라 전체의 의미가 달라지거나 앞의 정보로 다음에 나올 정보를 추측하려는 경우에 RNN을 사용하면 성능 좋은 프로그램을 만들 수 있다.

이번 장에서는 RNN의 기본적인 사용법을 배우고 마지막에는 Sequence to Sequence모델을 이용해 간단한 번역 프로그램을 만든다.

## 10.1 MNIST를 RNN으로

앞에서 사용해온 손글씨 이미지를 RNN방식으로 학습하고 예측하는 모델을 만들어보자.

기본적인 RNN 개념은 다음 그림과 같다.

(사진)

- 이 그림의 가운데에 있는 한 덩어리의 신경망을 RNN에서는 **셀**이라고 하며 RNN은 이 셀을 여러개 중첩하여 심층 신경망을 만든다.

- 간단하게 말해 앞 단계에서 학습한 결과를 다음 단계의 학습에도 이용하는 것인데, 이런 구조로 인해 학습 데이터를 단계별로 구분하여 입력해야 한다. 따라서 MNIST의 입력값도 단계별로 입력할 수 있는 형태로 변경해줘야한다.

(사진)

- 사람은 글씨를 위에서 아래로 내려가면서 쓰는 경향이 많으니 데이터를 위 그림처럼 구성한다.

- MNIST가 가로,세로 28 * 28크기이니 가로 한 줄의 28픽셀을 한 단계의 입력값으로 삼고 세로줄이 총 28개이므로 28단계를 거쳐 데이터를 입력받는 개념이다.

1️⃣ 코드를 작성해본다. 다음은 학습에 사용할 하이퍼파라미터들과 변수, 출력층을 위한 가중치와 편향을 정의한 부분이다.

    import tensorflow as tf
    
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("./mnist/data/", one_hot = True)
    
    learning_rate = 0.001
    total_epoch = 30
    batch_size = 128
    
    n_input = 28
    n_step = 28
    n_hidden = 128
    n_class = 10
    
    X = tf.placeholder(tf.float32, [None, n_step, n_input])
    Y = tf.placeholder(tf.float32, [None, n_class])
    
    W = tf.Variable(tf.random_normal([n_hidden,n_class]))
    b = tf.Variable(tf.random_normal([n_class]))
    
 - 기존 모델과 다른 점은 입력값 X에 n_step이라는 차원을 하나 추가한 부분이다. RNN은 순서가 있는 데이터를 다루므로 한 번에 입력 받을 개수와 총 몇 단계로 이뤄진 데이터를 받을지 설정해야 한다.
 
 - 출력값은 MNIST의 분류인 0~9까지 10개의 숫자를 원-핫 인코딩으로 표현하도록 만들었다.

2️⃣ 그런 다음 n_hidden개의 출력값을 갖는 RNN셀을 생성한다. RNN을 저수준부터 직접 구현하려면 다른 신경망보다 복잡한 계산을 거쳐야 하지만 텐서플로를 이용하면 다음처럼 매우 간단하게 생성할 수 있다.

    cell = tf.nn.rnn_cell.BasicRNNCell(n_hidden)
    
  - RNN 기본 신경망은 긴 단계의 데이터를 학습할 때 맨 뒤에서는 맨 앞의 정보를 잘 기억하지 못하는 특성이 있다. 이를 보완하기 위해 다양한 구조가 만들어졌고 그 중 많이 사용되는 것이 **LSTM**이라는 신경망이다.
  
  - GRU는 LSTM과 비슷하지만 구조가 조금 더 간단한 신경망 아키텍처이다.

3️⃣ 다음으로 dynamic_rnn함수를 이용해 RNN 신경망을 완성한다.

    outputs, states = tf.nn.dynamic_rnn(cell, X, dtype = tf.float32)
    
 - 앞서 생성한 RNN 셀과 입력값, 그리고 입력값의 자료형을 넣어주기만 하면 간단하게 신경망 생성할 수 있다.

 - 앞의 RNN 그림처럼 한 단계를 학습한 뒤 상태를 저장하고 그 상태를 담은 단계의 입력 상태로 하여 다시 학습한다. 
 
 - 이렇게 주어진 단계만큼 반복하여 상태를 전파하면서 출력값을 만들어가는 것이 RNN의 기본 구조이다.
 
 - 하지만 반복 단계에서 고려해야 할 것이 많기 때문에 우리는 이 과정을 대신 해주는 dynamic_rnn함수를 사용했다. 이 함수를 사용하면 단 두줄로 RNN 모델의 핵심 구조(셀과 신경망)를 만들 수 있다.

4️⃣ RNN에서 나온 출력값을 가지고 최종 출력값을 만들어본다.

결과값을 원-핫 인코딩 형태로 만들 것이기 때문에 손실 함수로 tf.nn.softmax_cross_entropy_with_logits을 사용한다. 

이 함수를 사용하려면 최종 결과값이 실측값 Y와 동일한 형태인 [batch_size, n_class]여야 한다. 

앞에서 이 형태의 출력값을 만들기 위해서 가중치와 편향을 다음과 같이 설정했다.

    W = tf.Variable(tf.random_normal([n_hidden,n_class]))
    b = tf.Variable(tf.random_normal([n_class]))
    
그런데 RNN 신경망에서 나오는 출력값은 각 단계가 포함된 **[batch_size, n_step, n_hidden] 형태**이다.

따라서 은닉층의 출력값을 가중치 W와 같은 형태로 만들어줘야 행렬곱을 수행하여 원하는 출력값을 얻을 수 있다.

(참고로 dynamic_rnn 함수의 옵션 중 time_major의 값을 True로 하면 [batch_size, n_step, n_hidden] 형태로 출력된다.)

    # outputs : [batch_size, n_step, n_hidden] -> [ n_step, batch_size, n_hidden]
    outputs = tf.transpose(outputs,[1,0,2])   # n_step과 batch_size 차원의 순서를 바꿈
    # outputs : [batch_size, n_hidden]
    outputs = outputs[-1]  # n_step 차원 제거 -> 마지막 단계의 결과값만 취함
    
5️⃣ 이제 인공신경망의 기본 수식이자 핵심인 y = X * W + b를 이용하여 최종 결과값을 만든다.

    model = tf.matmul(outputs,W) + b
    
6️⃣ 지금까지 만든 모델과 실측값을 비교하여 손실값을 구하고 신경망을 최적화하는 함수를 사용하여 신경망 구성을 마무리한다.

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = model, label = Y))
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

7️⃣ 앞서 구성한 신경망을 학습시키고 결과를 확인하는 코드를 작성한다. 앞 장의 코드와 거의 같지만 입력값이 [batch_size, n_step,n_input]형태이므로 CNN에서 사용한 것처럼 reshape함수를 이용해 데이터 형태를 바꿔주는 부분만 주의하면 된다.

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    total_batch = int(mnist.train.num_exmples / batch_size)
    
    for epoch in range(total_epoch):
        total_cost = 0
        
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            batch_xs = batch_xs.reshape((batch_size,n_step, n_input))
        
            _, cost_val = sess.run([optimizer,cost], feed_dict = {X:batch_xs, Y: batch_ys})
        
            total_cost += cost_val
        
        print('Epoch:', '%04d' %(epoch+1), 'Avg. cost = ', '{:.3f}'.format(total_cost / total_batch))
      
    print('최적화 완료!')
    
    is_correct = tf.equal(tf.argmax(model,1), tf.argmax(Y,1))
    accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
    
    test_batch_size = len(mnist.test.images)
    test_xs = mnist.test.images.reshape(test_batch_size, n_step, n_input)
    test_ys = mnist.test.labels
    
    print('정확도:', sess.run(accuracy, feed_dict = {X:test_xs, Y: test_ys}))
    
















