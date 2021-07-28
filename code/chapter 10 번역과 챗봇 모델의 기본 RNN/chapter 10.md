# Chapter 10

이미지 인식에 CNN이 있다면 자연어 인식에는 **순환 신경망**이라고 하는 RNN(Recurrent Neural Network)가 있다.

RNN은 상태가 고정된 데이터를 처리하는 다른 신경망과는 달리 자연어 처리나 음성 인식처럼 **순서가 있는 데이터**를 처리하는 데 강점을 가진 신경망이다.

앞이나 뒤의 정보에 따라 전체의 의미가 달라지거나 앞의 정보로 다음에 나올 정보를 추측하려는 경우에 RNN을 사용하면 성능 좋은 프로그램을 만들 수 있다.

이번 장에서는 RNN의 기본적인 사용법을 배우고 마지막에는 Sequence to Sequence모델을 이용해 간단한 번역 프로그램을 만든다.

## 10.1 MNIST를 RNN으로

앞에서 사용해온 손글씨 이미지를 RNN방식으로 학습하고 예측하는 모델을 만들어보자.

기본적인 RNN 개념은 다음 그림과 같다.

![image](https://user-images.githubusercontent.com/66320010/127316303-679ead38-17d1-454a-af2a-31e4bfe0005f.png)

- 이 그림의 가운데에 있는 한 덩어리의 신경망을 RNN에서는 **셀**이라고 하며 RNN은 이 셀을 여러개 중첩하여 심층 신경망을 만든다.

- 간단하게 말해 앞 단계에서 학습한 결과를 다음 단계의 학습에도 이용하는 것인데, 이런 구조로 인해 학습 데이터를 단계별로 구분하여 입력해야 한다. 따라서 MNIST의 입력값도 단계별로 입력할 수 있는 형태로 변경해줘야한다.

![image](https://user-images.githubusercontent.com/66320010/127316346-157abb34-67aa-4bb4-8bea-db5d8cb38748.png)

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
    
## 10.2 단어 자동 완성

이번에는 RNN 모델을 이용하여 단어를 자동 완성하는 프로그램을 만들어본다.

영문자 4개로 구성된 단어를 학습시켜 3글자만 주어지면 나머지 한 글자를 추천하여 단어를 완성하는 프로그램이다.

참고로 dynamic_rnn의 sequence_length옵션을 사용하면 가변 길이 단어를 학습시킬 수 있다. 

짧은 단어는 가장 긴 단어의 길이 만큼 뒷부분을 0으로 채우고 해당 단어 길이르 계산해 sequence_length로 넘겨주면 된다.

하지만 코드가 복잡해지므로 **여기서는 고정 길이 단어를 사용한다.**

💡 학습시킬 데이터는 영문자로 구성된 임의의 단어를 사용할 것이고 한 글자 한글자를 하나의 단계로 볼 것이다.

그러면 한 글자가 한 단계의 입력값이 되고 총 글자 수가 전체 단계가 된다.

![image](https://user-images.githubusercontent.com/66320010/127316378-2cb4e26f-e491-4423-bab7-6837b436c71a.png)

1️⃣ 입력으로는 알파벳 순서에서 각 글자에 해당하는 인덱스를 원-핫 인코딩으로 표현한 값을 취한다. 이를 위해 알파벳 글자들을 배열에 넣고 해당 글자의 인덱스를 구할 수 있는 연관 배열(딕셔너리)도 만들어둔다.

    import tensorflow as tf
    import numpy as np
    
    char_arr = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    num_dic = {n:i for i, n in enumerate(char_arr)}   # {'a':0 , 'b':1 , ~~ }
    dic_len = len(num_dic)
    
2️⃣ 그리고 학습에 사용할 단어를 배열로 저장한다.

    seq_data = ['word', 'wood', 'deep', 'dive', 'cold', 'cool', 'load', 'love', 'kiss', 'kind']
    
3️⃣ 단어들을 학습에 사용할 수 있는 형식으로 변환해주는 유틸리티 함수를 작성한다. 이 함수는 다음 순서로 데이터를 반환한다.

1. 입력값용으로 단어의 처음 세 글자의 알파벳 인덱스를 구한 배열을 만들어준다.

    - input = [num_dic[n] for n in seq[:-1]]

2. 출력값용으로 마지막 글자의 알파벳 인덱스를 구한다.

    - target = num_dic[seq[-1]]

3. 입력값을 원-핫 인코딩으로 변환한다.

    - input_batch.append(np.eye(dic_len)[input])

예를 들어 "deep"는 입력으로 d,e,e를 취하고 각 알파벳의 인덱스를 구해 배열로 만들면 [3,4,4]가 나온다.

그리고 이를 원-핫 인코딩하면 최종 입력값은 다음과 같이 된다.

    [[0.  0.  0.  1.  0.  0.  0.  ~~ 0.]
     [0.  0.  0.  0.  1.  0.  0.  ~~ 0.]   
     [0.  0.  0.  0.  1.  0.  0.  ~~ 0.]]
   
실측값은 p의 인덱스인 15가 되는데 실측값은 원-핫 인코딩하지 않고 15를 그대로 사용할 것이다. 

그 이유는 손실 함수로 지금까지 사용하던 softmax_cross_entropy_with_logits가 아닌 sparse_softmax_cross_entropy_with_logits를 사용할 것이기 때문이다.

sparse_softmax_cross_entropy_with_logits 함수는 실측값, 즉 labels 값에 원-핫 인코딩을 사용하지 않아도 자동으로 변환하여 계산해준다.

이렇게 변환하는 함수를 코드로 작성하면 다음과 같다.

    def make_batch(seq_data):
        input_batch = []
        target_batch = []
        
        for seq in seq_data:
            input = [num_dic[n] for n in seq[:-1]]
            target = num_dic[seq[-1]]
            input_batch.append(np.eye(dic_len)[input])  # 입력값을 원-핫 인코딩으로 변환
            target_batch.append(target)   # 원-핫 인코딩 하지않고 인덱스 번호 그대로 사용
            
        return input_batch, target_batch

4️⃣ 데이터를 전처리하는 부분을 마쳤으니 신경망 모델을 구성한다. 먼저 옵션들을 설정한다.

    learning_rate = 0.01
    n_hidden = 128
    total_epoch = 30
    
    n_step = 3  # 단어 전체 중 처음 3글자를 단계적으로 학습할 것이므로 n_step은 3이 됨
    n_input = n_class = dic_len

 - 입력값과 출력값은 알파벳의 원-핫 인코딩을 사용할 것이므로 알파벳 글자들의 배열 크기인 dic_len과 같다.
 
 - 여기서 주의할 것은 **sparse_softmax_cross_entropy_with_logits 함수를 사용하더라도 비교를 위한 예측 모델의 출력값은 원-핫 인코딩을 사용해야 한다.** 
 
 - 그래서 n_class값도 n_input값과 마찬가지로 dic_len과 크기가 같도록 설정했다.

❗ 즉 sparse_softmax_cross_entropy_with_logits 함수를 사용할 때 **실측값인 labels값은 인덱스의 숫자를 그대로 사용하고 예측 모델의 출력값은 인덱스의 원-핫 인코딩을 사용한다.**

5️⃣ 본격적으로 신경망 모델을 구성해본다.

    X = tf.placeholder(tf.float32, [None, n_step, n_input])
    Y = tf.placeholder(tf.int32, [None])  # 원-핫 인코딩이 아니라 인덱스 숫자를 그대로 사용하기 때문에 값이 하나뿐인 1차원 배열을 입력으로 받음. ex) [3] [3] [15] [4] ~
    
    W = tf.Variable(tf.random_normal([n_hidden, n_class]))
    b = tf.Variable(tf.random_normal([n_class])) 

6️⃣ 다음으로 두 개의 RNN 셀을 생성한다. 여러 셀을 조합해 심층 신경망을 만들기 위해서이다. DropoutWrapper함수를 사용하여 RNN에도 과적합 방지를 위한 드롭아웃 기법을 쉽게 적용할 수 있다.

    cell1 = tf.nn.rnn_cell.BasicLSTMCell(n_hidden)
    cell1 = tf.nn.rnn_cell.DropoutWrapper(cell1, output_keep_prob = 0.5)   # 드롭아웃
    cell2 = tf.nn.rnn_cell.BasicLSTMCell(n_hidden)
    
7️⃣ 앞서 만든 셀들을 MultiRNNCell 함수를 사용하여 조합하고 dynamic_rnn함수를 사용하여 심층 순환 신경망, 즉 DeepRNN을 만든다.

    multi_cell = tf.nn.run_cell.MultiRNNCell([cell1,cell2])
    outputs, states = tf.nn.dynamic_rnn(multi_cell, X, dtype = tf.float32)
    
8️⃣ 이전 예제인 MNIST예측 모델과 같은 방식으로 최종 출력층을 만든다.

    outputs = tf.transpose(outputs,[1,0,2])
    outputs = outputs[-1]
    model = tf.matmul(outputs,W) + b
    
9️⃣ 마지막으로 손실 함수로는 sparse_softmax_cross_entropy_with_logits를, 최적화 함수로는 AdamOptimizer를 쓰도록 설정하여 신경망 모델 구성을 마무리한다.

    cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = model, labels =Y))
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    
🔟 구성한 신경망을 학습시킨다. make_batch함수를 사용하여 seq_data에 저장한 단어들을 입력값(처음 세 글자)과 실측값(마지막 한 글자)으로 분리하고 이 값들을 최적화 함수를 실행하는 코드에 넣어 신경망을 학습시킨다.

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    input_batch, target_batch = make_batch(seq_data)
    
    for epoch in range(total_epoch):
        _, loss = sess.run([optimizer,cost], feed_dict = {X:input_atch, Y: target_batch})
        print('Epoch:', '%04d' % (epoch+1), 'cost=', '{:.6f}.format(loss))
        
    print('최적화 완료!')
    
1️⃣1️⃣ 결과값으로 예측한 단어를 정확도와 함께 출력한다.    

    prediction = tf.cast(tf.argmax(model,1), tf.int32))
    prediction_check = tf.equal(prediction,Y)  # 실측값은 원-핫 인코딩이 아닌 인덱스를 그대로 사용하므로 Y는 정수 -> 따라서 argmax로 변환한 예측값도 정수로 변환해줌
    accuracy = tf.reduce_mean(tf.cast(prediction_check, tf.float32))
    
1️⃣2️⃣ 학습에 사용한 단어들을 넣고 예측 모델을 돌린다.

    input_batch, target_batch = make_batch(seq_data)
    predict, accuracy_val = sess.run([prediction, accuracy], feed_dict = {X:input_batch, Y:target_batch})
    
1️⃣3️⃣ 마지막으로 모델이 예측한 값들을 가지고, 각각의 값에 해당하는 인덱스의 알파벳을 가져와서 예측한 단어를 출력한다.

    predict_words = []
    for idx, val in enumerate(seq_data):
        last_char = char_arr[predict[idx]]
        predict_words.append(val[:3] + last_char)
        
    print('\n=== 예측 결과 ===')
    print('입력값:', [w[:3] +' ' for w in seq_data])
    print('예측값:', predict_words)
    print('정확도:',accuracy_val)
   
 - 결과는 매우 정확하게 나온다.

## 10.3 Sequence to Sequence

