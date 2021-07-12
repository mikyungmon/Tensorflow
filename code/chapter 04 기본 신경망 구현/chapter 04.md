# Chapter 04

이번 장에서는 심층 신경망, 즉 다층 신경망을 간단하게 구현해본다.

## 4.1 인공신경망의 작동 원리

**인공신경망**의 개념은 뇌를 구성하는 신경 세포, 즉 뉴런의 동작 원리에 기초한다. 

**뉴런**의 기본 원리는 매우 간단하다.

가지돌기에서 신호를 받아들이고, 이 신호가 축삭돌기를 지나 축살말단으로 전달되는 것이다. 

그런데 축삭돌기를 지나는 동안 신호가 약해지거나 너무 약해서 축삭말단까지 전달되지 않거나 또는 강하게 전달되기도 한다.

그리고 축삭말단까지 전달된 신호는 연결된 다음 뉴런의 가지돌기로 전달된다.

이러한 뉴런과 신경망의 원리에 인공 뉴런의 개념의 덧씌우면 다음 그림과 같이 표현할 수 있다.

![image](https://user-images.githubusercontent.com/66320010/125259543-5e9eb600-e33a-11eb-849f-55af01a4c089.png)

그림과 같이 입력 신호, 즉 입력값 X에 가중치(W)를 곱하고 편향(b)을 더한 뒤 활성화 함수(Sigmoid, ReLU 등)를 거쳐 결과값 y를 만들어내는 것이 바로 인공 뉴런의 기본이다.

원하는 y값을 만들어내기 위해 W와 b의 값을 변경해가면서 적절한 값을 찾아내는 최적화 과정을 **학습** 또는 **훈련**이라고 한다.

**활성화 함수**는 인공신경망을 통과해온 값을 최종적으로 어떤 값으로 만들지 결정한다. 즉, 이 함수가 인공 뉴런의 핵심중에서도 가장 중요한 요소이다.

활성화 함수에는 대표적으로 Sigmoid, ReLU, tanh함수가 있으며 다음 그림과 같은 모양이다.

![image](https://user-images.githubusercontent.com/66320010/125276732-84808680-e34b-11eb-87ce-ec6a28a46a36.png)
                                
최근 활성화 함수로 ReLU함수를 많이 사용하는데 ReLU는 입력값이 0보다 작으면 항상 0을, 0보다 크면 입력값을 그대로 출력한다.

✔ 다시 정리하자면, 인공 뉴런은 가중치와 활성화 함수의 연결로 이루어진 매우 간단한 구조이다. 이렇게 간단한 개념의 인공 뉴런을 충분히 많이 연결해놓는 것만으로 인간이 인지하기 어려운 매우 복잡한 패턴까지도 스스로 학습할 수 있게 된다는 사실이 매우 놀랍다.

그러나 수천~수만 개의 W와 b값의 조합을 일일이 변경해가며 계산하려면 매우 오랜시간이 걸리기 때문에 신경망을 제대로 훈련시키기가 어려웠다. 

특히 신경망의 층이 깊어질수록 시도해봐야 하는 조합의 경우가 기하급수적으로 늘어나기 때문에 과거에는 유의미한 신경망을 만드는 것은 거의 불가능하다고 여겨졌다.

💡 그러나 제프리 힌튼 교수가 제한된 볼트만 머신이라는 신경망 학습 알고리즘을 개발하였고, 이 방법으로 심층 신경망을 효율적으로 학습시킬 수 있음을 증명하면서 다시 한 번 신경망이 주목받게 되었다.

그 후 드롭아웃 기법, ReLU 등의 활성화 함수들이 개발되면서 딥러닝 알고리즘은 급속한 발전을 이루었다.

이렇게 발전해온 알고리즘의 중심에는 **역전파(backpropagation)** 가 있다. 

**역전파**는 간단히 말해, 출력층이 내놓은 결과의 오차를 신경망을 따라 입력층까지 역으로 전파하며 계산해나가는 방식이다.

이 방식은 입력층부터 가중치를 조절해가는 기존 방식보다 훨씬 유의미한 방식으로 가중치를 조절해줘서 최적화 과정이 훨씬 빠르고 정확해진다.

역전파는 신경망을 구현하려면 거의 항상 적용해야하는 알고리즘이지만 구현하기는 조금 어렵다.

텐서플로는 활성화 함수와 학습 함수 대부분에 역전파 기법을 기본으로 제공한다. 

텐서플로를 사용하면 다양한 학습 알고리즘을 직접 구현하지 않고도 매우 쉽게 신경망을 만들고 학습할 수 있다.

## 4.2 간단한 분류 모델 구현하기

딥러닝은 매우 다양한 분야에 사용되지만 그 중 가장 폭넓게 활용되는 분야는 패턴 인식을 통한 영상 처리이다. 예를 들어 사진이 고양이인지, 강아지인지 또는 자동차인지, 비행기인지 등을 판단하는 일에 쓰인다.

이처럼 패턴을 파악해 여러 종류로 구분하는 작업을 **분류**라고 한다.

이번 예에서는 털과 날개가 있느냐를 기준으로 포유류와 조류를 구분하는 신경망 모델을 만들어본다. 

이미지 대신 간단한 이진 데이터를 이용한다.

1️⃣ 텐서플로와 Numpy 라이브러리를 임포트한다. Numpy는 수치해석용 파이썬 라이브러리이다. 행렬 조작과 연산에 필수라 할 수 있고 텐서플로도 Numpy를 매우 긴밀하게 이용하고 있다.

    import tensorflow as tf
    import numpy as np
    
2️⃣ 다음은 학습에 사용할 데이터를 정의한다. 털과 날개가 있느냐를 담은 특징 데이터를 구성한다. 있으면 1, 없으면 0이다.

    x_data = np.array([[0,0],[1,0],[0,0][0,0],[0,1]])   # [털, 날개]
    
  그다음은 각 개체가 실제 어떤 종류인지를 나타내는 레이블 데이터를 구성한다. 즉 앞서 정의한 특징 데이터의 각 개체가 포유류인지 조류인지, 아니면 제 3의 종류인지를 기록한 실제 결과값이다.
  
  레이블 데이터는 원-핫 인코딩이라는 특수한 형태로 구성한다. 원-핫 인코딩이란 데이터가 가질 수 있는 값들을 일렬로 나열한 배열을 만들고 그 중 표현하려는 값을 뜻하는 인덱스의 원소만 1로 표기하고 나머지 원소는 모두 0으로 채우는 표기법이다.
  
  우리가 판별하고자 하는 개체의 종류를 원-핫 인코딩으로 나타내면 다음과 같다.
  
    기타 = [1,0,0]
    포유류 = [0,1,0]
    조류 = [0,0,1]

  이를 특징 데이터와 연관 지어 레이블 데이터로 구성하면 다음처럼 만들 수 있다.
  
    y_data = np.array([[1,0,0], [0,1,0], [0,0,1], [1,0,0], [1,0,0], [0,0,1]])
    
3️⃣ 신경망 모델을 구성해보자. 특징 X와 레이블 Y와의 관계를 알아내는 모델이다. 이때 X와 Y에 실측값(ground truth)를 넣어서 학습시킬 것이니까 X와 Y는 플레이스홀더로 설정한다.

    X = tf.placeholder(tf.float32)
    Y = tf.placeholder(tf.float32)
  
  그다음은 앞서 배운 신경망을 결정하는 가중치와 편향 변수를 설정한다. 이 변수들의 값을 여러가지로 바꿔가면서 X와 Y의 연관 관계를 학습하게 된다.
  
    W = tf.Variable(tf.random_uniform([2,3], -1.,1.))   # [특징 수(입력층), 레이블 수(출력층)]의 구성인 [2,3]으로 설정
    b = tf.Variable(tf.zeros([3]))   # 레이블 수인 3개의 요소를 가진 변수로 설정

  가중치를 곱하고 편향을 더한 결과를 활성화 함수인 ReLU에 적용하면 신경망 구성은 끝이다.
  
    L = tf.add(tf.matmul(X,W),b)
    L = tf.nn.relu(L)

4️⃣ 추가로 신경망을 통해 나온 출력값을 softmax함수를 이용하여 사용하기 쉽게 다듬어준다.

    model = tf.nn.softmax(L)  # 각각은 해당 결과의 확률로 해석
    
5️⃣ 이제 손실함수를 작성한다. 손실 함수는 원-핫 인코딩을 이용한 대부분의 모델에서 사용하는 **교차 엔트로피**함수를 사용하도록 한다. 교차 엔트로피 값은 예측값과 실제값 사이의 확률 분포 차이를 계산한 값으로 기본 코드는 다음과 같다.

    cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(model),axis = 1))   # reduce_xxx 함수들은 텐서의 차원을 줄임

 코드가 복잡해 보이지만 계산된 값을 보면 어렵지 않게 이해할 수 있다. 계산 과정을 천천히 따라가보자. 먼저 Y는 실측값이다. 그리고 model은 신경망을 통해 나온 예측값이다.
 
        Y              model
    [[1 0 0 ]     [[0.1 0.7 0.2]
     [0 1 0]]      [0.2 0.8 0.0]]
 
 그런 다음 model값에 log를 취한 값을 Y랑 곱하면 다음과 같이 된다.

        Y              model           Y * tf.log(model)
    [[1 0 0 ]     [[0.1 0.7 0.2]   ->   [[-1.0 0  0]
     [0 1 0]]      [0.2 0.8 0.0]]       [0 -0.09 0]]
     
 이제 행별로 값을 다 더한다.

    Y * tf.log(model)        reduce_sum(axis = 1)
     [[-1.0 0  0]       ->     [ -1.0  -0.09]
     [0 -0.09 0]]

 마지막으로 배열 안 값의 평균을 내면 그것이 바로 우리의 손실값인 교차 엔트로피 값이 된다.
 
    reduce_sum(axis = 1)      reduce_mean
      [ -1.0  -0.09]      ->     -0.545
   
6️⃣ 이제 학습을 시켜보자. 학습은 텐서플로가 기본 제공하는 최적화 함수를 사용하면 된다.

    # 경사하강법으로 최적화
    optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01)
    train_op = optimizer.minimize(cost)
    
    # 텐서플로의 세션을 초기화
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    
    # 특징와 레이블 데이터를 이용해 100번 학습
    for step in range(100):
      sess.run(train_op, feed_dict({X:x_data, Y: y_data}))
      
      # 학습 도중 10번에 한 번 씩 손실값 출력
      if (step+1) % 10 == 0:
        print(step +1, sess.run(cost,feed_dict = {X:x_data, Y: y_data}))

7️⃣ 학습된 결과를 확인하는 코드를 작성한다.

    prediction = tf.argmax(model, axis = 1)
    target = tf.argmax(Y, axis = 1)
    print('예측값:', sess.run(prediction, feed_dict = {X:x_data}))
    print('실제값:', sess.run(target, feed_dict = {Y:y_data}))
    
  - 예측값인 model을 바로 출력하면 [0.2 0.7 0.1]과 같이 확률로 나오기 때문에 요소 중 가장 큰 인덱스를 찾아주는 argmax함수를 사용하여 레이블 값을 출력하도록 했다. 즉 다음처럼 원-핫 인코딩을 거꾸로 한 결과를 만들어준다.

        [[0 1 0] [1 0 0]]  -> [1 0]
        [[0.2 0.7 0.1] [0.9 0.1 0.]] -> [1 0]

8️⃣ 정확도를 출력해본다.

    is_correct = tf.equal(prediction, target)
    accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))  # true / false로 나온 결과를 다시 tf.cast함수를 이용해 0과 1로 바꾸어 평균을 내면 간단히 정확도 구할 수 있음
    print('정확도: %.2f' % sess.run(accuracy * 100, feed_dict={X:x_data,Y:y_data}))

📍 전체 코드 📍

    import tensorflow as tf
    import numpy as np
    
    x_data = np.array([[0,0], [1,0], [1,1], [0,0], [0,0], [0,1]])
    y_data = np.array([[1,0,0], [0,1,0], [0,0,1], [1,0,0], [1,0,0], [0,0,1]])
    
    X = tf.placeholder(tf.float32)
    Y = tf.placeholder(tf.float32)
    
    W = tf.Variable(tf.random_uniform([2,3], -1. , 1.))
    b = tf.Variable(tf.zeros([3]))
    
    L = tf.add(tf.matmul(X,W),b)
    L = tf.nn.relu(L)
    
    model = tf.nn.softmax(L)
    cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(model), axis =1))
    
    optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01)
    train_op = optimizer.minimize(cost)
    
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    
    for step in range(100):
      sess.run(train_op, feed_dict = {X:x_data, Y:y_data})
      
      if (step+1) % 10 == 0:
        print(step +1, sess.run(cost, feed_dict = {X:x_data, Y:y_data}))
        
    prediction = tf.argmax(model, axis =1)
    target = tf.argmax(Y, axis =1)
    print('예측값: ', sess.run(prediction, feed_dict = {X:x_data}))
    print('실제값: ', sess.run(target, feed_dict = {Y:y_data}))
    
    is_correct = tf.equal(prediction, target)
    accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
    print('정확도: %.2f' % sess.run(accuracy *100, feed_dict={X:x_data, Y:y_data}))
    
## 4.3 심층 신경망 구현하기

이제 신경망의 층을 둘 이상으로 구성한 심층 신경망, 즉 딥러닝을 구현해본다.

1️⃣ 다층 신경망을 만드는 것은 매우 간단하다. 앞서 만든 신경망 모델에 가중치와 편향을 추가하기만 하면 된다.

    W1 = tf.Variable(tf.random_uniform([2,10], -1. , 1.))
    W2 = tf.Variable(tf.random_uniform([10,3], -1. , 1.))
    
    b1 = tf.Variable(tf.zeros([10]))
    b2 = tf.Variable(tf.zeros([3]))
    
  코드를 보면 첫 번째 가중치 형태는 [2,10]으로, 두 번째 가중치는 [10,3]으로 설정했고 편향을 각각 10과 3으로 설정했다. 그 의미는 다음과 같다.
  
    # 가중치
    W1 = [2,10]  -> [특징, 은닉층의 뉴런 수]
    W2 = [10,3]  -> [은닉 층이 뉴런 수, 분류 수]
    
    # 편향
    b1 = [10]   -> 은닉층의 뉴런 수
    b2 = [3]   -> 분류 수
    
  입력층과 출력층은 각각 특징과 분류 개수로 맞추고 중간의 연결 부부은 맞닿은 층의 뉴런 수와 같도록 맞추면 된다. 중간의 연결 부분을 **은닉층**이라고 하며 은닉층의 뉴런 수는 하이퍼파라미터이니 실험을 통해 가장 적절한 수를 정하면 된다.
  
2️⃣ 특징 입력값에 첫 번째 가중치와 편향, 그리고 활성화 함수를 적용한다.

    L1 = tf.add(tf.matmul(X,W1), b1)
    L1 = tf.nn.relu(L1)
    
3️⃣ 출력층을 만들기 위해 두 번째 가중치와 편향을 적용하여 최종 모델을 만든다. 은닉층에 두 번째 가중치 W2와 편향 b2를 적용하면 최종적으로 3개의 출력값을 가지게 된다.

    model = tf.add(tf.matmul(L1,W2), b2)
    
4️⃣ 손실 함수를 작성한다. 손실 함수는 교차 엔트로피 함수를 사용한다. 다만 이번에는 텐서플로가 기본 제공하는 교차 엔트로피 함수를 이용한다, 

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = Y, logits = model))
    
    optimizer = tf.train.AdamOptimizer(learning_rate = 0.01)   # 경사하강법보다 보편적으로 성능이 좋음
    train_op = optimizer.minimize(cost)
  
5️⃣ 학습 진행, 손실값과 정확도 측정 등 앞절에서 본 나머지 코드를 넣고 실행하면 정확한 예측값을 얻게 될 것이다.

📍 전체 코드 📍

    import tensorflow as tf
    import numpy as np
    
    x_data = np.array([[0,0], [1,0], [1,1], [0,0], [0,0], [0,1]])
    y_data = np.array([1,0,0], [0,1,0], [0,0,1], [1,0,0], [1,0,0], [0,0,1]])
  
    X = tf.placeholder(tf.float32)
    Y = tf.placeholder(tf.float32)  
  
    W1 = tf.Variable(tf.random_uniform([2,10], -1. , 1.))
    W2 = tf.Variable(tf.random_uniform([10,3], -1. , 1.))
    
    b1 = tf.Variable(tf.zeros([10]))
    b2 = tf.Variable(tf.zeros([3])) 
  
    L1 = tf.add(tf.matmul(X,W1), b1)
    L1 = tf.nn.relu(L1)  
    
    model = tf.add(tf.matmul(L1,W2), b2)
    
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = Y, logits = model))
    
    optimizer = tf.train.AdamOptimizer(learning_rate = 0.01)   # 경사하강법보다 보편적으로 성능이 좋음
    train_op = optimizer.minimize(cost)   
    
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    
    for step in range(100):
      sess.run(train_op, feed_dict = {X:x_data, Y:y_data})
      
      if (step+1) % 10 == 0::
        print(step +1, sess.run(cost, feed_dict = {X:x_data, Y:y_data}))
        
    prediction = tf.argmax(model, axis =1)
    target = tf.argmax(Y, axis =1)
    print('예측값: ', sess.run(prediction, feed_dict = {X:x_data}))
    print('실제값: ', sess.run(target, feed_dict = {Y:y_data}))
    
    is_correct = tf.equal(prediction, target)
    accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
    print('정확도: %.2f' % sess.run(accuracy *100, feed_dict={X:x_data, Y:y_data}))    
