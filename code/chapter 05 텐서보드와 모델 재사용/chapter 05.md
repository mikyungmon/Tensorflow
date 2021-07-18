# Chapter 05

이번 장에서는 **학습시킨 모델을 저장하고 재사용**하는 방법과 텐서플로의 가장 큰 장점 중 하나인 **텐서보드**를 이용해 손실값의 변화를 그래프로 추적해볼 것이다.

## 5.1 학습 모델 저장하고 재사용하기

앞 장에서 사용한 포유류와 조류를 구분하는 신경망 모델을 이용할 것이다. 

다만, 코드 안에 데이터를 같이 넣는 것은 비효율적이기 때문에 데이터를 csv파일로 분리한 뒤 해당 파일을 읽어 들여 사용하는 방법을 썼다.

먼저 다음 내용을 data.csv파일로 저장한다.

    # 털, 날개, 기타, 포유류, 조류
      
       0,   0,   1,   0,   0
       1,   0,   0,   1,   0
       1,   1,   0,   0,   1
       0,   0,   1,   0,   0
       0,   0,   1,   0,   0
       0,   1,   0,   0,   1
       
   - 단, 여기서 사용하는 방법은 내용에 한글이 있으면 읽는 데 문제가 생길 수 있으니 한글 주석은 제외하고 입력하자.

   -  1열과 2열은 털과 날개, 즉 특징값이고 3열부터 마지막 열까지는 개체의 종류를 나타내는 데이터이다. 앞 장에서 본 원-핫 인코딩을 이용한 값이다.

1️⃣ 데이터 파일을 만들었으면 이제 다음처럼 데이터를 읽어 들이고 변환하는 코드로 프로그램을 시작한다.

    import tensorflow as tf
    import numpy as np
    
    data = np.loadtxt('.data/csv', delimiter = ',', unpack = True, dtype = 'float32')
    x_data = np.transpose(data[0:2])
    y_data = np.transpose(data[2:])
    
 - numpy 라이브러리 loadtxt함수를 이용하여 간단하게 데이터를 읽어 들인 뒤, 1열과 2열은 x_data로, 3열부터 마지막 열까지는 y_data로 변환하였다.

 - loadtxt의 unpack매개변수와 transpose함수는 다음 그림처럼 데이터의 구조를 변환시켜준다.

  ![image](https://user-images.githubusercontent.com/66320010/126072799-0c89742f-6b87-41c3-a23a-d6017b57906c.png)

  보는 바와 같이 읽어들이거나 잘라낸 데이터의 행과 열을 뒤바꿔주는 옵션과 함수이다.
  
2️⃣ 이제 신경망 모델을 구성할 차례이다. 먼저 모델을 저장할 때 쓸 변수를 하나 만든다. 이 변수는 학습에 직접 사용되지는 않고 학습 횟수를 카운트하는 변수이다. 이를 위해 변수 정의 시 trainable = False라는 옵션을 주었다.

    global_step = tf.Variable(0,trainable = False, name = 'global_step')   # 학습 횟수 카운팅하는 변수
    
3️⃣ 앞 장에서보다 계층을 하나 더 늘리고 편향은 없이 가중치만 사용한 모델로 만들어보았다. 계층은 하나 늘었지만 모델이 더 간략해져서 신경망의 구성이 조금 더 명확히 드러날 것이다.

    X = tf.placeholder(tf.float32)
    Y = tf.placeholder(tf.float32)
    
    W1 = tf.Variable(tf.random_uniform([2,10], -1.,1.))
    L1 = tf.nn.relu(tf.matmul(X,W1))
    
    W2 = tf.Variable(tf.random_uniform([10,20], -1.,1.))   # 앞단 계층의 출력 크기가 10이고 뒷단 계층의 입력 크기가 20이기 때문
    L2 = tf.nn.relu(tf.matmul(L1,W2))
    
    W3 = tf.Variable(tf.random_uniform([20,3], -1.,1.))
    model = tf.matmul(L2,W3)
    
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = Y, logits = model))
    optimizer = tf.train.AdamOptiizer(learning_rate =0.01)
    train_op = optimizer.minimize(cost, global_step = global_step)
    
 - 마지막 줄에 보면 global_step 매개변수에 앞서 정의한 global_step변수를 넘겨준 것을 확인할 수 있다. 이렇게 하면 최적화 함수가 학습용 변수들을 최적화할 때마다 global_step변수의 값을 1씩 증가시키게 된다.   

4️⃣ 모델 구성이 끝났으니 이제 세션을 열고 최적화를 실행하기만 하면 된다. 그리고 모델을 불러들이고 저장하는 코드를 써본다.
    
    sess = tf.Session()
    saver = tf.train.Saver(tf.global_variables())   # global_variables는 앞서 정의한 변수들을 가져오는 함수, 정의한 변수들을 모두 가져와서 이 변수들을 파일에 저장하거나 이전에 학습한 결과를 불러와 담는 변수들로 사용함
    
5️⃣ 다음 코드는 ./model 디렉터리에 기존에 학습해둔 모델이 있는지를 확인해서 모델이 있다면 saver.restore함수를 사용해 학습된 값들을 불러오고 아니면 변수를 새로 초기화한다. 학습된 모델을 저장한 파일을 체크포인트파일이라고 한다.

    ckpt = tf.train_get_checkpoint_state('/model')
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
      saver.restore(sess, ckpt.model_checkpoint_path)    # 학습된 값들 불러오기
    else : 
      sess.run(tf.global_variables_initializer())   # 변수를 초기화
    
6️⃣ 간단하게 최적화를 수행한다. 이전과는 달리 step 값이 아니라 global_step값을 이용해 학습을 몇 번째 진행하고 있는지를 출력해준다. global_step은 텐서 타입의 변수이므로 값을 가져올때 sess.run(global_step)을 이용해야한다.

    for step in range(2):
      sess.run(train_op, feed_dict = {X:x_data, Y: y_data})
      print('Step: %d, ' %sess.run(global_step), 'Cost: %.3f' %sess.run(cost, feed_dict = {X:x_data, Y:y_data}))
      
   - 학습시킨 모델을 저장한 뒤 불러들여서 재학습한 결과를 보기 위해 학습 횟수를 2번으로 설정했다.
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
