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
    
7️⃣ 마지막으로 최적화가 끝난 뒤 학습된 변수들을 지정한 체크포인트 파일에 저장한다.

    saver.save(sess,'./model/dnn.ckpt', global_step = global_step)   # 두 번째 매개변수는 체크포인트 파일이 위치와 이름
    
  - global_step의 값은 저장할 파일의 이름에 추가로 붙게되며, 텐서 변수 또는 숫자값을 넣어줄 수 있다. 이를 이용해 여러 상태의 체크포인트를 만들 수 있고 가장 효과적인 체크포인트를 선별해서 사용할 수 있다.
    
8️⃣ 예측 결과와 정확도를 확인하는 다음 코드를 마지막으로 넣고 실행 결과를 확인해본다.

    prediction = tf.argmax(model,1)
    target = tf.argmax(Y,1)
    print('예측값: ' ,sess.run(prediction, feed_dict = {X:x_data}))
    print('실제값: ', sess.run(target, feed_dict = {Y:y_data}))
    is_correct = tf.equal(prediction, target)
    accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
    print('정확도: %.2f' %sess.run(accuracy * 100, feed_dict = {X:x_data, Y:y_data}))
    
  - 프로그램을 처음 실행하고 다시 한번 더 실행해본다.
    
  - 결과는 기대한 대로 2번을 실행하지만 global_step으로 저장한 값을 불러와서 증가시켰으므로 step이 3부터 시작했고 정확도도 올라간 것을 확인할 수 있다.
  
  - 이 방식을 응용해 모델 구성, 학습, 예측 부분을 각각 분리하여 학습을 따로 한 뒤 예측만 단독으로 실행하는 프로그램을 작성할 수도 있다.
    
## 5.2 텐서보드 사용하기

딥러닝 라이브러리와 프레임워크는 많지만 유독 텐서플로를 사용하는 곳이 급증한 데에는 텐서보드의 역할이 가장 컸다고해도 과언이 아니다.

딥러닝을 현업에 활용하게 되면 대부분의 경우 학습 시간이 상당히 오래걸리기 때문에 모델을 효과적으로 실험하려면 학습 과정을 추적하는 일이 매우 중요하지만 번거로운 추가 작업을 많이 해야한다.

이러한 어려움을 해결해주고자 텐서플로는 **텐서보드**라는 도구를 기본으로 제공한다.

**텐서보드**는 학습하는 중간중간 손실값이나 정확도 또는 결과물로 나온 이미지나 사운드 파일들을 다양한 방식으로 시각화해서 보여준다.

코드 몇 줄만 추가하면 이런 기능을 매우 쉽게 사용할 수 있다.
    
앞서 만든 코드에 텐서보드를 이용하기 위한 코드를 넣어보도록 하자.

이를 통해 신경망 계층의 구성을 시각적으로 확인하고 손실값의 변화도 그래프를 이용해 직관적으로 확인해보자.

1️⃣ 먼저 데이터를 읽어 들이는 코드와 플레이스홀더 값들을 똑같이 넣는다.

    import tensorflow as tf
    import numpy as np
    
    data = np.loadtxt('.data/csv', delimiter = ',', unpack = True, dtype = 'float32')
    x_data = np.transpose(data[0:2])
    y_data = np.transpose(data[2:])
    
    global_step = tf.Variable(0,trainable = False, name = 'global_step')   # 학습 횟수 카운팅하는 변수

    X = tf.placeholder(tf.float32)
    Y = tf.placeholder(tf.float32)
    
2️⃣ 그 다음으로 신경망의 각 계층에 다음 코드를 덧붙여준다.    
    
    with tf.name_scope('layer1'):
        W1 = tf.Variable(tf.random_uniform([2,10], -1., 1.), name = 'W1')   # name ='W1'처럼 쓰면 텐서보드에서 해당 이름의 변수가 어디서 사용되는지 쉽게 알 수 있음, 이름은 변수뿐만 아니라 플레이스홀더, 각각의 연산, 활성화 함수 등 모든 텐서에 붙일 수 있음
        L1 = tf.nn.relu(tf.matmul(X,W1))
        
  - with tf.name_scope로 묶은 블록은 텐서보드에서 한 계층 내부를 표현해준다.
    
이렇게 다른 계층들도 전부 tf.name_scope로 묶어주고 이름도 붙여주자.

    with tf.name_scope('layer2'):
        W2 = tf.Variable(tf.random_uniform([10,20], -1., 1.), name = 'W2')
        L2 = tf.nn.relu(tf,matmul(L1,W2))
        
    with tf.name_scope('output'):
        W3 = tf.Variable(tf.random_uniform[20,3] , -1., 1.), name = 'W3')
        model = tf.matmul(L2,W3)
        
    with tf.name_scope('optimizer'):
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=model))
        optimizer = tf.train.AdamOptimizer(learning_rate = 0.01)
        train_op = optimizer.minimize(cost, global_step = global_step)
        
3️⃣ 다음으로 손실값을 추적하기 위해 수집할 값을 지정하는 코드를 작성한다.

    tf.summary.scalar('cost',cost)
    
 - 위와 같은 코드로 cost텐서의 값을 손쉽게 지정할 수 있다. scalar함수는 값이 하나인 텐서를 수집할 때 사용한다. 물론 scalar뿐만 아니라 histogram, image, audio등 다양한 값을 수집하는 함수를 기본으로 제공한다.

4️⃣ 이제 모델을 불러들이거나 초기화하는 코드를 넣는다.

    sess = tf.Session()
    saver = tf.train.Saver(tf.global_variables())
    
    ckpt = tf.train.get_checkpoint_state('./model')
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpint_path):
        saver.restore(sess, ckpt.model_checkpoint_path)
        
    else:
        sess.run(tf.global_variables_initializer())
        
5️⃣ tf.summary.merge_all함수로 앞서 지정한 텐서들을 수집한 다음 tf.summary.FileWriter함수를 이용해 그래프와 텐서들의 값을 저장할 디렉터리를 설정한다.

    merged = tf.summary.merge_all()   # 앞서 지정한 텐서 수집
    writer = tf.summary.FileWriter('./logs', sess.graph)   # 그래프와 텐서값 저장할 디렉터리 설정
    
6️⃣ 그런다음 최적화를 실행하는 코드를 작성한다.

    for step in range(100):
        sess.run(train_op, feed_dict = {X:x_data, Y:y_data})
        
        print('Step: %d,' % sess.run(global_step), 'Cost:%.3f' % sess.run(cost, feed_dict = {X:x_data, Y:y_data}))
        
7️⃣ sess.run을 이용해 앞서 merged로 모아둔 텐서의 값들을 계산하여 수집한 뒤, writer.add_summary함수를 이용해 해당 값들을 앞서 지정한 디렉터리에 저장한다. 또한 적절한 시점에 값들을 수집하고 저장하면되고 나중에 확인할 수 있도록 global_step값을 이용해 수집한 시점을 기록해둔다.

    summary = sess.run(merged, feed_dict = {X:x_data, Y:y_data})
    writer.add_summary = (summary, global_step = sess.run(global_step))
    
8️⃣ 마지막으로 모델을 저장하고 예측하는 부분을 작성한다.

    saver.saver(sess,'./model/dnn.ckpt', global_step = global_step)
    
    prediction = tf.argmax(model,1)
    target = tf.argmax(Y,1)
    print('예측값:', sess.run(prediction, feed_dict = {X:x_data}))
    print('실제값:', sess.run(target, feed_dict = {Y:y_data}))
    
    is_correct = tf.equal(prediction, target)
    accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
    print('정확도: %.2f' %sess.run(accuracy *100, feed_dict = {X:x_data, Y:y_data}))
    
 - 실행하고 나면 현재 디렉터리에 logs라는 디렉터리가 새로 생긴 것을 볼 수 있다. 
 
윈도우 명령 프롬프트에서 다음 명령어를 입력한다.

    tensorboard --logdir=./logs
 
명령을 실행하면 다음과 같은 메시지가 출력되면서 웹서버가 실행된다.

    Starting Tensorboard b'41' on port 6006
    
웹 브라우저를 이용해 다음 주소로 들어가면 텐서보드의 내용을 확인할 수 있다. 주소 맨 마지막의 6006번은 앞의 출력 메시지의 맨 마지막 숫자이다.

    http://localhost:6006
    
웹 브라우저에서 정상적으로 페이지가 열린다면 [SCALARS]와 [GRAPHS]메뉴에 들어가보자.

[SCALARS] 메뉴에는 tf.summary.scalar('cost,cost)로 수집한 손실값의 변화를 그래프로 직관적으로 확인할 수 있다.

[GRAPHS] 메뉴에서는 with tf.name_scope로 그룹핑한 결과를 그림으로 확인할 수 있다.

참고로 각 가중치와 편향 등의 변화를 그래프로 살펴보고 싶다면 다음 코드를 넣고 학습을 진행하면된다. 텐서보드의 [DISTRIBUTIONS],[HISTOGRAMS]메뉴에서 그래프로 확인할 수 있다.

    tf.summary.histogram("Weights",W1)
    
    
