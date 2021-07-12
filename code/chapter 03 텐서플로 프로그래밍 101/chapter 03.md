# Chapter 03 #

**텐서플로**는 딥러닝 프레임워크로 유명하지만 사실 딥러닝용으로만 사용할 수 있는 것은 아니다.

**텐서플로**는 그래프 형태의 수학식 계산을 수행하는 핵심 라이브러리를 구현한 후 그 위에 딥러닝을 포함한 여러 머신러닝을 쉽게 할 수 있는 다양한 라이브러리를 올린 형태이다.

이를 위해 텐서플로는 일반적인 프로그래밍 방식과는 약간 다른 개념들을 포함한다. 

✅ 이번 장에서는 텐서플로를 이용하는데 필요한 **텐서(tensor), 플레이스 홀더(placeholder), 변수(variable)** 그리고 **연산**의 개념과 **그래프**를 실행하는 기본적인 방법을 배울 것이다.

## 3.1 텐서와 그래프 실행

1️⃣ 가장 먼저 텐서플로를 사용하기 위해 텐서플로 라이브러리를 임포트한다.

    import tensorflow as tf
    
2️⃣ 다음은 tf.constant로 상수를 hello 변수에 저장하는 코드이다. 텐서플로의 상수는 일반 프로그래밍 언어에서 써온 상수와 같다고 보면 된다.

    hello = tf.constant('Hello, TensorFlow!')
    print(hello)
    
  - 여기까지 입력하고 소스코드를 실행하면 다음 결과를 볼 수 있다.

        Tensor("Const:0", shape=(), dtype = string)   # dtype : 해당 텐서에 담긴 요소들의 자료형, 배열과 비슷하다고 생각하면 됨
        
    hello 변수의 값을 출력한 결과로, hello가 텐서플로의 **텐서**라는 자료형이고 상수를 담고 있음을 의미한다.
    
    📍 텐서는 텐서플로에서 다양한 수학식을 계산하기 위한 가장 기본적이고 중요한 자료형이며 다음과 같이 **랭크(rank)** 와 **셰이프(shape)** 라는 개념을 가지고 있다. 
    
    **랭크**는 차원의 수를 나타내는 것으로 랭크가 0이면 스칼라, 1이면 벡터, 2이면 행렬, 3이상이면 n-Tensor 또는 n차원 텐서라고 부른다. 
    
    **셰이프**는 각 차원의 요소 개수로, 텐서의 구조를 설명해준다.
    
        3   # 랭크가 0인 텐서, 셰이프는 []   #  스칼라
        [1. , 2. , 3.]   # 랭크가 1인 텐서, 셰이프는 [3]    # 벡터
        [[1. , 2. , 3.], [4. , 5. , 6.]]    # 랭크가 2인 텐서, 셰이프는 [2,3]   # 행렬
        [[[1. , 2. , 3.], [4. , 5. , 6.]]]   # 랭크가 3인 텐서, 셰이프는 [2,1,3]    # n차원 텐서
    
3️⃣ 이 텐서를 이용해 다양한 연산을 수행할 수 있으며 덧셈은 다음처럼 간단히 할 수 있다. 

    a = tf.constant(10)
    b = tf.constant(32)
    c = tf.add(a,b)
    print(c)
    
  - 이 코드를 실행하면 42가 나올 것으로 생각할 수 있지만 다음과 같이 텐서의 형태를 출력한다.
    
        Tensor("Add:0", shape=(), dtype=int32)
         
     그 이유는 텐서플로 프로그램의 구조가 다음의 두 가지로 분리되어 있기 때문이다.
     
     1) 그래프 생성
     2) 그래프 실행
     
![image](https://user-images.githubusercontent.com/66320010/125234467-c5f93d80-e31b-11eb-86d7-d33090ca2f0f.png)

   **그래프**는 간단하게 말해 텐서들의 연산 모음이라고 생각하면 된다. 
   
   텐서플로는 텐서와 텐서의 연산들을 먼저 정의하여 그래프를 만들고 이후 필요할 때 연산을 실행하는 코드를 넣어 '원하는 시점'에 실제 연산을 수행하도록 한다.
   
   이러한 방식을 **지연 실행(lazy evaluation)** 이라고 하며 함수형 프로그래밍에서 많이 사용한다.
   
   이런 방식을 통해 실제 계산은 C++로 구현한 코어 라이브러리에서 수행하므로 파이썬으로 프로그램을 작성하지만 매우 뛰어난 성능을 얻을 수 있다.
   
   또한 모델 구성과 실행을 분리하여 프로그램을 깔끔하게 작성할 수 있다.
   
4️⃣ 그래프의 실행은 Session안에서 이뤄져야하며 다음과 같이 Session객체와 run메서드를 이용하면 된다.

    sess = tf.Session()
     
    print(sess.run(hello))
    print(sess.run([a,b,c]))
     
    sess.close()
 
  이 코드를 실행하면 다음과 같이 기대한 계산 결과를 볼 수 있다.

    b'Hello, TensorFlow!'
    [10,32,42]
    
📍 전체 코드 📍

    import tensorflow as tf
    
    hello = tf.constant('Hello, TensorFlow!')
    print(hello)
    
    a = tf.constant(10)
    b = tf.constant(32)
    c = tf.add(a,b)
    print(c)
    
    # 그래프 실행
    sess = tf.Session()
    print(sess.run(hello))
    print(sess.run([a,b,c]))
    
    sess.close()
    
## 3.2 플레이스홀더와 변수

텐서플로로 프로그래밍할 때 알아야 할 가장 중요한 개념 두 개가 있다면 바로 플레이스홀더와 변수이다.

1️⃣ **플레이스홀더**는 그래프에 사용할 입력값을 나중에 받기 위해 사용하는 매개변수라고 생각하면 된다. **변수**는 텐서플로가 학습한 결과를 갱신하기 위해 사용하는 변수이다. 이 변수들의 값들이 바로 신경망의 성능을 좌우하게 된다.

- 먼저 플레이스홀더는 다음과 같이 사용한다.

      X = tf.placeholder(tf.float32, [None,3])   # 텐서를 미리 만듦, tf.placeholder(dtype,shape,name)
      print(X) 

    이 코드를 실행하면 다음과 같이 Placeholder라는 (?,3) 모양의 float32 자료형을 가진 텐서가 생성된 것을 확인할 수 있다.
    
      Tensor("Placeholder:0", shape = (?,3), dtype = float32)

2️⃣ 나중에 플레이스홀더 X에 넣을 자료를 다음과 같이 정의해볼 수 있다. 앞에서 텐서 모양을 (?,3)으로 정의했으므로 두 번째 차원은 요소를 3개씩 가지고 있어야 한다.

    x_data = [[1,2,3],[4,5,6]]
    
3️⃣ 다음은 변수들을 정의해본다.

     W = tf.Variable(tf.random_normal([3,2]))
     b = tf.Variable(tf.random_normal([2,1]))
     
   각각 W와 b에 텐서플로의 변수를 생성하여 할당한다.
   
   W는 [3,2] 행렬 형태의 텐서, b는 [2,1] 행렬 형태의 텐서로, tf.random_normal함수를 이용해 정규분포의 무작위 값으로 초기화한다.
   
   물론 다른 생성 함수를 사용하거나 다음처럼 직접 원하는 텐서의 형태의 데이터를 만들어서 넣어줄 수도 있다.
   
    W = tf.Variable([[0.1,0.1],[0.2,0.2],[0.3,0.3]])
    
4️⃣ 다음으로 입력값과 변수들을 계산할 수식을 작성해보자. X와 W가 행렬이기 때문에 tf.matmul함수를 사용해야한다. 행렬이 아닌 경우에는 단순히 곱셈 연산자(* )나 tf.mul 함수를 사용하면 된다.

    expr = tf.matmul(X,W) + b
    
   행렬곱 정의에 따라 앞서 X에 넣을 데이터를 [2,3] 형태의 행렬로 정의했으므로, 행렬곱을 하기 위해서 W의 형태를 [3,2]로 정의한 것이다. 참고로 [2,3]행렬이면 2가 행의 개수, 3이 열의 개수이다. 

5️⃣ 이제 연산을 실행하고 결과를 출력하여 설정한 텐서들과 계산된 그래프의 결과를 확인해보자.

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())   # 앞에서 정의한 변수들 초기화
    
    print("=== x_data ===")
    print(x_data)
    print("=== W ===")
    print(sess.run(W))
    print("=== b ===")
    print(sess.run(b))
    print("=== expr ===")
    print(sess.run(expr,feed_dict = {X:x_data}))
    
 - 두 번째 줄의 tf.global_variables_initializer는 앞에서 정의한 변수들을 초기화 하는 함수이다. 기존에 학습한 값들을 가져와서 사용하는 것이 아닌 처음 실행하는 것이라면 연산을 실행하기 전에 반드시 이 함수를 이용해 변수들을 초기화해야한다.
 
 - feed_dict 매개변수는 그래프를 실행할 때 사용할 입력값을 지정한다.
 
 - expr 수식에는 X,W,b를 사용했는데 이중 X가 플레이스홀더라 X에 값을 넣어주지 않으면 계산에 사용할 값이 없으므로 에러가 난다. 따라서 미리 정의해둔 x_data를 X의 값으로 넣어주었다.

 - 실행한 결과를 보면 X(즉 x_data)는 [2,3], W는 [3,2] 형태이고, 결과값은 행과 열의 수가 X의 행의 개수와 W의 열의 개수인 [2,2]형태임을 확인할 수 있다.

📍 전체 코드 📍

    import tensorflow as tf
    
    X = tf.placeholder(tf.float32, [None,3])
    
    x_data = [[1,2,3],[4,5,6]]
    
    W = tf.Variable(tf.random_normal([3,2]))
    b = tf.Variable(tf.random_normal([2,1]))
    
    expr = tf.manmul(X,W) +b
    
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    print("=== x_data ===")
    print(x_data)
    print("=== W ===")
    print(sess.run(W))
    print("=== b ===")
    print(sess.run(b))
    print("=== expr ===")
    print(sess.run(expr,feed_dict = {X:x_data}))
    
    sess.close()

## 3.3 선형 회귀 모델 구현하기

**선형회귀**란 간단하게 말해 주어진 x와 y값을 가지고 서로 간의 관계를 파악하는 것이다.

이 관계를 알고 나면 새로운 x값이 주어졌을 때 y값을 쉽게 알 수 있다. 

어떤 입력에 대한 출력을 예측하는 것, 이것이 바로 머신러닝의 기본이다.

이제 텐서플로의 최적화 함수를 이용해 X와 Y의 상관관계를 분석하는 기초적인 선형 회귀 모델을 만들고 실행해보자.

여기서는 다음과 같이 주어진 x_data와 y_data의 상관관계를 파악해보고자 한다.

    x_data = [1,2,3]
    y_data = [1,2,3]
 
1️⃣ 먼저 x와 y의 상관관계를 설명하기 위한 변수들인 W와 b를 각각 -1.0부터 1.0사이의 균등분포(uniform distribution)를 가진 무작위 값으로 초기화한다.

    W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
    b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
    
2️⃣ 다음은 자료를 입력받을 플레이스홀더를 설정한다.

    X = tf.placeholder(tf.float32, name = "X")
    Y = tf.placeholder(tf.float32, name = "Y")
    
  - name없이 설정한 플레이스홀더를 출력하면 이름이 자동으로 부여된다. 
  
  - 각각의 텐서에 이름을 주면 어떠한 텐서가 어떻게 사용되고 있는지 쉽게 알 수 있고, 텐서보드에서도 이 이름을 출력해주므로 디버깅도 더 수월하게 할 수 있다.  
    
3️⃣ 그다음으로 X와 Y의 상관관계(선형관계)를 분석하기 위한 수식을 작성한다.

    hypothesis = W * X + b
    
  이 수식은 W와의 곱과 b와의 합을 통해 X와 Y의 관계를 설명하겠다는 뜻이다. 다시말해 X가 주어졌을 때 Y를 만들어낼 수 있는 W와 b를 찾아내겠다는 의미이기도 하다.
  
  W는 가중치, b는 편향이라고 하며, 이 수식은 선형회귀는 물론 신경망 학습에 가장 기본이 되는 수식이다.
  
  여기서 W와 X가 행렬이 아니므로 tf.matmul이 아니라 기본 곱셈 연산자를 사용하였다.
    
4️⃣ 손실 함수를 작성해보자.

**손실 함수**는 한 쌍(x,y)의 데이터에 대한 손실값을 계산하는 함수이다. 

**손실값이란 실제값과 모델로 예측한 값이 얼마나 차이가 나는가를 나타내는 값이다.**

즉 손실값이 작을수록 그 모델이 X와 Y의 관계를 잘 설명하고 있다는 뜻이다. 

손실을 전체 데이터에 대해 구한 경우 이를 비용(cost)라고 한다.

**학습**이란 변수들의 값을 다양하게 넣어 계산해보면서 이 손실값을 최소화하는 W와 b값을 구하는 것이다. 손실값으로는 '예측값과 실제값의 거리'를 가장 많이 사용한다.

따라서 손실값은 예측값에서 실제값을 뺀 뒤 제곱하여 그리고 비용은 모든 데이터에 대한 손실값의 평균을 내어 구한다.

    cost = tf.reduce_mean(tf.square(hypothesis - Y))
    
5️⃣ 마지막으로 텐서플로가 기본 제공하는 **경사하강법** 최적화 함수를 이용해 손실값을 최소화하는 연산 그래프를 생성한다.

    optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.1)
    train_op = optimizer.minizier(cost)
    
  **최적화 함수**란 가중치와 편향 값을 변경해가면서 손실값을 최소화하는 가장 최적화된 가중치와 편향을 찾아주는 함수이다.
  
  이때 값들을 무작위로 변경하면 시간이 너무 오래걸리고 학습 시간도 예측하기 어려워서 빠르게 최적화하기 위한, 즉 빠르게 학습하기 위한 다양한 방법을 사용한다.
  
  **경사하강법**은 그러한 최적화 방법 중 가장 기본적인 알고리즘으로 함수의 기울기를 구하고 기울기가 낮은 쪽으로 계속 이동시키면서 최적의 값을 찾아 나가는 방법이다.
  
  **학습률**은 학습을 얼마나 '급하게' 할 것인가를 설정하는 값이다. 값이 너무 크면 최적의 손실값을 찾지 못하고 지나치게 되고 값이 너무 작으면 학습 속도가 매우 느리다.
  
  이렇게 학습을 진행하는 과정에 영향을 주는 변수를 하이퍼파라미터라고 하며 이 값에 따라 학습 속도나 신경망 성능이 크게 달라질 수 있다.
  
6️⃣ 선형 회귀 모델을 다 만들었으니 그래프를 실행해 학습을 시키고 그 결과를 확인해보자.

앞 절에서와 같이 세션을 생성하고 변수들을 초기화한다. 

이번에는 파이썬의 with기능을 이용해 세션 블록을 만들고 세션 종료를 자동으로 처리하도록 해보았다.

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
       
다음은 최적화 함수를 수행하는 그래프인 train_op를 실행하고 실행 시마다 변화하는 손실값을 출력하는 코드이다.

    for step in range(100):
        _, cost_val = sess.run([train_op,cost], feed_dict = {X:x_data, Y:y_data})
        print(step, cost_val, sess.run(W), sess.run(b))
        
  - 손실값이 점점 줄어든다면 학습이 정상적으로 이루어지고 있는 것이다.

7️⃣ 학습에 사용되지 않았던 값인 5와 2.5를 X값으로 넣고 결과를 확인해보자.

    print("X:5, Y:", sess.run(hypothesis, feed_dict= {X:5}))
    print("X:2.5, Y:", sess.run(hypothesis, feed_dict= {X:2.5}))
    
📍 전체 코드 📍

    import tensorflow as tf
    
    x_data = [1,2,3]
    y_data = [1,2,3]
    
    W = tf.Variable(tf.random_uniform([1],-1.0,1.0))
    b = tf.Variable(tf.random_uniform([1],-1.0,1.0))
    
    X = tf.placeholder(tf.float32, name = "X")
    Y = tf.placeholder(tf.float32, name = "Y")
    
    hypothesis = W * X + b
    
    cost = tf.reduce_mean(tf.square(hypothesis - Y))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.1)
    train_op = optimizer.minimize(cost)
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        for step in range(100):
            _, cost_val = sess.run([train_op, cost], feed_dict = {X:x_data, Y:y_data})
            
        print(step, cost_val, sess.run(W), sess.run(b))
        
     print("\n === Test ===")
     print("X:5, Y:", sess.run(hypothesis, feed_dict= {X:5}))
     print("X:2.5, Y:", sess.run(hypothesis, feed_dict= {X:2.5}))
    
    
    
    
    
    
    
    
    
    
