# Chapter 09

GAN은 오토인코더와 같이 결과물을 생성하는 생성 모델 중 하나로 서로 대립하는 두 신경망을 경쟁시켜가며 결과물 생성 방법을 학습한다.

이해를 돕기 위해 적절한 비유를 들어보겠다.

위조지폐범(생성자)과 경찰(구분자)에 대한 이야기로, 위조지폐범은 경찰을 최대한 속이려고 하고 경찰은 위조한 지폐를 최대한 감별하려고 노력한다.

이처럼 위조지폐를 만들고 감별하려는 경쟁을 통해 서로의 능력이 발전하게 되고 그러다 보면 결국 위조지폐범은 진짜와 거의 구분할 수 없을 정도로 진짜 같은 위조지폐를 만들 수 있게 된다는 것이다.

먼저 실제 이미지를 주고 **구분자**에게 이 이미지가 진짜임을 판단하게 한다. 그 다음 **생성자**를 통해 노이즈로부터 임의의 이미지를 만들고 이것을 다시 같은 구분자를 통해 진짜 이미지인지를 판단하게 한다.

💡 이렇게 생성자는 구분자를 속여 진짜처럼 보이게 하고, 구분자는 생성자가 만든 이미지를 최대한 가짜라고 구분하도록 훈련하는 것이 GAN의 핵심이다. 생성자와 구분자의 경쟁을 통해 결과적으로는 생성자는 실제 이미지와 상당히 비슷한 이미지를 생성해낼 수 있게 된다.

이번 장에서는 이 GAN 모델을 활용하여 MNIST손글씨 숫자를 무작위로 생성하는 간단한 예제를 만들어보고 모델을 확장하여 원하는 숫자에 해당하는 이미지를 생성하는 모델을 만들어 볼 것이다.

## 9.1 GAN 기본 모델 구현하기

1. 먼저 필요한 라이브러리를 불러들인다. 생성된 이미지들을 보여줄 것이므로 matplotlib과 numpy도 같이 임포트한다.

       import tensorflow as tf
       import matplotlib.pyplot as plt
       import numpy as np
       
       from tensorflow.examples.tutorials.mnist import input_data
       mnist = input_data.read_data_sets("./mnist/data/", one_hot = True)
       
2. 다음으로는 하이퍼파라미터를 설정한다.

       total_epoch  = 100
       batch_size = 100
       learning_rate = 0.0002
       h_hidden = 256
       n_input = 28 * 28
       n_noise = 128    # 생성자의 입력값으로 사용할 노이즈의 크기

3. 플레이스홀더를 설정한다. GAN도 비지도 학습이므로 Y를 사용하지 않는다. 다만 구분자에 넣을 이미지가 실제 이미지와 생성한 가짜 이미지 두 개이고, 가짜 이미지는 노이즈에서 생성할 것이므로 노이즈를 입력할 플레이스홀더 Z를 추가한다.

       X = tf.placeholder(tf.float32, [None,n_input])
       Z = tf.placeholder(tf.float32, [None,n_noise])
      
4. 생성자 신경망에 사용할 변수들을 설정한다. 

       # 첫 번째 가중치, 편향 -> 은닉층으로 출력하기 위한 변수들
       G_W1 = tf.Variable(tf.random_normal([n_noise, n_hidden], stddev = 0.01))
       G_b1 = tf.Variable(tf.zeros([n_hidden])
       # 두 번째 가중치, 편향 -> 출력층에 사용할 변수들
       G_W2 = tf.Variable(tf.random_normal([n_hidden, n_input], stddev = 0.01))
       G_b2 = tf.Variable(tf.zeros([n_input])

5. 구분자 생성망에 사용할 변수들을 설정한다. 은닉층은 생성자와 동일하게 구성한다. 

       D_W1 = tf.Variable(tf.random_normal([n_input, n_hidden], stddev = 0.01))
       D_b1 = tf.Variable(tf.zeros([n_hidden]))
       D_W2 = tf.Variable(tf.random_normal([n_hidden, 1], stddev = 0.01))
       D_b2 = tf.Variable(tf.zeros([1]))       

❗ **실제 이미지를 판별하는 구분자 신경망과 생성한 이미지를 판별하는 구분자 신경망을 같은 변수를 사용해야한다. 같은 신경망으로 구분을 시켜야 진짜와 가짜 이미지를 구분하는 특징들을 동시에 잡아낼 수 있기 때문이다.**

6. 먼저 생성자 신경망을 구현해본다.

       def generator(noise_z):
         hidden = tf.nn.relu(tf.matmul(noise_z, G_W1) + b1)
         output = tf.nn.sigmoid(tf.matmul(hidden, G_W2) + b2)

         return output

    - 생성자는 무작위로 생성한 노이즈를 받아 가중치와 편향을 반영하여 은닉층을 만들고 은닉층에서 실제 이미지와 같은 크기의 결과값을 출력하는 간단한 구성이다.

7. 구분자 신경망 역시 같은 구성이지만 0~1사이의 스칼라값을 하나 출력하도록 하였고 이를 위한 활성화 함수로 sigmoid함수를 사용한다.

       def discriminator(inputs):
          hidden = tf.nn.relu(tf.matmul(inputs,D_W1) + D_b1)
          output = tf.nn.sigmoid(tf.matmul(hidden,D_W2) + D_b2)
          
          return output
          
8. 무작위한 노이즈를 만들어주는 간단한 함수를 만든다.

       def get_noise(batch_size, n_noise):
          return np.random_normal(size = (batch_size, n_noise))
          
9. 마지막으로 노이즈 Z를 이용해 가짜 이미지를 만들 생성자 G를 만들고 이 G가 만든 가짜 이미지와 진짜 이미지 X를 각각 구분자에 넣어 입력한 이미지가 진짜인지를 판별하도록 한다.

       G = generator(Z)
       D_gene = discriminator(G)
       D_real = discriminator(X)

10. 다음으로는 손실값을 구해야하는데 이번에는 두 개가 필요하다. 생성자가 만든 이미지를 구분자가 가짜라고 판단하도록 하는 손실값(경찰 학습용)과 진짜라고 판단하도록 하는 손실값(위조지폐범 학습용)을 구해야한다. 경찰을 학습시키려면 진짜 이미지 판별값 D_real은 1에 가까워야하고 가짜 이미지 판별값 D_gene은 0에 가까워야한다.

        loss_D = tf.reduce_mean(tf.log(D_real) + tf.log(1-D_gene))   # 이 값을 최대화하면 경찰 학습이 이루어짐

다음으로 위조지폐범 학습은 판별값 D_gene를 1에 가깝게 만들기만 하면된다. 즉 가짜 이미지를 넣어도 진짜 같다고 판별해야 한다. 다음과 같이 D_gene를 그대로 넣어 이를 손실값으로 하고 이 값을 최대화하면 위조지폐범을 학습시킬 수 있다.

    loss G = tf.reduce_mean(tf.log(D_gene))
    
✔ 즉 GAN의 학습은 loss_D와 loss_G 모두를 최대화하는 것이다. 다만 loss_D와 loss_G는 서로 연관되어 있어서 두 손실값이 항상 같이 증가하는 경향을 보이지는 않을 것이다. loss_D가 증가하려면 loss_G는 하락해야하고 반대로 loss_G가 증가하려면 loss_D는 하락해야하는 경쟁 관계이기 때문이다.


11. 이제 이 손실값들을 이용해 학습시키는 일만 남았다. 이 때 주의할 점은 loss_D를 구할 때 구분자 신경망에 사용되는 변수들만 사용하고 loss_G를 구할 때는 생성자 신경망에 사용되는 변수들만 사용하여 최적화해야 한다. 그래야 loss_D를 학습할 때는 생성자가 변하지 않고 loss_G를 학습할 때는 구분자가 변하지 않기 때문이다.

        D_var_list = [D_W1, D_b1, D_W2, D_b2]
        G_var_list = [G_W1, G_b1, G_W2, G_b2]
        
        train_D = tf.train.AdamOptimizer(learning_rate).minimize(-loss_D, var_list = D_var_list)   # loss를 최대화해야 하지만 최적화에 쓸 수 있는 함수는 minimize뿐이므로 최적화하려는 loss_D 앞에 음수 부호 붙여줌
        train_G = tf.train.AdamOptimizer(learning_rate).minimize(-loss_D, var_list = D_var_list)

12. 학습을 시키는 코드를 작성한다. 지금까지 본 학습 코드와 거의 같지만 이번 모델에서는 두 개의 손실값을 학습시켜야 해서 코드가 약간 추가되었다.

        sess = tf.Session()
        sess.run(tf.global_variables_initialier())
        
        total_batch = int(mnist.train.num_examples / batch_size)
        loss_val_D , loss_val_G = 0, 0   # loss_D와 loss_G의 결과값을 받을 변수
        
        for epoch in range(total_epoch):
            for i in range(total_batch):
                batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                noise = get_noise(batch_size, n_noise)   # 배치 크기만큼 노이즈 생성
                
                _, loss_val_D = sess.run([train_D, loss_D], feed_dict = {X: batch_xs, Y:noise})
                _. loss_val_G = sess.run([train_G, loss_G], feed_dict = {Z:noise})
                
            print('Epoch:', '%04d' % epoch, 'D loss: {:.4}'.format(loss_val_D), 'G loss: {:.4}'.format(loss_val_G)
            
13. 모델을 완성하였으니 이제 학습 결과를 확인하는 코드를 작성해본다. 학습이 잘 되는지는 0,9,19,29번째,, ~ 마다 생성기로 이미지를 생성하여 눈으로 직접 확인하도록한다. 결과를 확인하는 코드는 학습 루프 안에 작성해야한다.
            
            
        # 노이즈를 만들고 생성자 G에 넣어 결과값을 만듦    
        if epoch == 0 or (epoch+1) % 10 == 0:
            sample_size = 10
            noise = get_noise(sample_size,n_noise)
            samples = sess.run(G, feed_dict = {Z:noise})   # samples폴더는 미리 만들어져 있어야 함
            
            # 이 결과값들을 28 * 28 크기의 가짜 이미지로 만들어 samples폴더에 저장하도록함
            fix, ax = plt.subplots(1,sample_size, figsize = (sample_size,1))
            
            for i in range(sample_size):
                ax[i].set.axis_off()
                ax[i].imshow(np.reshape(sample[i], (28,28)))
                
            plt.savefig('samples/{}.png'.format(str(epoch).zfill(3)), bbox_inches='tight')
            plt.close(fig)
            
        print('최적화 완료!')
              
학습이 정상적으로 진행되었다면 학습 세대가 지나면서 이미지가 점점 더 그럴듯해지는 것을 볼 수 있다.

## 9.2 원하는 숫자 생성하기

이번에는 숫자를 무작위로 생성하지 않고 원하는 숫자를 지정해 생성하는 모델을 만들어볼 것이다.

1. 간단하게 노이즈에 레이블 데이터를 힌트로 넣어 주는 방법을 사용한다.

       import tensorflow as tf
       import matplotlib.pyplot as plt
       import numpy as np
       
       from tensorflow.examples.tutorials.mnist import input_data
       mnist = input_data.read_data_set("./mnist/data/", one_hot = True)
       
       total_epoch = 100
       batch_size = 100
       h_hidden = 256
       n_input = 28 * 28
       n_noise = 128
       n_class = 10
       
       X = tf.placeholder(tf.float32, [None,n_input])
       Y = tf.placeholder(tf.float32, [None,n_class])  # 결과값 판정용은 아니고 노이즈와 실제 이미지에 각각 해당하는 숫자를 힌트로 넣어주는 용도
       Z = tf.placeholder(tf.float32, [None,n_noise])

2. 생성자 신경망을 구성해볼건데 여기서는 변수들을 선언하지 않고 tf.layers를 사용한다. 앞서 본 것처럼 GAN 은 생성자와 구분자를 동시에 학습시켜야하고 따라서 학습 시 각 신경망의 변수들을 따로 학습시켜야 했다. 하지만 tf.layers를 사용하면 변수를 선언하지 않고 tf.variable_scope를 이용해 스코프를 지정해줄 수 있다.

       def generator(noise, labels):
          with tf.variable_scope('generator'):
              inputs = tf.concat([noise,labels],1)    # tf.concat함수를 이용해 noise값에 labels정보를 간단하게 추가
              hidden = tf.layers.dense(inputs,n_hidden, activation = tf.nn.relu)
              output = tf.layers.dense(hidden,n_input, activation = tf.nn.sigmoid)
             
          return output

3. 생성자 신경망과 같은 방법으로 구분자 신경망을 만든다. 여기서 주의할 점은 진짜 이미지를 판별할 때와 가짜 이미지를 판별할 때 똑같은 변수를 사용해야 한다는 점이다. 그러기 위해 scope.reuse_variables함수를 이용해 이전에 사용한 변수를 재사용하도록 짠다.

       def discriminator(inputs, labels, reuse = None):
          with tf.variable_scope('discriminaor') as scope :
              if reuse : 
                  scope.reuse_variables()
              input = tf.concat([inputs, labels],1)
              hidden = tf.layers.dense(inputs, n_hidden, activation = tf.nn.relu)
              output = tf.layers.dense(hidden, 1, activation = None)   # 출력값에 활성화 함수 사용 안 함 -> 손실값 계산에 sigmoid_cross_entropy_with_logits함수 사용하기 위함
              
          return output
          
4. 노이즈 생성 유틸리티 함수에서 이번에는 노이즈를 균등분포로 생성하도록 작성한다.

       def get_noise(batch_size, n_noise):
          return np.random.uniform(-1.,1., size = [batch_size, n_noise])
          
5. 생성자를 구하고 진짜 이미지 데이터와 생성자가 만든 이미지 데이터를 이용하는 구분자를 하나씩 만들어준다. 이 때 생성자에는 레이블 정보를 추가하여 추후 레이블 정보에 해당하는 이미지를 생성할 수 있도록 유도한다. 그리고 가짜 이미지 구분자를 만들 때는 진짜 이미지 구분자에서 사용한 변수들을 재사용하도록 reuse옵션을 True로 설정한다.

       G = generator(Z,Y)
       D_real = discriminator(X,Y)
       D_gene = discriminator(G,Y, True)
       
6. 손실 함수를 만들 차례이다. 앞과 똑같이 진짜 이미지를 판별하는 D_real은 1에 가까워지도록하고 가짜 이미지를 판별하는 D_gene값은 0에 가까워지도록 하는 것이지만 simoid_cross_entropy_with_logits 함수를 이용하면 코드를 좀 더 간편하게 작성할 수 있다.

       loss_D_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = D_real, labels = tf.ones_like(D_real)))  # D_real 결과값과 D_real크기만큼 1로 채운 값들을 비교
       loss_D_gene = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = D_gene, labels = tf.zeros_like(D_gene))  # D_gene 결과값과 D_gene크기만큼 0으로 채운 값들을 비교
       loss D = loss_D_real + loss_D_gene   # 이 값을 최소화하면 구분자를 학습시킬 수 있음

   
7. 그런 다음 loss_G를 구한다. loss_G는 생성자를 학습시키기 위한 손실값으로 sigmoid_cross_entropy_with_logits함수를 이용하여 D_gene를 1에 가깝게 만드는 값을 손실값으로 취하도록 한다.

       loss_G = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = D_gene, labels = tf.ones_like(D_gene)))
       
8. 마지막으로 텐서플로가 제공하는 tf.get_collection 함수를 이용해 discriminator와 generator스코프에서 사용된 변수들을 가져온 뒤 이 변수들을 최적화에 사용할 각각의 손실 함수와 함께 최적화 함수에 넣어 학습 모델 구성을 마무리 한다.

       vars_D = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'discriminator')
       vars_G = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'generator')
       
       train_D = tf.train.AdamOptimizer().minimize(loss_D, var_list = vars_D)
       train_G = tf.train.AdamOptimizer().minimize(loss_G, var_list = vars_G)

9. 학습을 진행하는 코드를 작성한다. 앞서 만든 GAN 모델과 거의 똑같지만 플레이스홀더 Y의 입력값으로 batch_ys값을 넣어준다는 것만 주의하면 된다.

       sess = tf.Session()
       sess.run(tf.global_variables_initializer())
       
       total_batch = int(mnist.train.num_examples/batch_size)
       loss_val_D, loss_val_G = 0,0
       
       for epoch in range(total_epoch):
          for i in range(total_batch):
              batch_xs, batch_ys = mnist.train.next_batch(batch_size)
              noise = get_noise(batch_size,n_noise)
              
              _, loss_val_D = sess.run([train_D, loss_D], feed_dict = {X:batch_xs, Y: batch_ys, Z:noise})
              _, loss_val_G = sess.run([train_G, loss_G], feed_dict = {Y:batch_ys, Z: noise})
              
          print('Epoch:', '%04d' % epoch, 'D loss: {:.4}'.format(loss_val_D), 'G loss: {:.4}'.format(loss_val_G))
       

10. 학습 중간중간에 생성자로 만든 이미지를 저장하는 코드를 작성한다. 플레이스홀더 Y의 입력값을 넣어준다는 것이 다르고 진짜 이미지와 비교해보기 위해 위쪽에는 진짜 이미지를 출력하고 아래쪽에는 생성한 이미지를 출력하도록 하였다.

        if epoch == 0 or (epoch+1) % 10 == 0:
            sample_size = 10
            noise = get_noise(sample_size, n_noise)
            samples = sess.run(G, feed_dict = {Y: mnist.test.labels[:sample_size], Z:noise})
            
            fit, ax = plt.subplots(2,sample_size, figsize=(sample_size,2))
            
            for i in range(sample_size):
                ax[0][i].set_axis_off()
                ax[1][i].set_axis_off()
                
                ax[0][i].imshow(np.reshape(mnist.test.images[i], (28,28)))
                ax[1][i].imshow(np.reshape(samples[i], (28,28)))
                
            plt.savefig('samples2/{}.png'.format(str(epoch).zfill(3)), bbox_inches='tight')
            plt.close(fig)
         
        print('최적화 완료!')



















          
          
          
          
          
