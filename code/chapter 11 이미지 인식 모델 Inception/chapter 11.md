# Chapter 11

지금까지 학습을 통해 텐서플로를 사용하면 신경망 모델을 매우 쉽게 만들 수 있었다.

또한 다수의 사람이 풀고자 하는 공통 문제에 대해서 이미 많은 연구자가 훌륭한 모델을 많이 만들어 놓았다.

그러한 공개 모델 중 구글이 만든 **인셉션(Inception)** 이 있다.

이는 이미지 인식 모델로 이미지 인식 대회인 ILSVRC에서 2014년에 우승한 전력이 있다.

인셉션은 기본적으로 작은 컨볼루션 계층을 매우 많이 연결한 것인데 구성이 꽤 복잡해서 구현하기가 조금 까다롭다.

그런데 구글이 텐서플로로 구현해 공개해두었고 또 그 모델을 다양한 목적에 쉽게 활용할 수 있도록 간단한 스크립트까지 제공하고 있다.

✔ 이번 장에서는 이 스크립트를 사용해 꽃 사진을 학습시키고 꽃의 종류를 알아내는 간단한 스크립트를 작성하고 테스트해본다.

## 11.1 자료 준비

인터넷에서 자료를 두 개 내려받아야 한다.

**하나는 학습을 위한 꽃 사진이고 다른 하나는 일정한 기준에 따라 사진을 학습시키는 스크립트이다.**

이 두 자료 모두 텐서플로 홈페이지에 있다.

먼저 학습 자료와 학습시킨 모델을 저장할 디렉터리를 'workspace'라는 이름으로 만들고 꽃 사진은 workspace 디렉터리에, 학습 스크립트는 현재 디렉터리에 저장한다.

그러면 디렉터리 구조는 다음과 같다.

retrain.py /workspace/flower_photos/daisy, dandelion, roses, sunflowers, tulips

retrain.py 스크립트는 디렉터리 이름을 꽃 이름으로 하여 각 디렉터리에 있는 사진들을 학습 시킨다.

## 11.2 학습시키기

자료가 다 준비됐으니 이제 retrain.py 스크립트로 꽃 사진들을 학습시켜본다.

윈도우 명령 프롬프트를 열고 retrain.py 파일이 있는 위치에서 다음 명령을 실행한다.

    C:\> python retrain.py \
        --bottleneck_dir = ./workspace/bottleneck \    # 학습할 사진을 인셉션용 학습 데이터로 전환해서 저장해둘 디렉터리
        --model_dir = ./workspace/incenption \         # 인셉션 모델을 내려받을 경로
        --output_graph = ./workspace/flowers_graph.pb \    # 학습된 모델(.pb)을 저장할 경로
        --output_labels = ./workspace/flower_labels.txt    # 레이블 이름들을 저장해둘 파일 경로
        --image_dir ./workspace/flower_photos \     # 원본 이미지 경로
        --how_many_training_steps 1000    # 반복 학습 횟수
        
  - 스크립트를 실행하면 학습이 이루어진다.

## 11.3 예측 스크립트

1️⃣ retrain.py 파일이 들어있는 텐서플로 저장소에는 이미지를 예측하는 스크립트도 포함되어 있다. 이 스크립트를 사용해도 되지만 그 내용을 파악해두면 나중에 활용하기가 좋다. 따라서 간략한 버전을 작성하면서 어떤 내용으로 채워져 있는지 알아본다.

    import tensorflow as tf
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    import sys

  - matplotlib 라이브러리는 맨 마지막에 예측 결과가 맞는지 눈으로 확인할 수 있도록 이미지를 출력하는 데 사용한다.

   ❗ 이미지 처리에 pillow 라이브러리가 필요하니 미리 설치해둬야 한다.

2️⃣ 다음으로는 텐서플로의 유명한 모듈인 app.flags를 사용해 스크립트에서 받을 옵션과 기본값을 설정한다. tf.app모듈을 사용하면 터미널이나 명령 프롬프트에서 입력받는 옵션을 쉽게 처리할 수 있다.

    tf.app.flags.DEFINE_string("output_graph", "./workspace/flowers_graph.pb", "학습된 신경망이 저장된 위치")
    tf.app.flags.DEFINE_string("output_labels", "./workspace/flowers_labels.txt", "학습할 레이블 데이터 파일")
    tf.app.flags.DEFINE_boolean("show_image", True-, "이미지 추론 후 이미지를 보여줍니다.")
    
    FLAGS = tf.app.flags.FLAGS
    
 - 앞서 설명한 방법으로 retrain.py로 학습을 진행하면 workspace디렉터리의 flowers_labels.txt 파일에 꽃의 이름을 전부 저장해두게 된다.
 - 한줄에 하나씩 들어가 있고 줄 번호를 해당 꽃 이름의 인덱스로 하여 학습을 진행한다.

        # flower_labels.txt
        daisy
        dandelion
        roses
        sunflowers
        tulips
        
 - 그리고 예측 후 출력하는 값은 다음과 같이 모든 인덱스에 대해 총합이 1인 확률을 나열한 softmax값이다.

       # daisy  dandelion roses   sunflowers  tulips
       [0.6509   0.1175   0.0147   0.2099,  0.0067]

3️⃣ 테스트 결과를 확인할 때 사용하기 위해 파일에 담긴 꽃 이름들을 가져와 배열로 저장해둔다. 이름을 출력하는 데 사용할 것이다.

    def main(_):
        labels = [line.srtrip() for line in tf.gfile.Gfie(FLAGS.output_labels)]
       
4️⃣ retrain.py를 실행해 학습이 끝나면 flowers_graph.pb파일이 생성된다. 학습 결과를 **프로토콜 버퍼**라는 데이터 형식으로 저장해둔 파일이다. 꽃 사진 예측을 위해 이 파일을 읽어 들여 신경망 그래프를 생성한다. 텐서플로를 이용하면 이 작업 역시 매우 쉽게 할 수 있다.

    with tf.gfile.FastGFile(FLAGS.output_graph, 'rb') as fp:
      graph_def = tf.GraphDef()
      graph_def.ParseFromString(fp.read())
      tf.import_graph_def(graph_Def, name='')
    
5️⃣ 읽어 들인 신경망 모델에서 예측에 사용할 텐서를 지정한다. 저장 되어 있는 모델에서 최종 출력층은 'final_result:0'이라는 이름의 텐서이다. 이 텐서를 가져와 예측에 사용한다.

    with tf.Session() as sess:  
        logits = sess.graph.get_tensor_by_name('final_result:0')
        
6️⃣ 그리고 예측 스크립트를 실행할 때 주어진 이름의 이미지 파일을 읽어 들인 뒤 그 이미지를 예측 모델에 넣어 예측을 실행한다. 다음 코드의 'DecodeJpeg/contents:0'은 이미지 데이터를 입력값으로 넣을 플레이스 홀더의 이름이다.

    image = tf.gfile.FastGFile(sys.argv[1], 'rb')read()
    prediction = sess.run(logits, {DecodeJpeg/contents:0' :image})
    
  - 프로토콜 버퍼 형식으로 저장되어 있는 다른 모델들도 이와 같은 방법으로 쉽게 읽어 들여 사용할 수 있다.

7️⃣ 이제 예측 결과를 출력하는 코드를 작성해본다. 다음 코드로 앞에서 읽어온 꽃 이름(레이블)에 해당하는 모든 예측 결과를 출력한다.

    print('=== 예측 결과 ===')
    for i in range(len(labels)):
        name = labels[i]
        score = prediction[0][i]
        print('%s (%.2f%%)' % (name, score * 100))
        
8️⃣ 다음 코드는 주어진 이름의 이미지 파일을 matplotlib모듈을 이용해 출력한다.

    if FLAGS.show_image :
        img = mping.imread(sys.argv[1])
        plt.imshow(img)
        plt.show()
        
9️⃣ 마지막으로 스크립트 실행 시 주어진 옵션들과 함께 main()함수를 실행하는 코드를 작성해준다.

    if __name__ == "__main__":
        tf.app.run()
        
실행은 터미널이나 명령 프롬프트를 연 뒤 predict.py 스크립트 파일이 있는 곳에서 다음 명령을 실행하면 된다.

    C:\> python predict.py workspace/flower_photos.roses.3065719996-c16ecd5551.jpg
    
예측 결과는 매우 정확하게 나왔다. 

💡 이처럼 구글이 공개한 인셉션 모델과 스크립트를 사용하면 매우 뛰어난 수준의 이미지 분류 프로그램을 아주 쉽게 만들 수 있다.

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    









