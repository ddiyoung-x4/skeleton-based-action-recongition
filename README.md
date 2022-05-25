# skeleton-based-action-recongition
This is 2021-fall self-directed research course.

you can click on the presentation of research(korean language).
https://drive.google.com/file/d/1e7rfCIGqoo2gxP-f360igyMGGzbiNeJ3/view?usp=sharing


### 1. 연구 개요  
- 연구 배경  
컴퓨터 비전은 시각적 세계를 해석하고 이해하도록 하는 기계를 학습시키는 분야로, 카메라와 동영상의 디지털 이미지에서 특정 객체를 식별하고 이에 대한 반응을 한다. 특별 객체에 대한 대상은 현실 세계에 존재하는 모든 것들이 가능하지만, 그 중 가장 중요한 정보를 나타내는 것은 역시 사람이다.  
**사람의 행동 인식**은 인간 관계에서의 상호 작용, 혹은 인간 대 컴퓨터 상호작용 등 필요에 따라 다양한 어플리케이션으로 이용될 수 있다. 비디오 감시, 피트니스, 의료, HCI 등 앞으로 다방면에서 활용이 가능할 것으로 전망된다. 해당 인공지능 연구를 통해서 누군가를 폭행하거나 수상한 행동을 하는 경우에 영상인식을 통해 기계가 스스로 이상 행동에 대한 경보를 울리거나, 특정 인물들의 행동 패턴을 분석하는 등 다양한 application으로 확장될 수 있다.  
본 연구는 이러한 흐름을 바탕으로 **pose estimation**를 접목한 **action recognition** 행동 인식을 진행하고자 한다.  

- 연구 목표  
본 연구는 **HMDB51 dataset**을 사용하여 **pose estimation으로 추출된 신체의 특징점들**을 **딥러닝 모델의 input features로 설정**하여 사람이 현재 어떠한 행동을 하고 있는 지를 판단하는 **행동 인식 인공지능을 탐색하는 것을 목표**로 한다. 또한, 사람의 행동 인식을 진행하는 과정에서 2가지 방법이 존재하는데, 각각의 frame마다 행동을 분류하는 방법과 여러 개의 sequential frame을 하나의 행동으로 분류하는 방법이 있다. **2가지 모델을 설계**하고 각각의 성능을 **비교 분석하여 더욱 효과적인 방법이 어떤 것인가**를 도출한다.  

### 2. 선행 연구 조사  
- MultiPoseNet: Fast Multi-Person Pose Estimation using Pose Residual Network  
- RMPE: Regional Multi-person Pose Estimation  
### 3. 연구 진행
- 행동 인식 모델  
본 연구는 간단한 행동들을 기준으로 **인식 정확도 성능이 뛰어나며 빠른 추론 속도를 갖는 모델을 탐색**하는 것이다. 기존 연구들은 다양한 행동들을 탐지하는데 있어 뛰어난 성능을 갖고 있음에도 3D heatmap stack를 사용하며 추가적인 computation cost를 사용하기 때문에 무거운 모델들을 사용하고 있다. 이에 따라 60개의 많은 행동들을 분류할 수는 있지만 어플리케이션으로 활용하기에는 너무 무겁고 실제로 이렇게 많은 행동을 한 번에 분류하기 보다 그 일부분의 행동들만 사용하여 활용될 가능성이 높다. 따라서, **빠른 추론 속도를 위해 light model을 탐색하고 소수의 행동들(kick, punch, stand, squat, wave)에 최적화된 모델을 연구**한다.  
  
**Pipeline**  
1. 기존 Pose Estimation 연구들을 기반으로, 특정 features(keypoints) 추출  
2. 추출된 features 기반으로 action recognition 딥러닝 모델 연구  
  
<img width="70%" src="https://user-images.githubusercontent.com/69739208/170240530-e29b3ed2-c355-4694-8fad-ea6d167bd62e.png"/>  
  
  
### 4. 연구 결과
<img width="70%" src="https://user-images.githubusercontent.com/69739208/170240774-f10110c4-266b-453d-ae2c-94d7aa190012.png"/>  
<img width="60%" src="https://user-images.githubusercontent.com/69739208/170240988-b918ff85-c092-4cee-90d7-7fc27a359ea9.png"/>  

<img width="70%" src="https://user-images.githubusercontent.com/69739208/170241264-f45c7d69-852b-4744-ba96-955c7412dcc2.png"/>  
<img width="60%" src="https://user-images.githubusercontent.com/69739208/170241544-d308b683-3d1d-4542-959a-bf28ab86d671.png"/>  
  
- 1D CNNs model  
One Dimension CNNs model은 **여러 개의 frame을 하나의 행동이라 판단하는 모델**로서, 행동이란 여러 동작들의 흐름으로 이루어져 있다는 아이디어에서 고안해보았다. Sequential한 특징을 가지고 있는 데이터에서 1D CNNs 구조가 상당히 효과적인 사실에 근거하여 모델을 연구하였다. 최종적인 모델은 위의 그림과 같으며 모델에 대한 간략한 설명을 진행한다. 5개의 frame을 하나의 행동이라 판단하고 총 26개의 points가 존재하기 때문에, (5 x 1)에 26개의 channel를 가지는 feature가 하나의 input으로 들어온다. 각 points마다 kernel size가 2이고 64개의 filters를 거쳐서 인접한 동작들의 관계성을 찾는다. 이후, MaxPooling을 통해 spatial size를 한번 줄여준 후 Dense layer를 거치며 output layer에서 softmax를 통해 각 동작들에 대한 확률 값을 통해 classification이 이루어진다.  
  
- 결과 분석  
본 연구 문제 정의에서 언급한 가벼운 모델에서도 높은 인식 정확도를 가지며 소수의 행동들에 최적화된 모델을 설계할 수 있었다. 또한, 두 모델에 대한 성능을 비교해봤을 때, **1D convolution**을 사용하여 **골격의 연속적인 움직임에 대한 연관성을 파악하는 것**이 매 frame마다 동작을 보고 이에 대한 행동을 추론하는 것보다 더 좋은 성능을 보인다는 것을 알 수 있다.  
  
  
### 5. 결론  
본 연구의 목표는 pose estimation으로 추출된 신체의 특징점들을 딥러닝 모델의 input features로 설정하여 사람이 현재 어떠한 행동을 하고 있는 지를 판단하는 행동 인식 인공지능을 탐색하는 것이다. 또한, 각각의 frame마다 행동을 분류하는 방법과 여러 개의 sequential frame을 하나의 행동으로 분류하는 방법이 존재하는데, 각각의 성능을 비교 분석 연구를 진행하며 어떤 모델이 더욱 뛰어난 성능을 보이는 지를 탐색하고자 했다.  
해당 연구를 위하여 2가지 모델을 설계하고 비교 분석을 진행하였다. **각각의 frame마다 행동을 분류하는 모델의 경우 DNN model**을 설계하여 진행해보았고, **sequential frame마다 행동을 분류하는 모델의 경우 1D-CNNs model**을 설계하였다. 해당 모델들에 대한 성능 평가와 비교 분석을 마친 후, 행동은 연속된 동작들의 흐름으로 이루어진 행위이기 때문에 두번째 모델이 더욱 정확한 성능을 보인다는 결과를 도출해냈다.   
기존의 연구들은 60개 이상의 다양한 행동들을 분류할 수 있는 모델을 만들기 위해서 3D heatmap stack를 사용하며 이에 따라 높은 computation cost를 사용하기 때문에 정확도는 뛰어나지만 속도 측면을 보장하진 않았다. 따라서, 본 연구는 실생활에서 60개의 행동들을 분류하는 어플리케이션은 너무 무거우며 실제로 모든 행동들을 인식하기 보다 특정 도메인에 맞는 행동들만 인식하는 것이 더욱 효과적이라 판단하여 **소수의 행동들을 인식할 수 있는 모델**을 설계하였다. 또한, frame마다 동작을 탐지하는 것보다 **연속된 frame들을 살펴보고 동작을 탐지하는 것이 더욱 효과적**임을 도출하였다.  
**본 연구를 통해 향후** 행동을 추가하거나 다른 행동으로 변경해보았을 때도 기존과 같은 높은 성능을 보이는 가에 대해 연구해볼 수 있다. 또한, 실생활에서 사용될 수 있는 가능성을 제시했기 때문에 **CCTV를 통한 이상 행동 탐지**, Anomaly Detection과 같은 **Video application**을 연구할 수 있다.   
