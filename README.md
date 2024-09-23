# 🪶 팀원 소개 
**SKN03-3rd-5Team**
### 🕊️early bird🕊️

| <img src="https://github.com/user-attachments/assets/babc6853-a31b-43ca-833c-15240486796c" alt="김성은" width="1800" /> | <img src="https://github.com/user-attachments/assets/3f9a9d23-16da-4a30-8825-39d2986b9a2a" alt="김지훈" width="2000" /> | <img src="https://github.com/user-attachments/assets/e4171517-6554-42ca-8f6c-8c3603a24a10" alt="송영빈" width="2000" /> | <img src="https://github.com/user-attachments/assets/65a67801-0ea0-429c-94cf-5844047bcfa8" alt="김재성" width="2000" /> | <img src="https://github.com/user-attachments/assets/c8931a7a-f025-4b4c-90d0-b894f3fa34ae" alt="박규택" width="2000" /> | 
|:----------:|:----------:|:----------:|:----------:|:----------:|
| 김성은 (PM) | 김지훈 | 송영빈 | 김재성 | 박규택 |   
| Data Analysis<br>Data-preprocessing<br>모델 학습<br>(Logistic Regression)<br>프론트<br>Result Comfirm | Data Analysis<br>Data-preprocessing<br>모델 학습<br>(Random Forest Classification)<br>프론트<br>Label Flipping | Data Analysis<br>모델 학습<br>(Random Forest Classification)<br>Analystic Arrange | Data Analysis<br>모델 학습<br>(Logistic Regression)<br>Analystic Arrange | Data Analysis<br>모델 학습<br>(Logistic Regression)<br>(XGBoost)<br>(LightGBM) |   
# :computer: 프로젝트 개요
- **프로젝트 목표 및 목적**  
1. 통신사 가입 고객 데이터를 분석하여 이탈 고객 예측 모델 개발   
2. 고객 이탈 대상 확인 후 해당 대상들의 특성을 파악하여 레이블 플리핑을 활용한 이탈 고객 예측 모델의 근거 확립  
###    
- **레이블 플리핑**
  데이터셋에서 일부 샘플의 레이블(타겟 값)을 변경하는 것을 말합니다. 예를 들어, 레이블이 0인 샘플을 1로 바꾸거나, 그 반대로 바꾸는것

- **왜 레이블 플리핑을 사용하는가?**
당신이 수행한 작업은 FP를 TN으로 변경하기 위해 레이블 플리핑을 사용한 것입니다. 이는 다음과 같은 목적을 가지고 있다.

오류 분석: 모델이 왜 FP 오류를 내는지 이해하기 위해 해당 샘플들의 레이블을 변경하고 모델의 반응을 관찰한다.
특징 분석: FP로 분류된 샘플의 특징(feature)들을 살펴보고, 어떤 특징들이 모델의 예측에 영향을 미치는지 파악한다.
모델 개선: 이러한 분석을 통해 모델의 성능을 향상시킬 수 있는 인사이트를 얻는다.

- **프로젝트 요구사항**   
1. 가입 고객 이탈 예측 모델 개발   
2. 장고를 이용한 예측 결과 화면 개발   
###   
- **가설**   
1) 가입 고객 이탈 예측 모델을 사용하여 confusion matrix를 만들었을 때, 실제로 고객이 이탈하지 않았지만 예측 모델이 이탈이라고 예측하는 FP(False Positive)의 데이터를 확인한다.
   위의 confusion matrix의 threshold 값의 변화로 도출되는 FP의 결과가 이탈 확률이 높은 고객과 낮은 고객을 구분하는 기준이 될 수 있다.
<img src="https://github.com/user-attachments/assets/a6785af6-b478-40c7-a2f8-fd5761dd87dd" alt="김성은" width="700" />
</br></br>
2) 해당 데이터를 트리 기반 모델을 사용하여 이탈할 확률이 높게 예측된 이유 확인
</br></br>
<img src="https://github.com/user-attachments/assets/dde29d9a-890c-4903-8eec-3a55519b68e7" alt="김성은" width="700" />
</br></br>

- **결론**   
모델 결과값의 FP(False Positive) 데이터의 값이 오류가 아닌 가능성 있는 데이터의 값으로 이해 할 수 있다.

###    
- **트리기반 모델을 사용하는 이유?**    
1) Tree모델을 사용하면 결과값에 대한 이유를 어느정도 이해할 수 있다.   
2) 데이터를 분할하여 예측을 수행하기 때문에 각 예측이 어떻게 이루어졌는지에 대한 설명이 가능하다.   
3) 추가로 트리 시각화를 통해 각 데이터 분할 과정과 최종 결정이 어떻게 이루어졌는지 시각적으로 이해할 수 있다.
###   
</br></br>
- **아키텍처**
<img src="https://github.com/user-attachments/assets/1b4f6470-4e39-438b-8029-bccb8986fa64" alt="김성은" width="800" />
</br></br>

- **workflow**   
<img src="https://github.com/user-attachments/assets/847a5338-366c-4ccc-9995-592c18689c6d" alt="김성은" width="800" />


# 📁데이터   
- **데이터 전처리 및 인코딩**   
![5678](https://github.com/user-attachments/assets/7710c89a-9bcc-4511-9be1-6980f9a09fe4)

  - costomerID 컬럼 삭제   
  - 범주형 데이터를 숫자형으로 인코딩
  - 결측치 제거

# 📊분석방법   
- **사용모델**   
logistic regression / Random Forest Classfication / confusion matrix / Matplotlib
### 
- **분석 과정 설명**

1) logistic regression
###   
   - 가입 고객
   ![logistic regression](https://github.com/user-attachments/assets/4440bdba-29ab-43f5-8115-b80fe40e6125)   
###   
   - 이탈 고객  
   ![logistic regression](https://github.com/user-attachments/assets/eb47f65f-e070-49be-b8d5-a93237a9da6d)
###   
   - 이탈 예측 고객
   ![logistic regression](https://github.com/user-attachments/assets/d9ed339c-dc74-4213-b3b9-aa183512b38c)

      - 이탈 고객층과 가입 고객층의 결혼 유무와 부양가족 여부가 지속적인 가입에 영향을 줄 수 있다.
      - 이탈 고객층의 경우에 서비스 전반의 참여가 저조하다.
      - 이탈 고객층은 대개 1~2 개월의 가입기간을 가지고 구독을 갱신하지 않는 것으로 보인다.
         - 이탈 고객층 중 비정상적으로 Total Charge가 높은 사람들이 있으며, 해당 고객은 필요할 때만 구독을 하고 그 외엔 구독을 갱신하지 않는 것으로 판단된다.
      - 이탈 고객층의 대부분의 결제 방법이 Electronic check인 것으로 보인다.
      - 이탈 고객층의 월간 지출 금액이 장기 고객층의 지출 금액과 비교하여 상대적으로 높은 것을 확인할 수 있다.
      - 이탈 여부를 확률적으로 분석하였을 때, 장기 고객의 경우 0(고객이라 판단)에 분포가 몰려있는 것을 확인할 수 있다.
      - 이탈 여부를 확률적으로 분석하였을 때, 이탈 고객의 경우 1(고객이 아니라 판단)로 확실하게 판단한 케이스는 없다.
         - 이는 확실하게 이탈하는 사람의 패턴이 없다는 것과 필요할 때 구독을 하는 사람으로 인해 결과에 혼동을 준다고 판단할 수 있다.
      - 따라서, 위의 결과를 통해서 이탈예측고객의 그래프가 이탈고객의 그래프와 유사함을 확인 

2) Random Forest Classfication   
   ![Random Forest Classfication](https://github.com/user-attachments/assets/69e230d8-540a-4fee-bdef-282408966495)   
   - 실제 결과값은 이탈하지 않았지만 모델이 이탈한다고 예측한 고객(FP)의 실제 결과값을 이탈 고객(Target: 0→ 1)으로  변환한다.
   - 변환한 데이터를 포함한 전체 데이터를 트리 모델로 학습한다.
   - 처음에 결과값을 변환했던 FP 데이터의 노드를 확인한다.
   - 트리가 분할한 컬럼의 값을 하나씩 바꿔보면서 하나의 컬럼으로 인해 결과값이 바뀌는 데이터를 확인한다.
   - 해당 컬럼을 기준으로 솔루션을 기획하면 고객 맞춤형 솔루션을 제공할 수 있다.



###   
# 🚩결과 및 해석   
- **주요 발견사항**   
2년 계약이 고객 유지에 중요한 요인으로 확인된다.     
부양가족이 있는 경우 고객 유지율이 높은 것으로 확인된다.   
월 요금, 지불 방법, 계약 유형 등이 이탈에 영향을 미치는 것으로 나타난다.   
스트리밍 서비스 이용 여부도 고객 유지에 영향을 준다.   
기술 지원을 받지 않고 장기 계약(2년)이 아닌 고객들이 이탈할 가능성이 더 높아 보인다.   

###   
- ** **    
 
###   
# ⚠오류 해결 과정   
###   
# 📌느낀점   
김성은 : 처음 Logistic으로 할 계획이었는데 다들 자발적으로 다른 모델로 분석을 해주셔서 다양한 관점으로 분석할 수 있었습니다. 
다들 적극적인 의견 제시와 제안한 의견 모두 반영하기 위해 노력을 해주셔서 프로젝트 운영이 수월했습니다. 
수고 많으셨습니다~   

김지훈 :    

김재성 : 능력좋은 팀원들을 만나서 배울것도 많았고 시간적으로 공부 할 여유도 생겨 많이 배울 수 있었습니다. 감사합니다.   

박규택 : 실 다른모델로 한게....기존 성은님 모델로 어떻게 수정 해야 할지 몰라서 예전 강사님 모델 가져다 쓴건데...다행이 좋은쪽으로 적용된거 같습니다 ㅎㅎ
처음에 혼동행렬 아이디어 이후에....혼동행렬의 HPO 조절만 한거 같은데, 다른 분들이 거의 다 해주셔서 편하게 맡은 일부 부분에 집중할수 있었던거 같습니다.   
       
송영빈 : 다양한 방법으로 데이터를 분석하는 경험을 할 수 있었고 적절한 데이터 시각화가 굉장히 중요하다고 느꼈다.   
###   
# 📌기술스택   
<div style="display: flex; gap: 10px;">
<h3>Language</h3>
  <img src="https://img.shields.io/badge/python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" style="height: 40px; object-fit: contain;">
 
</div>    

      
<div style="display: flex; gap: 10px;">
<h3>Frontend</h3>
  <img src="https://img.shields.io/badge/django-092E20?style=for-the-badge&logo=django&logoColor=white" alt="django" style="height: 40px; object-fit: contain;">
  <img src="https://img.shields.io/badge/css-1572B6?style=for-the-badge&logo=css3&logoColor=white" alt="CSS" style="height: 40px; object-fit: contain;">
  <img src="https://img.shields.io/badge/javascript-F7DF1E?style=for-the-badge&logo=javascript&logoColor=black" alt="JavaScript" style="height: 40px; object-fit: contain;">
  <img src="https://img.shields.io/badge/html5-E34F26?style=for-the-badge&logo=html5&logoColor=white"" alt="html" style="height: 40px; object-fit: contain;">
</div>
