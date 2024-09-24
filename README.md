# 🪶 팀원 소개 
**SKN03-3rd-5Team**
### 🕊️early bird🕊️

| <img src="https://github.com/user-attachments/assets/babc6853-a31b-43ca-833c-15240486796c" alt="김성은" width="1800" /> | <img src="https://github.com/user-attachments/assets/3f9a9d23-16da-4a30-8825-39d2986b9a2a" alt="김지훈" width="2000" /> | <img src="https://github.com/user-attachments/assets/e4171517-6554-42ca-8f6c-8c3603a24a10" alt="송영빈" width="2000" /> | <img src="https://github.com/user-attachments/assets/65a67801-0ea0-429c-94cf-5844047bcfa8" alt="김재성" width="2000" /> | <img src="https://github.com/user-attachments/assets/c8931a7a-f025-4b4c-90d0-b894f3fa34ae" alt="박규택" width="2000" /> | 
|:----------:|:----------:|:----------:|:----------:|:----------:|
| 김성은 (PM) | 김지훈 | 송영빈 | 김재성 | 박규택 |   
| Data Analysis<br>Data-preprocessing<br>모델 학습<br>(Logistic Regression)<br>프론트<br>Result Comfirm | Data Analysis<br>Data-preprocessing<br>모델 학습<br>(Random Forest Classification)<br>프론트<br>Label Flipping | Data Analysis<br>모델 학습<br>(Random Forest Classification)<br>Analystic Arrange | Data Analysis<br>모델 학습<br>(Logistic Regression)<br>Analystic Arrange | Data Analysis<br>모델 학습<br>(Logistic Regression)<br>(XGBoost)<br>(LightGBM) |   
# :computer: 프로젝트 개요
- **프로젝트 목적**  
통신사 가입 고객 데이터를 분석하여 이탈 고객 예측 모델 개발   

###    
- **프로젝트 요구사항**   
1. 가입 고객 이탈 예측 모델 개발   
2. 장고를 이용한 예측 결과 화면 개발
3. 관리자 페이지 개발

###   
</br>

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
logistic regression  
###   
   - 가입 고객   
   ![tn](https://github.com/user-attachments/assets/5214c445-5895-48eb-b18c-8dce37293f80)   

###    
   - 이탈 고객
   ![tp](https://github.com/user-attachments/assets/9d43cdda-8505-4561-84e5-eac0ff324031)

###    
   - 이탈 예측 고객    
   ![fp](https://github.com/user-attachments/assets/b4f0655d-1808-4bb0-a538-d641696e25ea)

      - 이탈 고객층과 가입 고객층의 결혼 유무와 부양가족 여부가 지속적인 가입에 영향을 줄 수 있다.   
      - 이탈 고객층의 경우에 서비스 전반의 참여가 저조하다.   
      - 이탈 고객층은 대개 1~2 개월의 가입기간을 가지고 구독을 갱신하지 않는 것으로 보인다.   
         - 이탈 고객층 중 비정상적으로 Total Charge가 높은 사람들이 있으며, 해당 고객은 필요할 때만 구독을 하고 그 외엔 구독을 갱신하지 않는 것으로 판단된다.   
      - 이탈 고객층의 대부분의 결제 방법이 Electronic check인 것으로 보인다.   
      - 이탈 고객층의 월간 지출 금액이 장기 고객층의 지출 금액과 비교하여 상대적으로 높은 것을 확인할 수 있다.   
      - 이탈 여부를 확률적으로 분석하였을 때, 장기 고객의 경우 0(고객이라 판단)에 분포가 몰려있는 것을 확인할 수 있다.   
      - 이탈 여부를 확률적으로 분석하였을 때, 이탈 고객의 경우 1(고객이 아니라 판단)로 확실하게 판단한 케이스는 없다.   
         - 이는 확실하게 이탈하는 사람의 패턴이 없다는 것과 필요할 때 구독을 하는 사람으로 인해 결과에 혼동을 준다고 판단할 수 있다.   
      - 위에서 확인한 이탈 고객의 경향을 기준으로, 예측 모델이 이탈고객으로 파악한 고객의 경향은 이탈고객의 경향과 유사함을 확인할 수 있다.    
# 관리자 페이지 개발
ML 모델을 사용한 데이터 분석 외에 추가로 더 얻을 수 있는 정보나 실제 서비스에 활용할 수 있는 방안이 있을지 고민

## 모델의 이탈 예측 고객들의 레이블 플리핑을 활용하여 각 고객별 특성 파악
###    
- **레이블 플리핑**   
  데이터셋에서 일부 샘플의 레이블(타겟 값)을 변경하는 것을 말한다.   
  예를 들어, 레이블이 0인 샘플을 1로 바꾸거나, 그 반대로 바꾸는 것을 말한다.      
###    
- **왜 레이블 플리핑을 사용하는가?**   
  오류 분석: 모델이 왜 FP 오류를 내는지 이해하기 위해 해당 샘플들의 레이블을 변경하고 모델의 반응을 관찰한다.   
  특징 분석: FP로 분류된 샘플의 특징(feature)들을 살펴보고, 어떤 특징들이 모델의 예측에 영향을 미치는지 파악한다.   
  모델 개선: 이러한 분석을 통해 모델의 성능을 향상시킬 수 있는 인사이트를 얻는다.
   
###    
## 모델의 FP 데이터의 재해석, 잘못 예측한 것이 아닌 이탈 가능성이 있거나 높은 고객으로 판단, 해당 고객이 어떤 이유로 모델이 이탈 했다고 예측한 것인지 그 이유를 탐색
1. 가입 고객 이탈 예측 모델을 사용하여 confusion matrix를 만들었을 때, 실제로 고객이 이탈하지 않았지만 예측 모델이 이탈이라고 예측하는 FP(False Positive)의 데이터를 확인한다. 

![image](https://github.com/user-attachments/assets/46d976d5-8bb6-4025-8206-a4f8a2133224)
![result1-1](https://github.com/user-attachments/assets/d067c488-8ba5-42d9-963f-def05f0a3a0d)
![result1-2](https://github.com/user-attachments/assets/34a437ba-f20e-4d86-81a8-62ad1ff1c4ac)


FN 데이터에 대해서는 실제 이탈 고객이지만 가입고객으로 예측한 데이터에 대해서는 하나의 이상치라고 해석, threshold, 임의의 비용 함수를 사용하여 FP 의 비율을 높이고 FN 의 비율을 낮춰간다

2. 레이블 플러핑 시행, FP 데이터의 타겟을 Positive로, FN 를 Negative 로 변경해준다.
![result2](https://github.com/user-attachments/assets/5e2631f6-9d9b-4651-9c40-38465c087e98)

3. 바꾼 데이터를 트리 기반 모델로 재학습, 최대한 과대적합을 수행
</br></br>
<img src="https://github.com/user-attachments/assets/dde29d9a-890c-4903-8eec-3a55519b68e7" alt="김성은" width="700" />
</br></br>

4. 과대적합된 트리 모델의 생성, 데이터 별로 임의의 노드 경로를 바꿨을 때 반대의 타겟값이 나오게 되는 노드의 탐색
![image](https://github.com/user-attachments/assets/0cfb83d1-2e93-45d4-992b-94e0b8f22440)




###   
# 🚩결과 및 해석    
- **주요 발견사항**    
단기 고객으로 필요한 경우에만 구독을 하고 연장 갱신을 하지 않는 고객이 있다.
장기간 구독을 유지하는 고객은 2년 계약을 한 경우가 많으며, 이는 장기 고객 유지에 중요한 요인으로 확인된다.     
부양가족이 있는 경우 구독 유지율이 높은 것으로 확인된다.   
월 요금, 지불 방법, 계약 유형 등이 이탈에 영향을 미칠 수 있으나, 장기 구독 고객에서는 유의미한 패턴은 나타나지 않는다.   

###   

- **해석**



 
###   
# ⚠오류 해결 과정    
1. for문 반복 관련 오류    
   원인: 가중치 도입 실수(혼동행렬에 그냥 곱하기)   
   해결: sample_weights 따로 적용   
###    
2. for문 반복 관련 오류2   
   for문을 통해 기준값(trashold)과 가중치(weights) 동시 적용시 학습 문제 trashold 와 weights 중에서 1가지만 학습이 진행됨   
   오류가 되는 부분은 확인- 정확한 원인은 파악 불가 trashold만 정상작동 : All_y_pred = model.predict_proba(X)[:,1] weights만 정상작동 : y_pred_series = pd.Series(All_y_pred)   
   해결: 각각 한번씩 반복하며 학습을 진행함   
###   
3. 소수점 오류   
   그래프 title에 보이는 소수값이 틔는 현상이 존재   
   해결 : " *0.1 " => " /10 " 변경   
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
