# 🪶 팀원 소개 
**SKN03-3rd-5Team**
### 🕊️early bird🕊️

| <img src="https://github.com/user-attachments/assets/babc6853-a31b-43ca-833c-15240486796c" alt="김성은" width="1800" /> | <img src="https://github.com/user-attachments/assets/3f9a9d23-16da-4a30-8825-39d2986b9a2a" alt="김지훈" width="2000" /> | <img src="https://github.com/user-attachments/assets/e4171517-6554-42ca-8f6c-8c3603a24a10" alt="송영빈" width="2000" /> | <img src="https://github.com/user-attachments/assets/65a67801-0ea0-429c-94cf-5844047bcfa8" alt="김재성" width="2000" /> | <img src="https://github.com/user-attachments/assets/c8931a7a-f025-4b4c-90d0-b894f3fa34ae" alt="박규택" width="2000" /> | 
|:----------:|:----------:|:----------:|:----------:|:----------:|
| 김성은 (PM) | 김지훈 | 송영빈 | 김재성 | 박규택 |   

# :computer: 프로젝트 개요
- **프로젝트 목표 및 목적**  
1. 통신사 가입 고객 데이터를 분석하여 이탈 고객 예측 모델 개발   
2. 고객 이탈 대상 확인 후 해당 대상들의 특성을 파악하여 이탈 방지 솔루션 제공   
###    

- **프로젝트 요구사항**   
1. 가입 고객 이탈 예측 모델 개발   
2. 장고를 이용한 예측 결과 화면 개발   
3. 예측 가능한 이탈 고객 방지 대안 솔루션   
###   
- **가설**   
1) 
   가입 고객 이탈 예측 모델을 사용하여 confusion matrix를 만들었을 때 모델이 이탈하지 않았지만 이탈한다고 예측한 고객 FP(False Positive)의 데이터를 threshold을 사용,
   변화를 분석하면서 이탈 할 확률이 높은 고객과 낮은 고객을 구분 할 수 있을 것이다.  
<img src="https://github.com/user-attachments/assets/a6785af6-b478-40c7-a2f8-fd5761dd87dd" alt="김성은" width="700" />

2) 해당 데이터를 트리 기반 모델을 사용하여 이탈할 확률이 높게 예측된 이유 확인  
<img src="https://github.com/user-attachments/assets/c2ad9567-1010-4f97-928f-7f2c262e816b" alt="김성은" width="700" />

- **결론**   
이를 통해 이탈 확률이 높은 고객들의 특징을 분석하고 이탈을 방지하는 솔루션을 제공할 수 있을 것이다.   

###    
- **트리기반 모델을 사용하는 이유?**    
1) Tree모델을 사용하면 결과값에 대한 이유를 어느정도 이해할 수 있다.   
2) 데이터를 분할하여 예측을 수행하기 때문에 각 예측이 어떻게 이루어졌는지에 대한 설명이 가능하다.   
3) 추가로 트리 시각화를 통해 각 데이터 분할 과정과 최종 결정이 어떻게 이루어졌는지 시각적으로 이해할 수 있다.
###   

- 아키텍처
<img src="https://github.com/user-attachments/assets/1b4f6470-4e39-438b-8029-bccb8986fa64" alt="김성은" width="800" />

- workflow   
<img src="https://github.com/user-attachments/assets/847a5338-366c-4ccc-9995-592c18689c6d" alt="김성은" width="800" />


# 📁데이터   
- **데이터 전처리 및 인코딩**   
![image](https://github.com/user-attachments/assets/3a95958c-cf5c-4a29-81ae-757f52c9418c)
![image](https://github.com/user-attachments/assets/f5060d2f-35fa-4fac-9e10-18c3d7110115)

  - costomerID 컬럼 삭제   
  - 범주형 데이터를 숫자형으로 인코딩   

# 📊분석방법   
- **사용한 기법 및 알고리즘**   
logistic regression / Random Forest Classfication / confusion matrix 
### 
- **분석 과정 설명**   
1) logistic regression 
###   
- 이탈 고객  
![logistic regression](https://github.com/user-attachments/assets/eb47f65f-e070-49be-b8d5-a93237a9da6d)   
###   
- 가입 고객
![logistic regression](https://github.com/user-attachments/assets/4440bdba-29ab-43f5-8115-b80fe40e6125)   
###   
- 이탈 예측 고객
![logistic regression](https://github.com/user-attachments/assets/d9ed339c-dc74-4213-b3b9-aa183512b38c)   
그래프를 통해서 이탈고객과 이탈예측고객의 그래프가 유사함을 확인 

3) Random Forest Classfication   
![Random Forest Classfication](https://github.com/user-attachments/assets/69e230d8-540a-4fee-bdef-282408966495)   



###   
# 🚩결과 및 해석   
- **주요 발견사항**   
2년 계약이 고객 유지에 중요한 요인으로 보임
부양가족이 있는 경우 고객 유지율이 높아 보임
월 요금, 지불 방법, 계약 유형 등이 이탈에 영향을 미치는 것으로 나타남
스트리밍 서비스 이용 여부도 고객 유지에 영향을 줌
기술 지원을 받지 않고 장기 계약(2년)이 아닌 고객들이 이탈할 가능성이 더 높아 보입니다.

###   
- **솔루션**   
첫 가입고객을 위한 첫 결제시 할인, 서비스 세분화
트리를 통해 기업은 고객 이탈 위험이 높은 세그먼트를 식별하고, 적절한 고객 유지 전략을 수립할 수 있습니다.
기술 지원 서비스 개선, 장기 계약 혜택 강화, 온라인 보안 서비스 홍보 등의 전략을 고려할 수 있습니다.
###   
# ⚠오류 해결 과정   
###   
# 📌느낀점   
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




   
  
