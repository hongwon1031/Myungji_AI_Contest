# AI를 활용한 Warpage 발생 기판의 형태 분류 및 개선안 도출
## 프로젝트 개요
본 프로젝트는 AI 기술을 활용하여 PCB(인쇄회로기판)의 Warpage 발생을 분석하고, 이를 기반으로 최적의 공정 조건을 도출함으로써 생산 수율을 향상시키는 것을 목표로 합니다. Warpage는 제조 과정 중 PCB 기판에 직접적으로 열이 가해질 때 발생할 수 있는데, 이 현상은 기판의 평탄도를 저하시켜 제품의 신뢰성을 떨어뜨릴 수 있습니다.
## 문제 정의
PCB 기판은 Reflow 공정에서 열을 받으면 Warpage 현상이 발생할 수 있습니다. 이 현상의 정도가 산업 기준을 초과할 경우 기판이 사용 불가능해지며, 이로 인해 생산 수율이 감소할 위험이 있습니다. 따라서, Warpage를 사전에 예측하고 이를 효과적으로 개선할 수 있는 방법을 모색하는 것이 필수적입니다.
# 프로젝트 구성
## 데이터셋
PCB 기판 Dataset: Ansys workbench와 python을 사용하여 시뮬레이션 데이터를 생성하며, 이 데이터의 정확성과 일관성을 전제로 합니다.
## 모델 구현
- CNN (Convolutional Neural Network): PCB 기판의 Warpage 형태를 분석하고 분류하는데 사용됩니다.
- LSTM (Long Short-Term Memory): 시계열 데이터를 바탕으로 최적의 공정 조건을 도출하는데 사용됩니다. 이를 통해 Reflow 공정 시간을 최적화하고, 다양한 공정 조건에 대한 시뮬레이션을 통해 개선안을 도출합니다.
## 주요 결과
모델 학습 및 성능 평가 결과, 트레이닝 정확도와 밸리데이션 정확도가 각각 99.4%, 99.87%로 높은 정확도를 달성하였습니다.
LSTM 모델을 통해 시간에 따른 변위값을 성공적으로 예측하여 공정 최적화에 기여하였습니다.
## 사용 기술
Programming Languages: Python
Tools & Libraries: Ansys, Tensorflow, Keras, PyTorch
Data Handling: JSON, PNG output
## 성능평가
자세한 내용은 [결과보고서.pdf](https://github.com/hongwon1031/Myungji_AI_Contest/blob/main/%EA%B2%B0%EA%B3%BC%EB%B3%B4%EA%B3%A0%EC%84%9C.pdf) 참조
