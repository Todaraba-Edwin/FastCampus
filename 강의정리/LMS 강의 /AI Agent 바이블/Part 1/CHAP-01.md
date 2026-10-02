[← Part 1](./PART-1-README.md)

# CHAP-01 현재 LLM의 근간, 트랜스포머 아키택처 알아보기

## 📑 Index
- [첫째, LLM(Large Language Model)](#첫째-llmlarge-language-model)
- [둘째, 2017년 "Attention is All You Need" 논문](#둘째-2017년-attention-is-all-you-need-논문)
- [셋째, Transformer 아키택처](#셋째-transformer-아키택처)
- [넷째, NLP의 역사](#넷째-nlp의-역사)
- [다섯째, Transformer의 원리는 무엇인가?](#다섯째-transformer의-원리는-무엇인가)
- [여섯째, LLM은 어떻게 만들어지는가?](#여섯째-llm은-어떻게-만들어지는가)

---

### 첫째, LLM(Large Language Model)
- 방대한 텍스트 데이터로 학습된 신경망 모델
- 자연어 이해와 생성 능력 보유
- 확률 기반으로 다음 토큰 예측

### 둘째, 2017년 "Attention is All You Need" 논문
- 논문명: "Attention is All You Need" (Vaswani et al., 2017)
- Google이 발표한 혁신적 논문
- Transformer 아키택처 최초 제안
- 이전의 RNN, LSTM의 한계점 극복

### 셋째, Transformer 아키택처
- Self-Attention 메커니즘 도입
  - 입력 시퀀스의 각 요소가 다른 요소들과의 관계 학습
  - 병렬 처리 가능 (RNN 대비 매우 빠름)
- Encoder-Decoder 구조
  - Encoder: 입력 데이터 처리
  - Decoder: 출력 데이터 생성
- Position Encoding으로 순서 정보 보존
- 현대 LLM의 기반이 되는 구조

### 넷째, NLP의 역사

#### 초기 규칙 기반
- 언어 규칙을 명시적으로 정의하여 처리
- 전문가가 직접 규칙 입력
- **단점:** 낮은 확장성, 높은 유지보수 비용, 새로운 언어/규칙 추가 어려움

#### RNN, LSTM (1997~ 2010년대)
- 순차 데이터 처리에 효과적
- 이전 정보를 기억하여 현재 입력 처리
- **단점:** 장기 의존성 문제 (Vanishing Gradient), 느린 병렬 처리 속도

#### Transformer (2017년~)
- Self-Attention으로 병렬 처리 가능
- 모든 단어 간 직접적인 관계 학습
- 더 긴 시퀀스 학습 가능
- 신경망 기반 NLP의 패러다임 전환

#### 사전학습 모델 (2018~2020년대)

##### BERT (Bidirectional Encoder Representations from Transformers)
- Transformer의 **인코더** 기반
- 양방향 텍스트 이해 (마스킹된 토큰 예측)
- 텍스트 분류, 감정 분석 등에 최적화
- 인코더만 사용하여 입력 표현 학습

##### GPT-2 (Generative Pre-trained Transformer 2)
- Transformer의 **디코더** 기반
- 단방향 텍스트 생성 (다음 토큰 예측)
- 자연스러운 텍스트 생성에 최적화
- 디코더만 사용하여 순차적 생성

##### 인코더/디코더 기반 확장
- **Encoder Only (BERT 류):** 텍스트 이해, 분류, 표현 학습
- **Decoder Only (GPT 류):** 텍스트 생성, 언어 모델링
- **Encoder-Decoder (T5):** 기계 번역, 요약, 질문-답변 (원본 Transformer 구조 유지)

#### 거대 멀티모달 모델 (2020년대~)
- GPT-4, Claude, Gemini 등 대규모 LLM
- 텍스트, 이미지, 음성 등 다중 모달리티 처리
- 일반화된 지능(AGI) 방향으로 진화
- **단점:** 높은 학습 비용, 환경오염 우려, 윤리적 이슈

### 다섯째, Transformer의 원리는 무엇인가?

- 인코더와 디코더 구조를 갖춘 딥러닝 아키텍처

#### 인코더 (Encoder)
- 입력 시퀀스를 받아 의미 있는 표현으로 변환
- Self-Attention으로 입력 내 모든 단어 간의 관계 파악
- 각 토큰의 위치 정보(Positional Encoding) 보존
- 다층 구조로 점진적으로 고수준 표현 생성
- 최종 출력: 전체 입력을 이해한 컨텍스트 벡터

#### 디코더 (Decoder)
- 인코더의 출력(컨텍스트)을 받아 출력 시퀀스 생성
- Masked Self-Attention으로 이전 출력만 참고 (미래 정보 차단)
- Cross-Attention으로 인코더 출력과 현재 디코더 상태 결합
- 한 번에 하나의 토큰씩 순차 생성
- 각 단계마다 확률 분포 계산하여 다음 토큰 선택

#### Transformer에서 Attention이 가지는 의미
- **"무엇에 집중할 것인가"를 학습하는 메커니즘**
  - 입력의 모든 요소에 동일 가중치가 아닌 선택적 가중치 부여
  - 관련 높은 정보에만 집중, 불필요한 정보 무시

- **병렬 처리 가능**
  - RNN처럼 순차 처리 불필요
  - 전체 시퀀스를 동시에 처리하여 학습 속도 대폭 향상

- **장기 의존성 해결**
  - 아무리 멀리 떨어진 단어도 직접 연결
  - Vanishing Gradient 문제 해결

- **다양한 관계 패턴 학습**
  - Multi-Head Attention으로 여러 시점에서 동시 분석
  - 문법, 의미, 문맥 등 다층적 특성 포착

- **모든 NLP 태스크의 기반**
  - 기계 번역, 텍스트 분류, 감정 분석, 질문 응답 등 모두에 적용
  - 현대 LLM의 핵심 메커니즘

#### 활용범위
- **번역** (Machine Translation)
  - 소스 언어 입력을 타겟 언어로 변환
  - 인코더가 원문 이해, 디코더가 번역문 생성

- **감정분석** (Sentiment Analysis)
  - 텍스트의 긍정/부정/중립 감정 판단
  - 인코더만 사용하여 입력 표현 학습 후 분류

- **언어(예측)생성** (Language Generation)
  - 주어진 컨텍스트로부터 자연스러운 텍스트 생성
  - 디코더만 사용하여 다음 토큰 순차 예측 (자동 완성, 챗봇 등)

### 여섯째, LLM은 어떻게 만들어지는가?

- **Transformer의 디코더 구조만 사용**
  - 인코더-디코더 구조 대신 디코더만 확장
  - Causal Attention으로 미래 토큰 마스킹 (현재와 이전 토큰만 참고)

- **다음 토큰 예측의 의미**
  - 훈련 데이터: 실제 존재하는 텍스트 (인터넷, 책, 코드 등)
  - 학습 목표: "앞의 N개 토큰이 주어졌을 때 다음 토큰이 뭘까?" 통계적 패턴 학습
  - 가중치 업데이트: 실제 다음 토큰 vs 모델 예측값 비교하여 조정
  - **결과:** 훈련 데이터의 확률 분포 학습 (새로운 텍스트 생성 아님)

- **훈련 vs 추론의 차이 (Hallucination의 원인)**
  - **훈련 단계 (Teacher Forcing)**
    - 실제 정답 토큰 사용: "The quick brown" → 다음은 "fox"
    - 손실함수로 오차 계산 후 가중치 업데이트
  
  - **추론 단계 (Autoregressive Generation)**
    - 모델의 예측 토큰 사용: "The quick brown" → 모델이 "fox" 생성 (확률 기반)
    - 생성된 "fox"를 다시 입력: "quick brown fox" → 다음 토큰 예측
    - 훈련과 다른 분포의 데이터 사용 → 오류 누적 (Distribution Shift)
  
  - **Hallucination 발생:**
    - 확률적 예측이 훈련 데이터의 패턴 조합으로 그럴듯한 하지만 거짓 정보 생성
    - 예: "파리의 높이는?" → 훈련 데이터에 없던 수치를 그럴듯하게 생성

- **대규모 데이터로 사전학습 (Pre-training)**
  - 인터넷 텍스트, 책, 코드 등 수십억~수조 개 토큰 학습
  - 더 많은 패턴 학습 = 더 정확한 다음 토큰 예측 능력
  - 매개변수 수가 많을수록 성능 향상 (스케일링 법칙)
  - **한계:** 순수 통계 모델이므로 근본적으로 hallucination 위험 존재

#### GPT 모델의 진화 (연도별)
- **GPT (2018)** - 1.17억 파라미터
  - Transformer 디코더만 사용한 첫 대규모 LLM

- **GPT-2 (2019)** - 15억 파라미터
  - 텍스트 생성 능력 대폭 향상
  - Zero-shot 학습 능력 입증

- **GPT-3 (2020)** - 1,750억 파라미터
  - Few-shot 학습으로 미세조정 없이 다양한 태스크 수행
  - 자연 언어 이해와 생성의 분수령 모델
