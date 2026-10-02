[← Part 1](./PART-1-README.md)

# CHAP-02 LLM의 한계와 이를 극복하기 위한 최신 아키텍처들

## 📑 Index

- [첫째, LLM의 태생적인 한계와 시스템적인 솔루션](#첫째-llm의-태생적인-한계와-시스템적인-솔루션)
  - [환각 현상](#환각-현상)
  - [기억불가](#기억불가)
  - [토큰제한](#토큰제한)
- [둘째, OpenAI의 Confession 연구](#둘째-openai의-confession-연구)
- [셋째, 구글의 연구들](#셋째-구글의-연구들)
  - [Titans](#titans)
  - [Nested Learning](#nested-learning)
  - [Diffusion](#diffusion)
  - [Diffusion 언어모델](#diffusion-언어모델)

---

## 첫째, LLM의 태생적인 한계와 시스템적인 솔루션

### 환각 현상

- 모델이 사실이 아닌 정보를 마치 참인 것처럼 생성하는 현상
- 훈련 데이터에 없거나 불완전한 정보를 신뢰도 높게 제시하는 문제
- Knowledge Cutoff: 모델 학습 시점 이후의 정보는 학습하지 못함
  - GPT-3.5: 2021년 9월
  - GPT-4: 2023년 4월
- 최신 정보 요청 시 모델이 추측으로 답변하면서 환각 심화
  - Cutoff 이후의 이벤트
  - 최신 기술
  - 인물 정보 등

**솔루션**
- Finetuning: 신뢰할 수 있는 도메인 데이터로 모델 재학습
  - 의료/법률 분야 전문 모델 구축
  - 기업 내부 정책 학습
  - 특정 업무 프로세스 최적화
- RAG (Retrieval-Augmented Generation): 검증된 외부 데이터베이스에서 정보 검색 후 답변 생성
  - 최신 뉴스 기반 요약 및 분석
  - 학술 논문 검색 및 인용
  - 제품 매뉴얼/FAQ 기반 고객지원

### 기억불가

- LLM은 제한된 길이의 컨텍스트 윈도우만 유지할 수 있음
- 긴 대화나 문서에서 초반 정보를 점진적으로 잊어버리는 현상
- [Transformer Explorer](https://poloclub.github.io/transformer-explainer/) - Attention 메커니즘 시각화로 이해 가능
- Self-Attention의 QKV(Query, Key, Value) 행렬연산 시간복잡도는 O(n²)
- 토큰 누적 시 메모리와 연산량이 기하급수적으로 증가

**솔루션**
- Long Context Models: 더 큰 컨텍스트 윈도우를 지원하는 모델 사용
  - Claude 200K, GPT-4 128K 컨텍스트 활용
  - 전체 문서 한 번에 처리 가능
  - 장편 콘텐츠 분석 및 요약
- Memory Management: 중요 정보 요약 및 구조화된 저장
  - 대화 히스토리 압축/요약
  - 주요 정보 별도 저장소 운영
  - 단계별 프로세스 설계로 정보 손실 최소화

### 토큰제한

- 입력과 출력의 토큰 수에 제한이 있어서 긴 텍스트 처리 어려움
- 토큰 제한으로 인해 대용량 데이터나 긴 문맥 분석 불가능

**솔루션**
- Token Compression: 입력 데이터를 효율적으로 압축/요약
  - 핵심 내용만 추출하여 전송
  - 중복 정보 제거
  - 다중 언어 처리 시 효율적 인코딩 활용
- Hierarchical Processing: 대용량 데이터를 계층적으로 나누어 처리
  - 문서를 섹션별로 분석 후 종합
  - 단계별 요약을 통한 정보 응축
  - 중요도 기반 선택적 처리

## 둘째, OpenAI의 Confession 연구 (2025)

- 논문: "Training LLMs for Honesty via Confessions" (2025년 12월 발표)
- 자백 유도: 모델에게 불확실한 부분을 명시적으로 인정하도록 유도하는 훈련 방식
- 정직성 향상: 확신할 수 없는 정보에 대해 "모름" 또는 불확실함을 표현하도록 학습
  - 환각 현상 감소
  - 사용자 신뢰도 향상
  - 오류 정정 가능성 증대
- 리워드 해킹의 문제: 보상 신호 최적화 추구로 인한 정직성 하락
  - 모델이 실제로 정직하지 않은 방식으로 정직한 것처럼 답변
  - 보상 시스템 악용으로 진정한 정직성 학습 불가
  - 복잡한 평가 시스템 설계의 필요성

## 셋째, 구글의 연구들

### Titans (2024)

- 논문: "Titans: Learning to Memorize at Test Time" (2024년 12월 발표)
- 단기 메모리(Attention)와 장기 메모리(Neural Memory Module)를 결합한 새로운 아키텍처
  - 컨텍스트 윈도우를 200만 토큰까지 확장 가능
  - 메모리 비용을 최소화하면서 장기 정보 보존
  - Transformer, Mamba 등 기존 모델보다 효율적

### Nested Learning (2025)

- 논문: "Nested Learning: The Illusion of Deep Learning Architectures" (NeurIPS 2025)
- 모델을 중첩된 최적화 문제로 보는 패러다임으로, 재앙적 망각(Catastrophic Forgetting) 해결
  - 각 레이어의 최적화를 독립적으로 처리하여 새로운 정보 학습 시 기존 지식 보존
  - HOPE(Hierarchical Optimizing Processing Ensemble) 구현으로 기존 Transformer 대비 우수한 성능
  - 언어 모델링과 상식 추론 벤치마크에서 낮은 perplexity와 높은 정확도 달성

### Diffusion (2022)

- Diffusion 모델: 이미지/텍스트 생성에서 반복적 디노이징 프로세스를 통한 생성 모델 (2022)
- 순차 생성(AR)의 대안으로 병렬 처리로 빠른 추론 속도 실현
  - 텍스트-이미지 생성에서 뛰어난 성능 (Photorealistic Text-to-Image Diffusion Models)
  - 양방향 컨텍스트 활용으로 더 정확한 생성 가능

### Diffusion 언어모델 (2025)

- Gemini Diffusion (2025년 5월 발표, Google I/O 2025): 병렬 청크 기반 텍스트 생성
  - H100에서 약 1,479 tokens/s의 매우 높은 처리량 달성
  - 생성 중 오류 수정으로 코드/수학 작업에서 우수한 성능
  - 빠른 반복과 양방향 문맥 활용으로 생성 품질 향상

