# RAG System 구축 (Part II)
- Part I에서 배운 검색의 원리를 실제로 구현하는 단계

## 📚 Index

- [<- Part I. 검색의 원리](1006-PART-01-검색의원리.md)

- [Part II. RAG System 구축](#part-ii-rag-system-구축)
  - [(1) RAG란? : Retrieval System + LLM](#1-rag란--retrieval-system--llm)
  - [(2) LangChain으로 RAG Pipeline 만들기](#2-langchain으로-rag-pipeline-만들기)
  - [(3) Chunking, Metadata, Prompt 설계](#3-chunking-metadata-prompt-설계)
  - [(4) RAG Evaluation : Ragas, LLM-as-a-Judge](#4-rag-evaluation--ragas-llmasamajudge)
  - [(5) Naive RAG의 한계와 Advanced Retrieval](#5-naive-rag의-한계와-advanced-retrieval)
  - [(6) Agentic RAG와 LangGraph, 최신 연구·사례](#6-agentic-rag와-langgraph-최신-연구사례)
  - [(7) RAG를 운영한다는 것 : 보안, Caching, Index 갱신](#7-rag를-운영한다는-것--보안-caching-index-갱신)
  - [(8) Practical TIPs : 기업 사례로 배우는 교훈](#8-practical-tips--기업-사례로-배우는-교훈)

## Part II. RAG System 구축

### (1) RAG란? : Retrieval System + LLM

**R (Retrieval)**

- **검색**: Part I에서 배운 검색의 원리를 실제로 구현하여, 사용자 쿼리에 맞는 관련 문서를 대규모 데이터베이스에서 빠르고 정확하게 찾아, Ranked된 우선순위를 부여해 LLM에 제공하는 검색 시스템

- **RAG의 필수 요소 - Index 구축**: RAG 시스템을 구축한다는 것 = **반드시 Retrieval을 위한 Index를 만든다는 의미**. 구체적으로: (1) Sparse RAG → Inverted Index 생성 (단어 → 문서), (2) Dense RAG → Vector DB/Index 생성 (벡터 → 문서, HNSW). Index 없이 RAG는 불가능 (매번 1억 문서 스캔 필요). 따라서 (2)~(8)의 모든 선택(Chunking, Embedding Model, Vector DB)은 결국 "최적의 Index를 만들기 위한 것"

**A (Augmented) & G (Generation)**

- **Augmented**: 검색된 문맥을 LLM 입력에 추가하여 생성 능력 강화 (모델 파라미터 변경 없음)
- **Generation**: 강화된 프롬프트를 바탕으로 LLM이 답변 생성

**Retrieval Augmented Generation 용어 분석**

- **Retrieval**: 쿼리에 맞는 관련 문서를 외부 데이터 저장소에서 동적으로 검색·조회
- **Augmented**: 검색된 문맥을 LLM 입력에 추가하여 생성 능력 강화 (파라미터 변경 없음)
- **Generation**: 강화된 프롬프트를 바탕으로 LLM이 답변 생성

**파인튜닝(Fine-tuning) vs RAG의 본질적 차이**

- **파인튜닝**: 모델 파라미터를 업데이트하여 특정 작업에 적응 (영구적 변경, 높은 계산비용)
  - 엄밀하게는 **Post-training**: 이미 학습된 기반 모델(Qwen, Claude, GPT 등)을 가져다 추가 학습하는 것
  - Pre-training(처음부터 학습)과 다름 → 대부분의 기업은 기존 모델 기반이므로 Fine-tuning = Post-training
  
- **RAG**: 모델 파라미터는 고정, 런타임(running)에 검색 결과를 컨텍스트로 주입 (유연성, 낮은 비용)

**LLM 시대의 등장배경**

- **LLM의 근본적 한계**: 
  - "안 본 눈 삽니다" 없음 → 학습 데이터에 없는 정보 환각(hallucination)
  - 지식 cutoff: 특정 시점(e.g., 2024-04)까지만 학습, 그 이후 정보 무지
  - 파라미터 크기 유한 → 모든 정보를 모델에 저장 불가능

- **2023 ChatGPT 출범 직후**: 파라미터 크기의 한계 노출 → 최신 정보/도메인 지식 부족 문제 심화
- **파인튜닝의 한계**: 모든 변경마다 파라미터 재학습 필요 → RAG는 런타임 가변성 제공으로 대체
- **현실적 선택지**: 파라미터 고정 + 동적 컨텍스트 주입 = 비용 효율적·유지보수 용이한 확장 패러다임

**Retrieval System + LLM: RAG의 작동 원리**

- **파이프라인 흐름**: 
  1. 사용자 질문 입수
  2. 미리 저장해둔 Knowledge에서 관련 정보 검색(Retrieval)
  3. 검색된 정보 + 사용자 질문을 합쳐서 프롬프트 구성
  4. LLM이 보강된 프롬프트를 기반으로 답변 생성 및 전달

- **Knowledge의 역할**: 정확한 답변을 제공하기 위한 검증된 정보 저장소
  - 최신 정보, 도메인 특화 데이터, 조직 내부 지식 등을 체계적으로 관리
  - LLM의 파라미터에 저장되지 않는 "외부 뇌" 역할

- **Knowledge 저장소 선택**:
  - **RDBMS** (PostgreSQL, MySQL): 
    - 행과 열로 정렬된 테이블 구조 데이터베이스 (엑셀 같은 형태)
    - SQL 언어로 데이터 검색/저장 ("이름이 김철수인 사람을 찾아줘")
    
    > **SQL (Structured Query Language)**
    > 
    > 데이터베이스에 명령을 내리는 언어. "이 조건에 맞는 데이터를 찾아줘" / "이 데이터를 저장해줘" 같은 요청을 RDBMS에 전달하는 방식
    > 
    > 예: `SELECT * FROM users WHERE name='김철수'` (이름이 김철수인 모든 사람 찾기)
    
    - 구조화된 정보, 정확한 단어 매칭 필요 시 사용 (Sparse Retrieval)
  - **Vector Store** (Pinecone, Weaviate, Milvus):
    - 텍스트를 벡터(숫자 배열)로 변환해서 저장하는 특화 DB
    - 벡터의 거리/유사도로 검색 (의미가 비슷한 문서 찾기 가능)
    - 자연어 이해, 의미론적 유사도 검색 필요 시 사용 (Dense Retrieval)

**RAG의 관건: Knowledge Base(KB)**

- **KB 없이는 RAG 불가능**: Retrieval할 정보가 없으면 단순 LLM이 됨 (파라미터에만 의존)
- **KB 품질 = RAG 성능**: 정확한 정보, 최신 데이터, 체계적 구조가 필수 → 답변의 정확성·신뢰도 결정
- **KB 구축 비용**: 데이터 수집 → 정제 → 인덱싱 → 지속적 운영 (시간·인력·비용 소요)

**LLM 확장 방식 비교: RAG vs Fine-tuning vs Long Context**

| 항목 | **RAG** | **Fine-tuning** | **Long Context** |
|------|---------|-----------------|------------------|
| **파라미터 변경** | ✗ 고정 | ✓ 업데이트 | ✗ 고정 |
| **새 정보 추가** | 실시간 추가 (DB 저장) | 재학습 필요 (오래 걸림) | 컨텍스트에만 포함 |
| **학습 비용** | 낮음 (인덱싱만) | 높음 (모델 학습) | 낮음 (추론 확장) |
| **추론 속도** | 중간 (Retrieval 거쳐야 함) | 빠름 (파라미터만 사용) | 느림 (긴 시퀀스 처리) |
| **정확성** | **높음** (최신 정보 활용) | **높음** (깊이 있는 학습) | 낮음 (정보 손실, context window 제한) |
| **유지보수** | **용이** (DB만 관리) | 어려움 (모델 재학습) | 어려움 (토큰 제한) |
| **최적 사용** | 빠르게 변하는 최신 정보 | 특정 분야 전문성 | 매우 긴 문맥 처리 |

**RAG vs Long Context: 200페이지 자료 처리 사례**

- **RAG 방식** (ChromaDB 활용):
  - 200페이지 전체를 청크(예: 500토큰)로 분할 → 각 청크를 벡터로 변환 → DB 저장
  - 사용자 질문 → 관련 청크만 검색(예: 상위 3개) → 검색된 청크만 프롬프트에 포함
  - **토큰 사용량**: 적음 (검색된 청크만 ~2,000토큰) → **비용 저렴**

- **Long Context 방식** (파일 첨부):
  - 200페이지 전체(예: 100,000토큰)를 그대로 프롬프트에 포함
  - LLM이 전체 문맥을 분석하여 답변
  - **토큰 사용량**: 매우 많음 (전체 자료 ~100,000토큰) → **비용 높음**
  - 정보 손실 위험: Context window 제한(예: 128k) 초과 시 일부 정보 누락

- **입력 토큰 & 속도 측면**:
  - **RAG**: 
    - 시간 = Retrieval 소모 시간(밀리초) + 입력 토큰 처리 시간(~2,000개 기준)
    - ⚠️ **속도 보장 불가**: Retrieval 자체가 병목 → 검색 인덱스 성능에 따라 전체 속도 결정
  - **Long Context**: 
    - 시간 = 입력 토큰 처리 시간(~100,000개 기준)
    - ⚠️ **하드웨어에 의존**: TPU/고성능 GPU가 있으면 대량 토큰 병렬 처리 가능 → 속도 확보
  - 예) 10개 질문 → RAG(일반 환경): 각 2~3초 vs Long Context(TPU 최적화): 각 3~5초 → 환경에 따라 달라짐

- **실제 제품의 선택**:
  - **Google Gemini**: 
    - TPU 인프라 보유 → **1M(백만) 토큰 Context window** 제공
    - Prompt Caching 지원 (반복 요청 시 캐시 활용)
    - ⚠️ **성능 이슈**: 1M 토큰 입력 시 응답 품질 저하, 실제 정확성 미흡 (이론적 우위 ≠ 실제 성능)
  - **OpenAI GPT**: 
    - 범용 인프라 기반 → RAG 구조 권장 (비용·효율성·안정성 우선)
    - 명확한 성능 기준 제시
  - **결론**: "어느 것이 빠르고 좋은가"는 **인프라 + 실제 성능**에 따라 다름

- **선택 기준**:
  - 자주 참조하는 자료 → **RAG** (Retrieval 활용, 입력 토큰 효율, 비용 저렴)
  - 일회성 대량 분석 → **Long Context** (Retrieval 불필요, 구현 간단)

**RAG vs Fine-tuning: 기업 시크릿 정보 관점**

- **RAG의 보안 우위**: 시크릿 정보가 외부 DB에만 저장, 모델 파라미터에 임베딩되지 않음 → 모델 유출·해킹 시에도 정보 보호 가능
- **Fine-tuning의 정보 노출 위험**: 학습 데이터가 모델 파라미터에 흡수됨 → 모델 추출(모델 도용) 시 기밀 정보 유출 가능성 높음
- **접근 제어 용이성**: RAG는 DB 레벨에서 권한 관리 가능 (누가 어떤 정보에 접근 가능한지 통제), Fine-tuning은 모델 전체로 학습되어 세밀한 제어 불가

**일상의 비유로 이해하기**

> **RAG**: "책의 목차를 보고 필요한 장(page)만 찾아서 답변한다" → 빠르고 효율적, 정보 업데이트 용이
>
> **Long Context**: "매번 책 전체를 읽고 답변한다" → 느리고 비효율적, 정보 손실 위험

### (2) LangChain으로 RAG Pipeline 만들기

### (3) Chunking, Metadata, Prompt 설계

### (4) RAG Evaluation : Ragas, LLM-as-a-Judge

### (5) Naive RAG의 한계와 Advanced Retrieval

### (6) Agentic RAG와 LangGraph, 최신 연구·사례

### (7) RAG를 운영한다는 것 : 보안, Caching, Index 갱신

### (8) Practical TIPs : 기업 사례로 배우는 교훈

