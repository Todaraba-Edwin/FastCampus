[← Part 1](./PART-1-README.md)

# CHAP-07 LangGraph로 구현하는 Basic RAG & Agentic RAG

## 📑 Index
- [첫째, LangGraph 기본 개념](#첫째-langgraph-기본-개념)
  - [(1) LangGraph 개요 및 특징](#1-langgraph-개요-및-특징)
  - [(2) StateGraph와 상태 관리](#2-stategraph와-상태-관리)
  - [(3) 노드(Node)와 엣지(Edge) 이해](#3-노드node와-엣지edge-이해)
- [둘째, 실습-02 LangGraph로 구현하는 Basic RAG](#둘째-실습-02-langgraph로-구현하는-basic-rag)
  - [(1) 환경 설정](#1-환경-설정)
  - [(2) 문서 로딩 및 벡터 스토어 생성](#2-문서-로딩-및-벡터-스토어-생성)
  - [(3) RAG 체인 설정](#3-rag-체인-설정)
  - [(4) LangGraph 상태 정의](#4-langgraph-상태-정의)
  - [(5) 그래프 노드 함수 정의](#5-그래프-노드-함수-정의)
  - [(6) 그래프 생성 및 컴파일](#6-그래프-생성-및-컴파일)
  - [(7) 그래프 시각화](#7-그래프-시각화)
  - [(8) RAG 실행 테스트](#8-rag-실행-테스트)
- [셋째, 실습-03 Agentic RAG 구현](#셋째-실습-03-agentic-rag-구현)
  - [(1) Agentic RAG 개요](#1-agentic-rag-개요)
  - [(2) 문서 전처리](#2-문서-전처리)
  - [(3) Retriever Tool 생성](#3-retriever-tool-생성)
  - [(4) 쿼리 생성 또는 응답](#4-쿼리-생성-또는-응답)
  - [(5) 문서 관련성 평가](#5-문서-관련성-평가)
  - [(6) 질문 재작성](#6-질문-재작성)
  - [(7) 답변 생성](#7-답변-생성)
  - [(8) Agentic RAG 그래프 조립](#8-agentic-rag-그래프-조립)
  - [(9) Agentic RAG 실행](#9-agentic-rag-실행)

---

## 첫째, LangGraph 기본 개념

### (1) LangGraph 개요 및 특징

#### 📌 학습 목표
- LangGraph의 정의 및 사용 목적 이해
- LangChain과의 핵심 차이점 파악

#### 🎓 스터디 노트 및 질문

> **LangGraph란?**

**정의**: LangChain을 기반으로 한 그래프 기반 워크플로우 프레임워크

**핵심 특징:**
- **상태 관리**: TypedDict를 통한 명확한 상태 정의
- **그래프 구조**: 노드(Node)와 엣지(Edge)로 복잡한 워크플로우 표현
- **조건부 분기**: 조건에 따른 동적 경로 선택
- **루프 구현**: 재시도 및 개선 루프 구성 가능
- **시각화**: 그래프 구조를 Mermaid 다이어그램으로 시각화
- **디버깅**: 각 노드의 입출력을 명확하게 추적 가능

**사용 사례:**
- 복잡한 다단계 에이전트 워크플로우
- 동적 의사결정 기반 RAG 시스템
- 멀티 턴 대화 관리
- 자동 평가 및 개선 루프
- 병렬 처리가 필요한 작업 흐름

**공식 문서**: [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)

---

### (2) StateGraph와 상태 관리

#### 📌 학습 목표
- StateGraph의 역할 이해
- GraphState를 통한 타입 안정성 확보

#### 🎓 스터디 노트 및 질문

> **StateGraph 란?**

```python
from langgraph.graph import StateGraph, START, END

# StateGraph 생성
workflow = StateGraph(GraphState)

# 노드 추가
workflow.add_node("retrieve", retrieve)
workflow.add_node("generate", generate)

# 엣지 연결
workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

# 컴파일
app = workflow.compile()
```

**역할:**
- 그래프 기반 워크플로우의 기본 구조 제공
- 노드와 엣지를 연결하여 실행 흐름 정의
- START와 END라는 특수 노드로 진입/종료 표시

**StateGraph vs LangChain Chains:**

| 항목 | **StateGraph** | **LangChain Chains** |
|------|---|---|
| **구조** | 그래프 기반 (노드 + 엣지) | 선형 체인 (순차적) |
| **제어 흐름** | 복잡한 분기, 루프 가능 | 단순 순차 실행 |
| **상태 관리** | 명확한 상태 객체 (TypedDict) | 암묵적 상태 전달 |
| **시각화** | Mermaid 다이어그램 | 불가 |
| **확장성** | 높음 | 낮음 |

---

### (3) 노드(Node)와 엣지(Edge) 이해

#### 📌 학습 목표
- 노드 함수의 입출력 구조 이해
- 다양한 엣지 연결 방식 학습

#### 🎓 스터디 노트 및 질문

> **노드(Node)의 구조**

**정의**: 상태를 입력받아 수정된 상태를 반환하는 함수

```python
def retrieve(state: GraphState) -> GraphState:
    """
    입력: GraphState 객체 (question, documents, generation)
    처리: question을 기반으로 문서 검색
    출력: 업데이트된 GraphState (documents 필드 추가)
    """
    question = state["question"]
    documents = retriever.invoke(question)
    return {"question": question, "documents": documents, "generation": ""}
```

**노드 함수의 특징:**
- 상태를 입력으로 받음
- 상태의 일부 또는 전부를 수정
- 수정된 상태를 딕셔너리 형태로 반환 (자동 병합됨)
- 각 노드는 독립적으로 실행 가능

> **엣지(Edge)의 종류**

**1. 일반 엣지 (add_edge)**
```python
workflow.add_edge("retrieve", "generate")
# retrieve 노드 → generate 노드로 항상 진행
```

**2. 조건부 엣지 (add_conditional_edges)**
```python
workflow.add_conditional_edges(
    "retrieve",
    grade_documents,  # 조건을 판단하는 함수
    {
        "generate_answer": "generate",
        "rewrite_question": "rewrite"
    }
)
# grade_documents 결과에 따라 다른 노드로 분기
```

**3. START/END 특수 노드**
```python
workflow.add_edge(START, "retrieve")  # 시작점
workflow.add_edge("generate", END)    # 종료점
```

---

## 둘째, 실습-02 LangGraph로 구현하는 Basic RAG

### (1) 환경 설정

#### 📌 학습 목표
- LangGraph 및 필수 패키지 설치 및 환경 구성
- OpenAI API 키 설정 및 연동

#### 🎓 스터디 노트 및 질문

> 여기에 학습하며 생긴 질문들을 기록하세요.

---

### (2) 문서 로딩 및 벡터 스토어 생성

#### 📌 학습 목표
- PyPDFLoader를 사용한 PDF 문서 로드
- RecursiveCharacterTextSplitter와 Chroma를 활용한 벡터 스토어 구축

#### 🎓 스터디 노트 및 질문

> **벡터 스토어 선택: InMemoryVectorStore vs Chroma**

| 항목 | InMemoryVectorStore | Chroma |
|------|---|---|
| **저장 방식** | 메모리에만 저장 | 디스크에 영구 저장 |
| **속도** | ⚡ 초고속 | 📊 약간의 오버헤드 |
| **데이터 크기** | 소규모 (<1000문서) | 대규모 (>1000문서) |
| **재시작 후** | 🔴 데이터 손실 | 🟢 데이터 유지 |
| **사용 사례** | 프로토타이핑, 테스트 | 프로덕션 서비스 |

**선택 기준:**
- **개발/테스트**: InMemoryVectorStore (빠르고 간단)
- **프로덕션/서비스**: Chroma (데이터 지속성 필요)

> **⚠️ Chroma의 영구 저장 메커니즘 이해하기**

**Chroma는 데이터를 디스크에 저장하므로, 명시적으로 삭제하지 않으면 계속 누적됩니다:**

```python
vectorstore = Chroma.from_documents(
    documents=doc_splits,
    collection_name="rag-chroma",
    persist_directory="./chroma_data"  # ← 이 경로에 계속 저장됨
)
```

- **첫 실행**: `./chroma_data` 디렉토리 생성, `rag-chroma` 컬렉션 저장
- **반복 실행**: 같은 컬렉션명으로 실행 → 기존 데이터 유지/업데이트
- **명시적 삭제 전까지**: 디스크에 계속 누적 (자동 정리 불가)

**메모리 vs 디스크 비교:**

| 항목 | InMemoryVectorStore | Chroma |
|------|---|---|
| **메모리 점유** | 프로세스 종료 시 해제 | 명시적 삭제까지 지속 |
| **디스크 공간** | 사용 안 함 | 1 MB ~ 수 GB (누적) |
| **정리 방법** | 자동 해제 | 수동 정리 필요 |
| **주의사항** | 없음 | `rm -rf ./chroma_data` 필요 |

> **문서 로딩 및 벡터 스토어 생성 과정**

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# 임베딩 모델 설정
embeddings = OpenAIEmbeddings()

# PDF 파일 경로
file_path = "docs/DeepSeek_OCR_paper.pdf"

# PDF 로드
loader = PyPDFLoader(file_path)
docs = loader.load()
print(f"📄 로드된 문서 수: {len(docs)} 페이지")

# 문서 분할
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=500, 
    chunk_overlap=50
)
doc_splits = text_splitter.split_documents(docs)
print(f"총 {len(doc_splits)}개의 문서 청크 생성됨")

# 벡터 스토어 생성
vectorstore = Chroma.from_documents(
    documents=doc_splits,
    collection_name="rag-chroma",
    embedding=embeddings,
)

# 리트리버 생성
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
```

**핵심 단계:**

1. **PDF 로딩**: PyPDFLoader로 문서 페이지 로드
2. **텍스트 분할**: 청크 크기 500, 오버랩 50으로 재귀적 분할
3. **임베딩**: OpenAIEmbeddings로 벡터 변환
4. **벡터 스토어**: Chroma에 저장하여 유사도 검색 가능

---

### (3) RAG 체인 설정

#### 📌 학습 목표
- ChatOpenAI 모델 초기화 및 설정
- LCEL 파이프라인 연산자를 사용한 체인 구성

#### 🎓 스터디 노트 및 질문

> **LCEL 파이프라인 연산자 `|` 이해하기**

```python
rag_chain = prompt | llm | StrOutputParser()
```

**`|`는 "할당"이 아니라 "파이프라인 연결" 연산자입니다:**

```
입력 데이터 {context, question}
         ↓
    prompt (템플릿 포맷팅)
         ↓ | (파이프로 연결)
      llm (LLM 실행)
         ↓ | (파이프로 연결)
StrOutputParser (텍스트 추출)
         ↓
   최종 답변 (str)
```

**각 단계의 역할:**

| 단계 | 입력 | 처리 내용 | 출력 |
|------|------|---------|------|
| `prompt` | `{context, question}` 딕셔너리 | ChatPromptTemplate으로 포맷팅 | 완성된 프롬프트 문자열 |
| `llm` | 프롬프트 문자열 | ChatOpenAI 모델 호출 | AIMessage 객체 |
| `StrOutputParser` | AIMessage 객체 | `.content` 필드 추출 | 순수 문자열 (str) |

> **StrOutputParser의 정확한 역할**

**LLM의 출력:**
```python
response = llm.invoke(prompt)
# → AIMessage(content="답변 텍스트", response_metadata={...})
```

**StrOutputParser의 역할:**
```python
StrOutputParser().invoke(response)
# → "답변 텍스트" (문자열로 변환)
```

**핵심 개념:**
- ✅ **문자열 추출**: AIMessage 객체 → 순수 str (`.content` 필드)
- ❌ **벡터 디코딩 아님**: 벡터는 LLM 내부에서 이미 처리됨
- 🔗 **함수형 파이프라인**: 각 단계의 출력이 다음 단계의 입력이 됨

**주의:**
- 임베딩 모델이 생성하는 벡터(embedding)와 LLM 출력은 다릅니다
- 임베딩 벡터 → 벡터 검색(유사도 계산) → 텍스트 추출
- LLM 출력(텍스트) → StrOutputParser → 순수 문자열

> **RAG 체인의 구조**

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

# LLM 모델 설정
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# RAG 프롬프트 템플릿
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an assistant for question-answering tasks. 
    Use the following pieces of retrieved context to answer the question. 
    If you don't know the answer, just say that you don't know. 
    Use three sentences maximum and keep the answer concise. Answer in Korean"""),
    ("human", "Question: {question}\n\nContext: {context}\n\nAnswer:")
])

# 문서 포맷팅 함수
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# RAG 체인 생성
rag_chain = prompt | llm | StrOutputParser()
```

**파이프라인 구성:**
- **prompt**: 시스템 메시지 + 사용자 질문 포맷팅
- **llm**: ChatOpenAI 모델 호출
- **StrOutputParser()**: 출력을 문자열로 파싱

---

### (4) LangGraph 상태 정의

#### 📌 학습 목표
- TypedDict를 사용한 GraphState 정의
- 그래프에서 사용할 상태 필드 설계

#### 🎓 스터디 노트 및 질문

> **GraphState 정의의 중요성**

```python
from typing import TypedDict, List
from langchain_core.documents import Document

class GraphState(TypedDict):
    """RAG 그래프의 상태를 정의"""
    question: str              # 사용자 질문
    documents: List[Document]  # 검색된 문서 리스트
    generation: str            # 생성된 답변
```

**각 필드의 역할과 데이터 흐름:**

| 필드 | 타입 | 역할 | 초기값 | 업데이트 노드 |
|------|------|------|--------|---|
| `question` | `str` | 사용자의 입력 질문 저장 | invoke() 입력값 | - |
| `documents` | `List[Document]` | 검색된 관련 문서 저장 | [] | retrieve 노드 |
| `generation` | `str` | 생성된 최종 답변 저장 | "" | generate 노드 |

**TypedDict 사용 이유:**
- 상태의 구조를 명시적으로 정의하여 타입 안정성 확보
- 각 노드에서 상태의 필드를 명확히 알 수 있음
- IDE의 자동완성 및 타입 체킹 가능

---

### (5) 그래프 노드 함수 정의

#### 📌 학습 목표
- retrieve 노드: 질문 기반 문서 검색 구현
- generate 노드: 검색 결과를 바탕으로 답변 생성 구현

#### 🎓 스터디 노트 및 질문

> **retrieve 노드 - 문서 검색**

```python
def retrieve(state: GraphState) -> GraphState:
    """문서 검색 노드"""
    print("---RETRIEVE---")
    question = state["question"]
    
    # 리트리버를 사용해 관련 문서 검색 (상위 k개)
    documents = retriever.invoke(question)
    
    return {
        "question": question, 
        "documents": documents, 
        "generation": ""
    }
```

**동작 과정:**
1. 입력 상태에서 `question` 필드 추출
2. 리트리버(벡터 스토어)에서 유사한 문서 상위 k개 검색
3. 검색 결과를 `documents` 필드에 저장하여 반환

**generate 노드 - 답변 생성**

```python
def generate(state: GraphState) -> GraphState:
    """답변 생성 노드"""
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    
    # 검색된 문서들을 문자열로 포맷팅
    docs_txt = format_docs(documents)
    
    # RAG 체인에 컨텍스트와 질문 전달
    generation = rag_chain.invoke({
        "context": docs_txt, 
        "question": question
    })
    
    return {
        "question": question,
        "documents": documents,
        "generation": generation
    }
```

**동작 과정:**
1. 입력 상태에서 `question`과 `documents` 추출
2. 문서들을 포맷팅하여 LLM 프롬프트용 컨텍스트 준비
3. RAG 체인 호출: 컨텍스트 + 질문 → LLM 답변
4. 생성된 답변을 `generation` 필드에 저장하여 반환

---

### (6) 그래프 생성 및 컴파일

#### 📌 학습 목표
- StateGraph를 사용한 워크플로우 구성
- 노드 추가 및 엣지 연결을 통한 실행 흐름 정의

#### 🎓 스터디 노트 및 질문

> **Basic RAG 그래프 구조: START → retrieve → generate → END**

```python
from langgraph.graph import StateGraph, START, END

# 그래프 생성
workflow = StateGraph(GraphState)

# 노드 추가
workflow.add_node("retrieve", retrieve)
workflow.add_node("generate", generate)

# 엣지 연결 (노드 간 실행 순서 정의)
workflow.add_edge(START, "retrieve")      # 시작 → 검색
workflow.add_edge("retrieve", "generate")  # 검색 → 생성
workflow.add_edge("generate", END)         # 생성 → 종료

# 그래프 컴파일
app = workflow.compile()
print("RAG 그래프가 성공적으로 컴파일되었습니다.")
```

**각 단계의 의미:**

1. **StateGraph 초기화**: GraphState를 상태로 사용하는 그래프 생성
2. **노드 추가**: retrieve와 generate 함수를 그래프에 등록
3. **엣지 정의**: 노드 간의 연결 순서 명시 (START → retrieve → generate → END)
4. **컴파일**: 그래프를 실행 가능한 형태로 변환

---

### (7) 그래프 시각화

#### 📌 학습 목표
- get_graph().draw_mermaid_png()를 사용한 그래프 시각화
- 워크플로우 구조를 다이어그램으로 확인

#### 🎓 스터디 노트 및 질문

> **그래프 시각화 방법**

```python
from IPython.display import Image, display

try:
    display(Image(app.get_graph().draw_mermaid_png()))
except Exception as e:
    print(f"시각화 오류: {e}")
    print("그래프 구조: START -> retrieve -> generate -> END")
```

**시각화의 의미:**
- Mermaid 다이어그램으로 그래프 구조 표현
- 각 노드와 엣지의 관계를 시각적으로 이해
- 복잡한 그래프 구조에서 실행 흐름 파악 용이

**기본 RAG 그래프의 시각적 구조:**
```
┌─────────┐
│ START   │
└────┬────┘
     │
     ▼
┌─────────────┐
│  retrieve   │
└────┬────────┘
     │
     ▼
┌─────────────┐
│  generate   │
└────┬────────┘
     │
     ▼
┌─────────┐
│   END   │
└─────────┘
```

---

### (8) RAG 실행 테스트

#### 📌 학습 목표
- invoke() 동기 실행: 전체 결과를 한 번에 반환받기
- stream() 스트리밍 실행: 각 노드 실행 과정을 순차적으로 확인하기

#### 🎓 스터디 노트 및 질문

> **invoke() - 동기 실행 방식**

```python
# RAG 그래프 실행
question = "Deepseek OCR이 뭐야?"
result = app.invoke({"question": question})

print("=" * 50)
print(f"질문: {result['question']}")
print("=" * 50)
print(f"\n답변:\n{result['generation']}")
print("=" * 50)
print(f"\n참조 문서 수: {len(result['documents'])}")
```

**특징:**
- 전체 워크플로우가 완료될 때까지 대기
- 최종 결과 상태(question, documents, generation)를 한 번에 반환
- 결과에서 원하는 필드(generation)에 접근 가능

**stream() - 스트리밍 실행 방식**

```python
# 스트리밍 방식으로 실행
question = "Omnidoc bench 결과는 어때?"
print(f"질문: {question}\n")
print("=" * 50)

for output in app.stream({"question": question}):
    for node_name, value in output.items():
        print(f"\n[{node_name}] 노드 실행 완료")
        if node_name == "generate" and "generation" in value:
            print(f"\n답변:\n{value['generation']}")
```

**특징:**
- 각 노드 실행 결과를 순차적으로 반환 (먼저 retrieve 결과, 그 다음 generate 결과)
- 중간 과정을 실시간으로 확인 가능
- 디버깅 및 진행 상황 추적에 유용
- 각 노드의 입출력을 명확하게 관찰 가능

**invoke vs stream 비교:**

| 항목 | `invoke()` | `stream()` |
|------|-----------|-----------|
| **대기 방식** | 전체 완료까지 대기 | 각 노드 후 반환 |
| **반환 방식** | 최종 상태 한 번 | 각 노드마다 여러 번 |
| **사용 시기** | 최종 결과만 필요 | 진행 과정 확인 필요 |
| **응답 속도 체감** | 느림 | 빠름 (진행 상황 보임) |

---

## 셋째, 실습-03 Agentic RAG 구현

### (1) Agentic RAG 개요

#### 📌 학습 목표
- LLM 에이전트가 검색 여부를 자율적으로 판단하는 고급 RAG 시스템 이해
- Basic RAG와의 구조적 차이점 파악 및 장점 학습

#### 🎓 스터디 노트 및 질문

> 여기에 학습하며 생긴 질문들을 기록하세요.

---

### (2) 문서 전처리

#### 📌 학습 목표
- PDF 문서 로딩 및 벡터 스토어 구축 (Basic RAG와 동일하게 재사용)
- Agentic RAG용 문서 준비 완료

#### 🎓 스터디 노트 및 질문

> 여기에 학습하며 생긴 질문들을 기록하세요.

---

### (3) Retriever Tool 생성

#### 📌 학습 목표
- @tool 데코레이터를 사용한 도구 정의 및 변환 원리
- 에이전트가 호출할 수 있는 retriever tool 생성

#### 🎓 스터디 노트 및 질문

> **@tool 데코레이터를 사용한 검색 도구 구현**

```python
from langchain.tools import tool

@tool
def retrieve(query: str) -> str:
    """DeepSeek OCR 논문에서 관련 정보를 검색합니다."""
    docs = retriever.invoke(query)
    return "\n\n".join([doc.page_content for doc in docs])

retriever_tool = retrieve
print("✅ Retriever Tool 생성 완료")
```

**동작 원리:**

1. **@tool 데코레이터**: 일반 함수를 LangChain Tool 객체로 변환
2. **함수명** → Tool 이름: `retrieve`
3. **Docstring** → Tool 설명: "DeepSeek OCR 논문에서..."
4. **함수 인자** → Tool 파라미터: `query: str`

**Tool 객체의 정보 확인:**

```python
retriever_tool.name           # "retrieve"
retriever_tool.description    # "DeepSeek OCR 논문에서..."
retriever_tool.args           # {"query": {"type": "string"}}
```

---

### (4) 쿼리 생성 또는 응답

#### 📌 학습 목표
- LLM이 검색 필요 여부를 자동으로 판단하는 generate_query_or_respond 노드 구현
- bind_tools()를 사용한 도구 바인딩 및 Tool Call 생성

#### 🎓 스터디 노트 및 질문

> **generate_query_or_respond 노드 - Tool 호출 여부 판단**

```python
from langgraph.graph import MessagesState

response_model = ChatOpenAI("gpt-4o-mini", temperature=0)

def generate_query_or_respond(state: MessagesState):
    """LLM이 검색 여부를 판단하여 tool 호출 또는 직접 응답"""
    response = (
        response_model
        .bind_tools([retriever_tool])  # retriever tool 바인딩
        .invoke(state["messages"])
    )
    return {"messages": [response]}

print("✅ generate_query_or_respond 노드 정의 완료")
```

**동작 원리:**

1. **bind_tools()**: LLM에 사용 가능한 도구 바인딩
2. **LLM 판단**: 사용자 메시지를 보고 retriever 호출 필요성 판단
3. **결과**:
   - 검색 필요 → `tool_calls` 포함된 응답 생성
   - 검색 불필요 → 직접 텍스트 응답 생성

**테스트 예시:**

```python
# 검색 불필요 (일반 인사)
test_input = {"messages": [{"role": "user", "content": "hello!"}]}
generate_query_or_respond(test_input)["messages"][-1].pretty_print()
# → "Hello! How can I assist you today?"

# 검색 필요 (주제별 질문)
test_input = {"messages": [{"role": "user", "content": "DeepEncoder란?"}]}
generate_query_or_respond(test_input)["messages"][-1].pretty_print()
# → Tool Calls: retrieve (call_id), Args: {query: DeepEncoder}
```

---

### (5) 문서 관련성 평가

#### 📌 학습 목표
- 검색된 문서가 질문과 관련있는지 평가하는 grade_documents 노드 구현
- Pydantic BaseModel을 사용한 구조화된 출력

#### 🎓 스터디 노트 및 질문

> **grade_documents - 문서 관련성 평가 노드**

```python
from pydantic import BaseModel, Field
from typing import Literal

GRADE_PROMPT = (
    "You are a grader assessing relevance of a retrieved document to a user question.\n"
    "Here is the retrieved document:\n\n{context}\n\n"
    "Here is the user question: {question}\n"
    "If the document contains keyword(s) or semantic meaning related to the user question, "
    "grade it as relevant.\n"
    "Give a binary score 'yes' or 'no' to indicate whether the document is relevant."
)

class GradeDocuments(BaseModel):
    """문서 관련성 평가를 위한 이진 점수"""
    binary_score: str = Field(description="Relevance score: 'yes' or 'no'")

grader_model = ChatOpenAI("gpt-4o-mini", temperature=0)

def grade_documents(state: MessagesState) -> Literal["generate_answer", "rewrite_question"]:
    """검색된 문서의 관련성을 평가"""
    question = state["messages"][0].content       # 사용자 질문
    context = state["messages"][-1].content       # 검색된 문서 내용
    
    prompt = GRADE_PROMPT.format(question=question, context=context)
    response = grader_model.with_structured_output(GradeDocuments).invoke(
        [{"role": "user", "content": prompt}]
    )
    
    if response.binary_score == "yes":
        print("---GRADE: 관련 있음---")
        return "generate_answer"
    else:
        print("---GRADE: 관련 없음 → 질문 재작성---")
        return "rewrite_question"

print("✅ grade_documents 조건부 엣지 정의 완료")
```

**동작 원리:**

1. **질문 추출**: `messages[0]` → 사용자의 원본 질문
2. **문서 추출**: `messages[-1]` → retriever tool의 검색 결과
3. **LLM 평가**: 문서가 질문과 관련있는지 yes/no로 판단
4. **조건부 분기**:
   - `yes` → generate_answer 노드로 진행 (답변 생성)
   - `no` → rewrite_question 노드로 진행 (질문 재작성)

---

### (6) 질문 재작성

#### 📌 학습 목표
- 관련 없는 검색 결과 시 질문을 개선하는 rewrite_question 노드 구현
- 루프를 통한 자동 개선 메커니즘 이해

#### 🎓 스터디 노트 및 질문

> **rewrite_question - 질문 개선 노드**

```python
from langchain_core.messages import HumanMessage

REWRITE_PROMPT = (
    "Look at the input and try to reason about the underlying semantic intent / meaning.\n"
    "Here is the initial question:\n---\n{question}\n---\n"
    "Formulate an improved question:"
)

def rewrite_question(state: MessagesState):
    """질문을 재작성"""
    print("---REWRITE QUESTION---")
    question = state["messages"][0].content
    prompt = REWRITE_PROMPT.format(question=question)
    response = response_model.invoke([{"role": "user", "content": prompt}])
    return {"messages": [HumanMessage(content=response.content)]}

print("✅ rewrite_question 노드 정의 완료")
```

**동작 원리:**

1. 사용자의 원본 질문을 의미 기반으로 분석
2. LLM이 더 명확하고 구체적인 질문으로 개선
3. 개선된 질문을 `messages`에 추가하여 다시 검색하도록 유도
4. 루프: rewrite_question → generate_query_or_respond로 재진입

---

### (7) 답변 생성

#### 📌 학습 목표
- 최종 답변을 생성하는 generate_answer 노드 구현
- 관련성 평가를 통과한 검색 결과를 활용한 답변 생성

#### 🎓 스터디 노트 및 질문

> **generate_answer - 최종 답변 생성 노드**

```python
GENERATE_PROMPT = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer the question. "
    "If you don't know the answer, just say that you don't know. "
    "Use three sentences maximum and keep the answer concise.\n"
    "Question: {question}\nContext: {context}"
)

def generate_answer(state: MessagesState):
    """최종 답변 생성"""
    print("---GENERATE ANSWER---")
    question = state["messages"][0].content
    context = state["messages"][-1].content
    prompt = GENERATE_PROMPT.format(question=question, context=context)
    response = response_model.invoke([{"role": "user", "content": prompt}])
    return {"messages": [response]}

print("✅ generate_answer 노드 정의 완료")
```

**동작 원리:**

1. 원본 질문 + 검색된 문서(관련성 평가를 통과한)를 컨텍스트로 준비
2. LLM으로 최종 답변 생성
3. 3문장 이내의 간결한 답변 작성
4. 모르는 경우 "모른다"고 명시

---

### (8) Agentic RAG 그래프 조립

#### 📌 학습 목표
- 모든 노드와 조건부 엣지를 연결하여 완전한 Agentic RAG 그래프 구성
- 루프와 분기를 포함한 복잡한 워크플로우 구현

#### 🎓 스터디 노트 및 질문

> **Agentic RAG 그래프 구조**

```
START
  ↓
generate_query_or_respond (검색 필요 판단)
  ├─ Yes (tool_calls) → retrieve (문서 검색)
  │                        ↓
  │                   grade_documents (관련성 평가)
  │                        ├─ Yes → generate_answer → END
  │                        └─ No → rewrite_question → (루프) generate_query_or_respond
  └─ No → END (직접 응답)
```

**그래프 생성 코드:**

```python
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

# Agentic RAG 그래프 생성
agentic_workflow = StateGraph(MessagesState)

# 노드 추가
agentic_workflow.add_node("generate_query_or_respond", generate_query_or_respond)
agentic_workflow.add_node("retrieve", ToolNode([retriever_tool]))
agentic_workflow.add_node("rewrite_question", rewrite_question)
agentic_workflow.add_node("generate_answer", generate_answer)

# 시작 엣지
agentic_workflow.add_edge(START, "generate_query_or_respond")

# 조건부 엣지 1: Tool 호출 여부 판단
agentic_workflow.add_conditional_edges(
    "generate_query_or_respond",
    tools_condition,  # LLM 응답에서 tool_calls 존재 여부 확인
    {"tools": "retrieve", END: END}  # tool_calls 있으면 retrieve, 없으면 END
)

# 조건부 엣지 2: 문서 관련성 평가
agentic_workflow.add_conditional_edges(
    "retrieve", 
    grade_documents  # "generate_answer" or "rewrite_question"
)

# 일반 엣지
agentic_workflow.add_edge("generate_answer", END)
agentic_workflow.add_edge("rewrite_question", "generate_query_or_respond")  # 루프

# 컴파일
agentic_graph = agentic_workflow.compile()
print("✅ Agentic RAG 그래프 컴파일 완료!")
```

**그래프 구조 분석:**

| 노드 | 역할 | 입력 | 출력 | 조건부 분기 |
|------|------|------|------|-----------|
| generate_query_or_respond | 검색 필요 여부 판단 | 메시지 리스트 | AI 응답 or Tool Call | tools_condition |
| retrieve | 문서 검색 실행 | Tool Call | 검색된 문서 | 없음 |
| grade_documents | 문서 관련성 평가 | 질문 + 문서 | yes/no | 자체 함수 |
| generate_answer | 최종 답변 생성 | 질문 + 문서 | 최종 답변 | 없음 |
| rewrite_question | 질문 개선 | 원본 질문 | 개선된 질문 | 없음 |

---

### (9) Agentic RAG 실행

#### 📌 학습 목표
- stream() 메서드로 Agentic RAG의 동적 의사결정 과정 확인
- 다양한 쿼리 타입에서의 에이전트 동작 추적

#### 🎓 스터디 노트 및 질문

> **Agentic RAG 실행 - 검색이 필요한 질문**

```python
# 테스트 1: 전문 용어 질문 (검색 필요)
for chunk in agentic_graph.stream(
    {"messages": [{"role": "user", "content": "DeepSeek OCR이 뭐야?"}]}
):
    for node, update in chunk.items():
        print(f"\n🔄 Update from node: {node}")
        update["messages"][-1].pretty_print()
```

**예상 실행 흐름:**

1. **generate_query_or_respond**: 
   - 입력: "DeepSeek OCR이 뭐야?"
   - 판단: 검색 필요
   - 출력: Tool Call (retrieve, query="DeepSeek OCR")

2. **retrieve**:
   - 입력: Tool Call (query="DeepSeek OCR")
   - 처리: 벡터 스토어에서 관련 문서 검색
   - 출력: 검색된 문서 텍스트

3. **grade_documents**:
   - 입력: 질문 + 검색된 문서
   - 판단: "yes" (관련성 높음)
   - 출력: "generate_answer" 선택

4. **generate_answer**:
   - 입력: 질문 + 검색된 문서
   - 처리: LLM으로 최종 답변 생성
   - 출력: "DeepSeek OCR은 긴 문서의 텍스트를 고해상도 입력에서..."

5. **END**: 그래프 종료

**Agentic RAG 실행 - 검색이 불필요한 질문**

```python
# 테스트 2: 일반 인사 (검색 불필요)
for chunk in agentic_graph.stream(
    {"messages": [{"role": "user", "content": "안녕하세요!"}]}
):
    for node, update in chunk.items():
        print(f"🔄 Update from node: {node}")
```

**예상 실행 흐름:**

1. **generate_query_or_respond**:
   - 입력: "안녕하세요!"
   - 판단: 검색 불필요
   - 출력: 직접 응답 ("안녕하세요! 뭘 도와드릴까요?")

2. **END**: 검색 없이 바로 종료

**stream() 메서드의 활용:**

```python
# 각 노드의 실행 과정을 실시간으로 관찰
for chunk in agentic_graph.stream({"messages": [{"role": "user", "content": "질문"}]}):
    for node_name, value in chunk.items():
        print(f"\n[{node_name}]")
        if "messages" in value:
            value["messages"][-1].pretty_print()
```

**장점:**
- 에이전트의 의사결정 과정을 단계별로 추적 가능
- 각 노드에서의 LLM 응답 확인
- 문제 발생 지점을 명확하게 파악
- 루프 실행 여부 및 횟수 확인

---

## 📚 학습 정리

### LangGraph의 핵심 개념 요약

1. **StateGraph**: 노드와 엣지로 워크플로우 정의
2. **GraphState**: TypedDict로 상태 구조 명시
3. **노드 함수**: 상태를 입력받아 수정하는 순수 함수
4. **엣지**: 노드 간의 실행 흐름 (일반/조건부)
5. **조건부 라우팅**: tools_condition, add_conditional_edges
6. **상태 머신**: START → 노드들 → END

### Basic RAG vs Agentic RAG

- **Basic RAG**: 단순하고 빠름 (항상 검색)
- **Agentic RAG**: 복잡하지만 유연함 (지능형 검색)

### 다음 학습 단계

- CHAP-08: 하이브리드 RAG (검색 개선 기법)
- CHAP-09: 에이전트 고급 기능 (메모리, 도구 확장)
- CHAP-10: 프로덕션 배포 및 모니터링

---

## 📚 참고 자료

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangChain Expression Language (LCEL)](https://python.langchain.com/docs/expression_language/)
- [Tool Use in LangChain](https://python.langchain.com/docs/modules/tools/)
