[← Part 1](./PART-1-README.md)

# CHAP-07 LangGraph로 구현하는 Basic RAG & Agentic RAG

## 📑 Index
- [첫째, LangGraph 기본 개념](#첫째-langgraph-기본-개념)
  - [(1) LangGraph 개요 및 특징](#1-langgraph-개요-및-특징)
  - [(2) StateGraph와 상태 관리](#2-stategraph와-상태-관리)
  - [(3) 노드(Node)와 엣지(Edge) 이해](#3-노드node와-엣지edge-이해)
- [둘째, 실습-02 LangGraph로 구현하는 Basic RAG](#둘째-실습-02-langgraph로-구현하는-basic-rag)
  - [(1) 환경 설정 및 벡터 스토어](#1-환경-설정-및-벡터-스토어)
  - [(2) GraphState 정의](#2-graphstate-정의)
  - [(3) 노드 함수 구현 (Retrieve, Generate)](#3-노드-함수-구현-retrieve-generate)
  - [(4) 그래프 생성 및 컴파일](#4-그래프-생성-및-컴파일)
  - [(5) 그래프 시각화 및 실행](#5-그래프-시각화-및-실행)
  - [(6) 실습: 질의응답 테스트](#6-실습-질의응답-테스트)
- [셋째, 실습-03 Agentic RAG 구현](#셋째-실습-03-agentic-rag-구현)
  - [(1) Agentic RAG 개요](#1-agentic-rag-개요)
  - [(2) Retriever Tool 생성](#2-retriever-tool-생성)
  - [(3) 조건부 라우팅 (Conditional Routing)](#3-조건부-라우팅-conditional-routing)
  - [(4) 문서 관련성 평가](#4-문서-관련성-평가)
  - [(5) Agentic RAG 그래프 조립](#5-agentic-rag-그래프-조립)
  - [(6) 실습: 동적 의사결정 테스트](#6-실습-동적-의사결정-테스트)

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

### (1) 환경 설정 및 벡터 스토어

#### 📌 학습 목표
- LangGraph와 필수 패키지 설치
- 벡터 스토어 생성 및 리트리버 설정

#### 🎓 스터디 노트 및 질문

> 여기에 학습하며 생긴 질문들을 기록하세요.

### (2) GraphState 정의

#### 📌 학습 목표
- TypedDict를 사용한 상태 객체 정의
- 상태 필드의 역할 이해

#### 🎓 스터디 노트 및 질문

> **GraphState 구조**

```python
from typing import TypedDict, List
from langchain_core.documents import Document

class GraphState(TypedDict):
    """RAG 그래프의 상태를 정의"""
    question: str           # 사용자 질문
    documents: List[Document]  # 검색된 문서 리스트
    generation: str         # 생성된 답변
```

**각 필드의 역할:**

| 필드 | 타입 | 설명 |
|------|------|------|
| `question` | `str` | 사용자로부터 입력받은 질문 |
| `documents` | `List[Document]` | 리트리버에 의해 검색된 관련 문서들 |
| `generation` | `str` | LLM이 생성한 최종 답변 |

**TypedDict의 장점:**
- 상태의 구조를 명확하게 정의
- IDE에서 타입 체킹 및 자동완성 지원
- 코드 가독성 향상
- 런타임 타입 검증 가능

**상태 업데이트 방식:**

```python
# 노드에서 반환된 딕셔너리는 기존 상태와 병합됨
def retrieve(state: GraphState) -> GraphState:
    return {"documents": docs}  # generation 필드는 기존값 유지

# 결과: {"question": "...", "documents": docs, "generation": ""}
```

---

### (3) 노드 함수 구현 (Retrieve, Generate)

#### 📌 학습 목표
- Retrieve 노드 구현: 질문 기반 문서 검색
- Generate 노드 구현: 검색 결과 기반 답변 생성

#### 🎓 스터디 노트 및 질문

> **Retrieve 노드 구현**

```python
def retrieve(state: GraphState) -> GraphState:
    """문서 검색 노드"""
    print("---RETRIEVE---")
    question = state["question"]
    
    # 리트리버를 사용해 관련 문서 검색
    documents = retriever.invoke(question)
    
    return {"question": question, "documents": documents, "generation": ""}
```

**역할:**
1. 사용자 질문 추출
2. 벡터 스토어에서 유사한 문서 검색 (top-k)
3. 검색된 문서를 상태에 저장

**Generate 노드 구현**

```python
def generate(state: GraphState) -> GraphState:
    """답변 생성 노드"""
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    
    # 문서를 텍스트로 포맷팅
    docs_txt = format_docs(documents)
    
    # RAG 체인을 사용해 답변 생성
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

**역할:**
1. 검색된 문서와 질문을 컨텍스트로 준비
2. LLM에 프롬프트와 함께 전달
3. 생성된 답변을 상태에 저장

---

### (4) 그래프 생성 및 컴파일

#### 📌 학습 목표
- StateGraph를 사용한 워크플로우 구성
- 노드와 엣지 연결
- 그래프 컴파일

#### 🎓 스터디 노트 및 질문

> **Basic RAG 그래프 구조**

```
START → retrieve → generate → END
```

**구현 코드:**

```python
from langgraph.graph import StateGraph, START, END

# 그래프 생성
workflow = StateGraph(GraphState)

# 노드 추가
workflow.add_node("retrieve", retrieve)
workflow.add_node("generate", generate)

# 엣지 연결
workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

# 그래프 컴파일
app = workflow.compile()
```

**각 단계의 의미:**

1. **StateGraph 생성**: GraphState를 상태로 사용하는 그래프 생성
2. **노드 추가**: retrieve, generate 함수를 노드로 등록
3. **엣지 연결**: 노드 간의 실행 순서 정의
4. **컴파일**: 그래프를 실행 가능한 형태로 변환

---

### (5) 그래프 시각화 및 실행

#### 📌 학습 목표
- 그래프 구조를 시각화하여 워크플로우 확인
- invoke와 stream 실행 방식 이해

#### 🎓 스터디 노트 및 질문

> **그래프 시각화**

```python
from IPython.display import Image, display

# Mermaid 다이어그램으로 시각화
display(Image(app.get_graph().draw_mermaid_png()))
```

**출력 예시:**
```
┌─────────────┐
│   START     │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  retrieve   │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  generate   │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│    END      │
└─────────────┘
```

> **두 가지 실행 방식**

**1. invoke() - 동기 실행**

```python
result = app.invoke({"question": "Deepseek OCR이 뭐야?"})

print(f"질문: {result['question']}")
print(f"답변: {result['generation']}")
print(f"참조 문서 수: {len(result['documents'])}")
```

**특징:**
- 전체 워크플로우가 완료될 때까지 대기
- 최종 결과만 한 번에 반환
- 동기식이므로 blocking 발생

**2. stream() - 스트리밍 실행**

```python
for output in app.stream({"question": "Deepseek OCR이 뭐야?"}):
    for node_name, value in output.items():
        print(f"[{node_name}] 노드 실행 완료")
        if node_name == "generate":
            print(f"답변: {value['generation']}")
```

**특징:**
- 각 노드 실행 결과를 순차적으로 반환
- 중간 과정을 실시간으로 확인 가능
- 장시간 작업에서 진행 상황 추적 용이

---

### (6) 실습: 질의응답 테스트

#### 📌 학습 목표
- 완성된 Basic RAG 그래프로 질의응답 테스트
- 다양한 쿼리에 대한 성능 평가

#### 🎓 스터디 노트 및 질문

> 여기에 학습하며 생긴 질문들을 기록하세요.

---

## 셋째, 실습-03 Agentic RAG 구현

### (1) Agentic RAG 개요

#### 📌 학습 목표
- Agentic RAG의 개념 이해
- Basic RAG와의 차이점 파악

#### 🎓 스터디 노트 및 질문

> **Agentic RAG란?**

**정의**: LLM 에이전트가 검색 여부와 방법을 **자율적으로 판단**하는 고급 RAG 시스템

**Basic RAG vs Agentic RAG:**

| 항목 | **Basic RAG** | **Agentic RAG** |
|------|---|---|
| **검색 결정** | 항상 검색 | LLM이 판단 |
| **검색 여부** | 고정된 흐름 | 동적 결정 |
| **문서 평가** | 없음 | 관련성 평가 |
| **질문 개선** | 없음 | 동적 재작성 |
| **유연성** | 낮음 | 높음 |
| **사용 사례** | 간단한 QA | 복잡한 에이전트 작업 |

**Agentic RAG 워크플로우:**

```
START 
  ↓
generate_query_or_respond (LLM이 검색 필요 판단)
  ├─ 검색 불필요 → END (직접 응답)
  └─ 검색 필요 → retrieve (문서 검색)
       ↓
    grade_documents (관련성 평가)
       ├─ 관련 있음 → generate_answer → END
       └─ 관련 없음 → rewrite_question → generate_query_or_respond (재시도)
```

**핵심 특징:**
- **자율적 의사결정**: LLM이 tool 호출 필요성 판단
- **관련성 평가**: 검색 결과가 질문과 관련있는지 평가
- **질문 개선**: 관련 없는 검색 결과 시 질문 재작성
- **루프 구현**: 만족스러운 결과까지 재시도

---

### (2) Retriever Tool 생성

#### 📌 학습 목표
- @tool 데코레이터를 사용한 도구 정의
- 에이전트가 호출할 수 있는 retriever tool 생성

#### 🎓 스터디 노트 및 질문

> **@tool 데코레이터 사용**

```python
from langchain.tools import tool

@tool
def retrieve(query: str) -> str:
    """DeepSeek OCR 논문에서 관련 정보를 검색합니다."""
    docs = retriever.invoke(query)
    return "\n\n".join([doc.page_content for doc in docs])

retriever_tool = retrieve
```

**동작 원리:**
1. `@tool` 데코레이터가 함수를 LangChain Tool 객체로 변환
2. 함수명이 tool의 이름이 됨 (retrieve)
3. 함수 docstring이 tool의 설명이 됨
4. 함수 인자(query)가 tool의 입력 파라미터가 됨

**tool 객체의 정보:**

```python
print(retriever_tool.name)        # "retrieve"
print(retriever_tool.description) # "DeepSeek OCR 논문에서..."
print(retriever_tool.args)        # {"query": {"type": "string"}}
```

---

### (3) 조건부 라우팅 (Conditional Routing)

#### 📌 학습 목표
- tools_condition을 사용한 동적 라우팅
- 조건부 엣지의 구현

#### 🎓 스터디 노트 및 질문

> **Tool 호출 여부에 따른 라우팅**

```python
from langgraph.prebuilt import ToolNode, tools_condition

# 노드 추가
workflow.add_node("generate_query_or_respond", generate_query_or_respond)
workflow.add_node("retrieve", ToolNode([retriever_tool]))

# 조건부 엣지: LLM이 tool 호출 여부 결정
workflow.add_conditional_edges(
    "generate_query_or_respond",
    tools_condition,
    {
        "tools": "retrieve",  # Tool 호출 시 → retrieve 노드로
        END: END              # Tool 호출 안 함 → 종료
    }
)
```

**tools_condition의 역할:**
- LLM 응답을 분석하여 tool_calls 존재 여부 확인
- tool_calls가 있으면 "tools" 반환
- tool_calls가 없으면 END 반환

**generate_query_or_respond 구현:**

```python
def generate_query_or_respond(state: MessagesState):
    """LLM이 검색 여부를 판단하여 tool 호출 또는 직접 응답"""
    response = (
        response_model
        .bind_tools([retriever_tool])  # tool 바인딩
        .invoke(state["messages"])
    )
    return {"messages": [response]}
```

---

### (4) 문서 관련성 평가

#### 📌 학습 목표
- 검색된 문서의 관련성 판단
- 조건에 따른 노드 분기 구현

#### 🎓 스터디 노트 및 질문

> **문서 관련성 평가 (Grading)**

```python
from pydantic import BaseModel, Field
from typing import Literal

class GradeDocuments(BaseModel):
    """문서 관련성 평가를 위한 이진 점수"""
    binary_score: str = Field(description="Relevance score: 'yes' or 'no'")

def grade_documents(state: MessagesState) -> Literal["generate_answer", "rewrite_question"]:
    """검색된 문서의 관련성을 평가"""
    question = state["messages"][0].content
    context = state["messages"][-1].content  # Tool의 출력
    
    prompt = f"""You are a grader assessing relevance of a retrieved document to a user question.
    Document: {context}
    Question: {question}
    If the document is relevant, score 'yes'; otherwise 'no'."""
    
    response = grader_model.with_structured_output(GradeDocuments).invoke(
        [{"role": "user", "content": prompt}]
    )
    
    if response.binary_score == "yes":
        return "generate_answer"
    else:
        return "rewrite_question"
```

**동작 방식:**
1. LLM이 문서와 질문을 분석
2. 관련성을 이진 점수로 반환 (yes/no)
3. 점수에 따라 다음 노드 결정
4. yes → 답변 생성
5. no → 질문 재작성

---

### (5) Agentic RAG 그래프 조립

#### 📌 학습 목표
- 모든 노드와 엣지를 연결한 완전한 Agentic RAG 그래프 구성
- 루프와 분기를 포함한 복잡한 워크플로우 구현

#### 🎓 스터디 노트 및 질문

> **Agentic RAG 그래프 구성**

```python
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

# 그래프 생성
workflow = StateGraph(MessagesState)

# 노드 추가
workflow.add_node("generate_query_or_respond", generate_query_or_respond)
workflow.add_node("retrieve", ToolNode([retriever_tool]))
workflow.add_node("rewrite_question", rewrite_question)
workflow.add_node("generate_answer", generate_answer)

# 시작 엣지
workflow.add_edge(START, "generate_query_or_respond")

# 조건부 엣지: Tool 호출 여부 판단
workflow.add_conditional_edges(
    "generate_query_or_respond",
    tools_condition,
    {"tools": "retrieve", END: END}
)

# 조건부 엣지: 문서 관련성 평가
workflow.add_conditional_edges("retrieve", grade_documents)

# 일반 엣지
workflow.add_edge("generate_answer", END)
workflow.add_edge("rewrite_question", "generate_query_or_respond")

# 컴파일
agentic_graph = workflow.compile()
```

**그래프 구조 분석:**

| 노드 | 입력 | 출력 | 다음 노드 |
|------|------|------|----------|
| generate_query_or_respond | 메시지 | Tool 호출 여부 | retrieve / END |
| retrieve | Tool 호출 | 문서 내용 | grade_documents |
| grade_documents | 문서 + 질문 | 관련성 판단 | generate_answer / rewrite_question |
| generate_answer | 문서 + 질문 | 답변 | END |
| rewrite_question | 질문 | 개선된 질문 | generate_query_or_respond |

---

### (6) 실습: 동적 의사결정 테스트

#### 📌 학습 목표
- Agentic RAG의 동적 의사결정 과정 확인
- 다양한 쿼리에서의 에이전트 동작 이해

#### 🎓 스터디 노트 및 질문

> **Agentic RAG 실행 및 분석**

**테스트 1: 검색이 필요한 질문**

```python
for chunk in agentic_graph.stream(
    {"messages": [{"role": "user", "content": "DeepSeek OCR이 뭐야?"}]}
):
    for node, update in chunk.items():
        print(f"🔄 Update from node: {node}")
        update["messages"][-1].pretty_print()
```

**예상 흐름:**
1. generate_query_or_respond: "검색 필요" 판단 → retrieve tool 호출
2. retrieve: 문서 검색
3. grade_documents: 문서 관련성 평가 ("yes" 반환)
4. generate_answer: 최종 답변 생성
5. END

**테스트 2: 검색이 불필요한 질문**

```python
for chunk in agentic_graph.stream(
    {"messages": [{"role": "user", "content": "안녕하세요!"}]}
):
    for node, update in chunk.items():
        print(f"🔄 Update from node: {node}")
```

**예상 흐름:**
1. generate_query_or_respond: "검색 불필요" 판단 → 직접 응답
2. END (검색 없이 종료)

**stream() 메서드의 장점:**
- 각 노드의 실행 과정을 실시간으로 관찰
- 에이전트의 "생각 과정" 추적 가능
- 문제 발생 시 어디서 오류났는지 파악 용이

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
