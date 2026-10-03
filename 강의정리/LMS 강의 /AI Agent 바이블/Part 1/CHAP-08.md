[← Part 1](./PART-1-README.md)

# CHAP-08 AI 에이전트의 개념과 원리

## 📑 Index
- [첫째, AI 에이전트는 LLM을 중심으로 하는 일종의 지능형 시스템](#첫째-ai-에이전트는-llm을-중심으로-하는-일종의-지능형-시스템)
  - [(1) 일반 대화형 AI와 AI 에이전트](#1-일반-대화형-ai와-ai-에이전트)
- [둘째, AI 에이전트의 핵심: 인지 → 추론 → 행동 → 결과 관찰](#둘째-ai-에이전트의-핵심-인지--추론--행동--결과-관찰)
  - [(1) 인지 (Perception)](#1-인지-perception)
  - [(2) 추론 (Reasoning)](#2-추론-reasoning)
  - [(3) 행동 (Action)](#3-행동-action)
  - [(4) 결과 관찰 (Observation)](#4-결과-관찰-observation)
- [셋째, ChatGPT의 사례로 보기](#셋째-chatgpt의-사례로-보기)
  - [(1) ChatGPT 3.5 (2022년 11월)](#1-chatgpt-35-2022년-11월)
  - [(2) ChatGPT 4.0 (2023년 3월)](#2-chatgpt-40-2023년-3월)
  - [(3) ChatGPT 4.0 Turbo (2023년 11월)](#3-chatgpt-40-turbo-2023년-11월)
  - [(4) ChatGPT 4o (2024년 5월)](#4-chatgpt-4o-2024년-5월)
  - [(5) ChatGPT o1 (2024년 9월)](#5-chatgpt-o1-2024년-9월)
  - [(6) ChatGPT 6.1](#6-chatgpt-61)
- [넷째, Function Calling은 AI 에이전트 활성화의 핵심](#넷째-function-calling은-ai-에이전트-활성화의-핵심)
  - [(1) Function Calling의 의미](#1-function-calling의-의미)
  - [(2) LangChain에서의 Function Calling](#2-langchain에서의-function-calling)
  - [(3) LangGraph에서의 Function Calling](#3-langgraph에서의-function-calling)
  - [(4) Function Calling의 실행 흐름](#4-function-calling의-실행-흐름)

---

## 첫째, AI 에이전트는 LLM을 중심으로 하는 일종의 지능형 시스템

### (1) 일반 대화형 AI와 AI 에이전트

| 항목 | 일반 대화형 AI | AI 에이전트 |
|-----|---|---|
| 작동 방식 | 사용자 입력 → 즉시 응답 | 사용자 지시 → 자동 작업 수행 |
| 의사결정 | 문맥 기반 응답 | 스스로 계획 수립 및 실행 |
| Tool 사용 | 불가능 또는 제한적 | 필요에 따라 주도적으로 활용 |
| 반복 처리 | 사용자가 매번 명령 | 종료 조건까지 자동 반복 |
| 독립성 | 사용자에 의존적 | 자율적 작업 수행 |
| 에러 처리 | 사용자가 처리 | 자동 오류 수정 및 재시도 |

## 둘째, AI 에이전트의 핵심: 인지 → 추론 → 행동 → 결과 관찰

### (1) 인지 (Perception)
- 사용자의 요청과 현재 상황 파악
- 환경 정보 수집 및 컨텍스트 분석

### (2) 추론 (Reasoning)
- 현재 상황을 기반으로 다음 단계 계획 수립
- 목표 달성을 위한 전략 및 의사결정

### (3) 행동 (Action)
- 계획된 작업을 실행 또는 필요한 도구 호출
- 외부 API 또는 시스템과의 상호작용

### (4) 결과 관찰 (Observation)
- 행동의 결과 수집 및 분석
- 목표 달성 여부 판단 및 다음 단계 결정

## 셋째, ChatGPT의 사례로 보기

### (1) ChatGPT 3.5 (2022년 11월)
- 할루시네이션(Hallucination): 사실이 아닌 내용을 마치 사실인 것처럼 생성
- 논리 및 추론 능력 부족으로 인한 부정확하거나 비논리적인 응답
- **에이전트 능력 평가**: 자율성 전무, 사용자 입력에만 반응하는 단순 대화형 AI 수준

### (2) ChatGPT 4.0 (2023년 3월)
- 추론 능력의 획기적 향상으로 더욱 정확하고 논리적인 응답 생성
- 자율적 의사결정과 문제해결 능력으로 AI 에이전트로 불릴 만한 수준 도달
- **에이전트 능력 평가**: 에이전트 기본 갖춤, 사용자 지시를 이해하고 단계별 계획 수립 후 자율적 문제해결 가능

### (3) ChatGPT 4.0 Turbo (2023년 11월)
- 컨텍스트 윈도우 128K 토큰으로 확대, 에이전트가 더 많은 정보 동시 처리 가능
- 더 빠른 응답 속도로 실시간 작업 수행 능력 향상
- **에이전트 능력 평가**: 작업 범위 확대, 대량의 정보를 동시 처리해 복잡한 장기 프로젝트 추진 가능

### (4) ChatGPT 4o (2024년 5월)
- 음성, 영상, 텍스트 멀티모달 지원으로 인지 능력 대폭 확장
- 다양한 형식의 정보 이해와 분석 능력 획득
- **에이전트 능력 평가**: 감각 확장, 텍스트·음성·영상을 통합으로 인식해 다차원적 상황 판단 능력 획득

### (5) ChatGPT o1 (2024년 9월)
- 추론 능력 특화로 복잡한 문제 해결 능력 극대화
- 논리적 단계별 사고로 더욱 정교한 에이전트 역할 수행
- **에이전트 능력 평가**: 사고 깊이 강화, 단계별 추론을 통해 논리적 복잡성 높은 고난도 문제까지 해결 가능

### (6) 최신 버전: MCP 지원 (2024년~2025년)
- Model Context Protocol (MCP) 지원으로 외부 도구·데이터 직접 연동 가능
- 에이전트가 외부 시스템을 자율적으로 활용하는 완전한 자동화 수준 도달
- **에이전트 능력 평가**: 행동 능력 극대화, 외부 도구·API·데이터베이스를 직접 활용해 현실의 업무를 자동 수행하는 완전 자동화 수준 도달

## 넷째, Function Calling은 AI 에이전트 활성화의 핵심

### (1) Function Calling의 의미

- **정의**: LLM이 특정 도구나 함수를 스스로 호출하도록 지시하는 메커니즘
- **에이전트 활성화**: Function Calling 없이는 에이전트가 외부 세계와 상호작용 불가능
- **자율성의 핵심**: 사용자가 명령하지 않아도 필요한 도구를 스스로 판단하고 호출
- **예시**: "날씨 알려줘" → LLM이 날씨 API를 자동으로 호출 → 결과 반환

**왜 중요한가?**
- 단순 대화형 AI: 정해진 문맥 범위 내에서만 응답
- AI 에이전트: Function Calling으로 필요한 도구를 동적으로 호출하며 문제 해결

### (2) LangChain에서의 Function Calling

#### @tool 데코레이터와 create_agent (CHAP-06 예시)

```python
# 도구 정의 - @tool 데코레이터 사용
from langchain.tools import tool

@tool
def retrieve_context(query: str) -> str:
    """문서에서 관련 정보를 검색합니다."""
    docs = retriever.invoke(query)
    return "\n\n".join([doc.page_content for doc in docs])

# 에이전트 생성 - 도구 바인딩
from langchain.agents import create_tool_calling_agent

tools = [retrieve_context]
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)

# 실행 - 에이전트가 자율적으로 retrieve_context 호출
result = agent_executor.invoke({
    "input": "DeepSeek OCR이 뭐야?"
})
```

**LangChain의 Function Calling 프로세스:**

1. **도구 등록**: @tool 데코레이터로 함수를 도구로 변환
2. **에이전트 생성**: create_tool_calling_agent()로 도구 바인딩
3. **LLM 판단**: 사용자 질문 분석 → retrieve_context 필요성 판단
4. **자동 호출**: LLM이 Tool Call 명령 생성 → AgentExecutor가 실행
5. **결과 통합**: 도구의 반환값을 LLM에 다시 전달 → 최종 답변 생성

### (3) LangGraph에서의 Function Calling

#### bind_tools와 ToolNode (CHAP-07 예시)

```python
# 도구 정의
from langchain.tools import tool

@tool
def retrieve(query: str) -> str:
    """벡터 스토어에서 관련 문서를 검색합니다."""
    docs = retriever.invoke(query)
    return "\n\n".join([doc.page_content for doc in docs])

# LLM에 도구 바인딩
response_model = ChatOpenAI("gpt-4o-mini", temperature=0)
response_model_with_tools = response_model.bind_tools([retrieve])

# 노드 정의 - LLM이 Function Call 생성
def generate_query_or_respond(state: MessagesState):
    response = response_model_with_tools.invoke(state["messages"])
    return {"messages": [response]}

# Tool 실행 노드
from langgraph.prebuilt import ToolNode
retrieve_node = ToolNode([retrieve])

# 그래프 구성
workflow = StateGraph(MessagesState)
workflow.add_node("generate_query_or_respond", generate_query_or_respond)
workflow.add_node("retrieve", retrieve_node)
workflow.add_conditional_edges(
    "generate_query_or_respond",
    tools_condition,  # Tool Call 여부 판단
    {"tools": "retrieve", END: END}
)
```

**LangGraph의 Function Calling 프로세스:**

1. **도구 바인딩**: bind_tools()로 LLM에 사용 가능한 도구 정보 전달
2. **LLM 판단**: 메시지 분석 → Tool Call 필요 여부 결정
3. **Tool Call 생성**: LLM이 구조화된 Tool Call 명령 생성
4. **조건부 라우팅**: tools_condition으로 Tool Call 존재 확인
5. **ToolNode 실행**: 실제 도구 함수 실행 후 결과 반환
6. **상태 업데이트**: 도구의 반환값을 MessagesState에 추가

**LangChain vs LangGraph 비교:**

| 항목 | LangChain | LangGraph |
|------|-----------|-----------|
| 도구 정의 | @tool 데코레이터 | @tool 데코레이터 (동일) |
| 바인딩 방식 | create_tool_calling_agent | bind_tools() |
| 실행 흐름 | 선형 (순차 실행) | 그래프 (조건부/루프 가능) |
| 상태 관리 | 암묵적 | 명시적 (MessagesState) |
| 루프 처리 | 제한적 | 유연함 (조건부 엣지) |

### (4) Function Calling의 실행 흐름

**일반적인 에이전트 사이클:**

```
사용자 입력
    ↓
[1] 인지: LLM이 사용자 질문 분석
    ↓
[2] 판단: 어떤 도구를 호출할지 결정
    ↓
[3] Function Call 생성: LLM이 도구명 + 파라미터 결정
    ↓
[4] 도구 실행: 실제 함수/API 호출
    ↓
[5] 결과 관찰: 도구의 반환값 수집
    ↓
[6] 재사고: LLM이 결과를 바탕으로 다음 단계 판단
    ↓
[7] 반복 또는 종료: 추가 Function Call 필요 vs 최종 응답 생성
    ↓
최종 답변 반환
```

**실제 예시 흐름 (CHAP-07 Agentic RAG):**

```
사용자: "DeepSeek OCR이 뭐야?"
    ↓
[1] generate_query_or_respond 노드
    - LLM 판단: retrieve 도구 필요
    - Tool Call 생성: retrieve(query="DeepSeek OCR")
    ↓
[2] ToolNode에서 retrieve 실행
    - 벡터 스토어 검색
    - 관련 문서 반환
    ↓
[3] grade_documents로 문서 관련성 평가
    - "yes" → generate_answer로 진행
    - "no" → rewrite_question으로 질문 개선 후 재시도
    ↓
[4] generate_answer 노드
    - 검증된 문서 + 질문으로 최종 답변 생성
    ↓
"DeepSeek OCR은 고해상도 입력에서 긴 문서의 텍스트를 효율적으로 인식하는 OCR 시스템입니다..."
```

**Function Calling이 없다면:**
- LLM이 내 학습 데이터에만 의존
- "모르겠습니다"로 답변 불가
- 실시간 정보 활용 불가
- 외부 시스템과 연동 불가

**Function Calling으로 활성화되는 능력:**
- ✅ 외부 도구 자동 호출
- ✅ 실시간 정보 검색
- ✅ 데이터베이스 쿼리
- ✅ API 연동
- ✅ 자율적 의사결정
- ✅ 복잡한 작업 자동화
