[← Part 1](./PART-1-README.md)

# CHAP-06 Langchain으로 구현하는 Basic RAG

## 📑 Index
- [첫째, Langchain 프레임워크](#첫째-langchain-프레임워크)
  - [(1) Langchain 개요](#1-langchain-개요)
  - [(2) Langchain 버전별 업데이트](#2-langchain-버전별-업데이트)
  - [(3) Langchain과 LangGraph 비교](#3-langchain과-langgraph-비교)

## 첫째, Langchain 프레임워크

### (1) Langchain 개요

- **등장**: 2022년 Harrison Chase에 의해 개발된 오픈소스 프레임워크
- **정의**: LLM 기반 애플리케이션 구축을 위한 통합 프레임워크
- **핵심 목표**: 복잡한 LLM 워크플로우를 모듈화하여 간편한 구현 제공
- **주요 구성 요소**: Chains(작업 흐름), Agents(자율 의사결정), Memory(대화 맥락), Tools(외부 도구 연동)
- **다중 LLM 지원**: OpenAI, Anthropic, Hugging Face 등 다양한 LLM 프로바이더 통합
- **외부 도구 연동**: 검색 엔진, 데이터베이스, API 등과 자유로운 연결
- **사용 사례**: 챗봇, Q&A 시스템, 에이전트, RAG 시스템, 데이터 분석 등
- **생태계**: 빠르게 성장하는 커뮤니티와 풍부한 플러그인 지원
- **공식 문서**: [LangChain Documentation](https://python.langchain.com/docs)

### (2) Langchain 버전별 업데이트

#### v0.0.x (2022~2023년 초)
- 기초적인 Chains, Agents, Memory 구현
- 제한된 LLM 프로바이더 지원
- 빠른 기능 추가로 인한 API 불안정성

#### v0.1.0 (2023년 중반)
- 너무 많은 의존성 문제 해결
- 모듈 구조 개편 (langchain-core 분리)
- 문서화 및 테스트 강화
- 설치 시간 및 의존성 크기 대폭 감소

#### v0.2.0 이후 (2023년 말~2024년)
- 성능 최적화 (임베딩 캐싱, 배치 처리 개선)
- LCEL (LangChain Expression Language) 도입으로 선언형 파이프라인 구성
- 더 나은 에러 처리 및 디버깅 도구
- 에코시스템 확장 (langchain-community, langchain-experimental)
- 프로덕션 환경 지원 강화 (모니터링, 로깅)

#### 현재 버전 (2024년~)
- **모듈화**: 핵심 기능과 커뮤니티 통합 분리
- **안정성**: LLM 애플리케이션 프로덕션 운영 지원
- **성능**: 비용 최적화 및 응답 속도 개선
- **확장성**: 다양한 LLM, 도구, 데이터 소스 통합

### (3) Langchain과 LangGraph 비교

#### LangChain의 특징
- **목적**: LLM 통합 프레임워크
- **워크플로우**: 선형적/순차적 실행 (Chains)
- **복잡도**: 단순하고 직관적인 구조
- **제어 흐름**: 기본 라우팅, 조건부 로직 제한적
- **사용 사례**: 챗봇, Q&A, 간단한 RAG, 기본 에이전트
- **학습 곡선**: 낮음

#### LangGraph의 특징
- **목적**: LangChain 기반 상태 머신 프레임워크
- **워크플로우**: 그래프 기반 (노드, 엣지, 루프)
- **복잡도**: 복잡한 다단계 워크플로우 표현 가능
- **제어 흐름**: 루프, 분기, 조건부 로직, 상태 관리 정밀 제어
- **사용 사례**: 복잡한 에이전트, 멀티 턴 대화, 동적 의사결정 흐름
- **학습 곡선**: 높음

#### 선택 기준
| 선택지 | 사용 시점 |
|--------|----------|
| **LangChain** | 간단한 RAG, 기본 챗봇, 빠른 프로토타이핑 필요 시 |
| **LangGraph** | 복잡한 에이전트, 정밀한 제어 흐름, 프로덕션급 시스템 필요 시 |

