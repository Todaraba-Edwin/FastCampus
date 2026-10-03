[← Part 1](./PART-1-README.md)

# CHAP-09 AI 에이전트가 적용된 서비스와 그 원리

## 📑 Index
- [첫째, ChatGPT (Operator & Dots)](#첫째-chatgpt-operator--dots)
  - [(1) 서비스 개요](#1-서비스-개요)
  - [(2) B2C 능력과 사용자 기반](#2-b2c-능력과-사용자-기반)
  - [(3) 에이전트 진화: Operator → Dots](#3-에이전트-진화-operator--dots)
- [둘째, Google Gemini](#둘째-google-gemini)
  - [(1) 서비스 개요](#1-서비스-개요-1)
  - [(2) 주요 특징](#2-주요-특징)
- [셋째, Claude](#셋째-claude)
  - [(1) 서비스 개요](#1-서비스-개요-2)
  - [(2) 주요 특징](#2-주요-특징-1)
  - [(3) 에이전트 역량 확장: MCP & Skills](#3-에이전트-역량-확장-mcp--skills)
- [넷째, Meta (Muse & Manus)](#넷째-meta-muse--manus)
  - [(1) Meta의 이중 전략](#1-meta의-이중-전략)
  - [(2) Muse: B2C 개인용 에이전트](#2-muse-b2c-개인용-에이전트)
  - [(3) Manus AI: 파일 시스템 기반 기업용 에이전트](#3-manus-ai-파일-시스템-기반-기업용-에이전트)

---

## 첫째, ChatGPT (Operator & Dots)

### (1) 서비스 개요

- 2025년 1월 Operator 출시 → 7월 ChatGPT에 통합
- 컴퓨터 기반 작업 자동화: 캘린더 관리, 파일 생성, 코드 실행 등 완전 자동 처리
- Plus, Pro, Business 사용자 대상 (Plus 사용자부터)

### (2) B2C 능력과 사용자 기반

- **자연어 기반 쇼핑 에이전트**: 사용자의 예산, 품질, 배송 속도 등 선호도를 자연스럽게 이해
- **실시간 데이터 활용**: 웹에서 가격, 리뷰, 재고 정보 자동 수집 및 비교
- **사용자를 잘 아는 기업**: 수억 명 사용자 기반의 학습 데이터로 일상 생활 패턴 파악

### (3) 에이전트 진화: Operator → Dots

- **Operator (2025년 1월)**: 사용자 명령 대기 방식, 가상 머신 기반 컴퓨터 작업 자동화
- **Dots (최신)**: GPT-6 Astra 기반 persistent agent로 진화, 사용자 없어도 계속 작업 (4,000+ 앱 연동)
- 점진적 학습: 사용자 선호도와 업무 습관 학습, Slack/Teams 등 업무 도구와 통합

## 둘째, Google Gemini

### (1) 서비스 개요

- Agent Mode: 사용자가 목표를 설정하면 자동으로 멀티스텝 작업 실행
- 예시: 예산 내 아파트 검색 → 비교 → 일정 예약 등 전체 프로세스 자동 처리
- Google 앱(Calendar, Gmail 등) 및 웹 검색 기반 깊이 있는 작업 지원

### (2) 주요 특징

- Code Assist Agent Mode: 자동 코드 변경 전 계획 검토 프로세스 제공
- Enterprise Agent Platform: 7일 연속 실행 가능한 엔터프라이즈급 에이전트
- Visual Flow Builder: 복잡한 멀티스텝 에이전트 직관적으로 설계 및 관리

## 셋째, Claude

### (1) 서비스 개요

- Claude Code (2025년 2월 출시): 자연어 기반 코딩 작업 위임 CLI 도구
- Claude Agent SDK (2025년 9월): Claude Code와 동일한 에이전트 인프라 공개 (에이전트 루프, 권한 관리 등)
- Claude Opus 4.5 (2025년 11월): 코딩, 에이전트, 컴퓨터 사용 최적화 모델

### (2) 주요 특징

- 개발자 중심: 코드 실행, MCP 커넥터, Files API 등 개발 환경 통합
- 프롬프트 캐싱: 최대 1시간 캐싱으로 비용 효율성 및 속도 향상
- Opus 4.5: 코딩, 에이전트, 컴퓨터 사용 최적화

### (3) 에이전트 역량 확장: MCP & Skills

- **MCP (Model Context Protocol)**: GitHub API, Playwright 브라우저, Brave 검색 등 외부 도구 자동 연동, 필요한 도구 스스로 판단
- **Claude Skills (2025년 10월)**: 지시사항, 스크립트, 리소스 모음으로 동적 로드, Progressive Disclosure로 컨텍스트 최적화
- **에이전트 완성도 UP**: 2025년 12월 Agent Skills를 개방 표준화 → OpenAI, Google, GitHub, Cursor 채택으로 산업 표준 확립

## 넷째, Meta (Muse & Manus)

### (1) Meta의 이중 전략

- **Muse (B2C)**: 개인 사용자의 일상 작업 자동화 (개인용 에이전트)
- **Manus (B2B)**: Meta가 2025년 12월 $2B에 인수, 기업용 복잡 업무 자동화
- **통합 목표**: WhatsApp Business에 Manus 통합으로 AI가 항공권 재예약, 결제 처리 등 거래 직접 처리

### (2) Muse: B2C 개인용 에이전트

- 채팅 인터페이스 기반, 웹 브라우싱, 구매, 이미지 생성, 문서 작성 등 종합 작업 지원
- 능동적 작업: 사용자 요청 전 자동으로 리마인더, 목표 추적, 모니터링 수행
- 무료 버전 + $20/월, $100/월 구독 옵션으로 접근성 높음

### (3) Manus AI: 파일 시스템 기반 기업용 에이전트

- **파일 시스템 패러다임**: todo.md 파일로 진행 상황 추적, 중간 결과 파일에 저장해 장기 기억 관리
- **장기 작업 관리**: Scheduled Tasks로 일일/주간/월간 반복 자동화, 인간 개입 최소화
- **기업 능력**: 시장 조사, 코드 개발, 데이터 분석 등 복잡한 비즈니스 작업 자동 처리
