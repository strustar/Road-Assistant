import os
import hashlib
import time
import re
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv

import streamlit as st

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from pinecone import Pinecone
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False

load_dotenv()

# =========================
# 페이지 설정
# =========================
st.set_page_config(
    page_title="설계실무지침 AI 검색",
    page_icon="🏗️",
    # layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# 스타일 + 자동 스크롤
# =========================
st.markdown("""
<style>
  .main { background-color: #f8f9fa; }
  .stButton > button { width: 100%; }
  .chat-message {
    padding: 1rem;
    border-radius: 12px;
    margin-bottom: 1rem;
  }
  .user-message {
    background: #e3f2fd;
    border-left: 4px solid #2196f3;
  }
  .assistant-message {
    background: #ffffff;
    border-left: 4px solid #4caf50;
    border: 1px solid #e9ecef;
  }
  .source-card {
    background: #f8f9fa;
    padding: 0.8rem;
    border-radius: 8px;
    border: 1px solid #dee2e6;
    margin: 0.5rem 0;
    font-size: 0.9rem;
  }
  .source-text {
    white-space: pre-wrap;
    font-family: 'Malgun Gothic', sans-serif;
    line-height: 1.6;
  }
  .example-btn {
    font-size: 0.85rem;
    padding: 0.3rem 0.5rem;
  }
</style>
""", unsafe_allow_html=True)

# =========================
# 설정값
# =========================
INDEX_NAME = "road"
EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIM = 1536

# =========================
# 시스템 프롬프트
# =========================
def get_system_prompt() -> str: 
    """
    [최적화 v10] 시스템 프롬프트 - 일반화 버전
    - 특정 수치/케이스 하드코딩 없음
    - 패턴 기반 규칙
    """ 
     
    base_prompt = """당신은 한국도로공사 '설계실무지침' 전문 수석 엔지니어입니다. 
제공된 RAG 컨텍스트를 분석하여 질문에 대한 정확한 답변을 제공하십시오.

## 표 출력 규칙 (매우 중요!)
1. 표는 **원본 구조 그대로** Markdown 표로 재현 (절대, 절대, 절대 변경하지 마세요. 원본 그대로 출력하세요.) Never, Never, Never change the table.
2. 다중 헤더(2단, 3단)는 **최대한 유사하게** 표현
3. 병합 셀은 반복 또는 빈칸으로 표현
4. **모든 수치, 괄호, 단위를 정확히** 유지
5. 표 내용을 인용했으면 반드시, 반드시 표 원본을 그대로 표시하세요 (상세 설명에, 반드시, 꼭)

## 🔴 절대 규칙 (Critical Rules)
0. '現', '검토배경', '현황', '문제점', '사례조사', '기존' 등은 참조용으로만(절대 결론성으로 이용하지 마세요) 사용하세요. 최종 결과는 개선(안), 변경(안) 등입니다.
1. **RAG 컨텍스트만 사용**: 외부 지식 금지. 제공된 문서 내에서만 답변.
2. **검토결과(결론) 최우선**: 문서 내에 '현황'과 '검토결과(개선안)'이 상충할 경우, 반드시 **'검토결과' 또는 '최종 결론'**을 정답으로 채택하십시오.
3. **표(Table) 절대 보존**: 문서 내의 표는 요약하지 말고, **Markdown 표 포맷을 사용하여 원본 구조 그대로** 출력하십시오. (열/행 변경 금지)
4. 관련 표가 제시되면, 그 밑에 반드시 표에 대한 요악, 설명 추가
5. **연도/부서 맞춤형**: 사용자가 특정 연도나 부서를 지정하면 해당 정보를 최우선으로 하고, 지정하지 않으면 **최신 기준**을 중심으로 **연도별 추이**를 설명하십시오.
6. **있는 연도만 비교**: 특정 연도(예: 2017)를 강제로 찾지 말고, **문서에 실제로 존재하는 연도들(예: 2015, 2019, 2023 등)** 간의 변화를 비교하십시오.
7. **없으면 솔직히**: 정보가 없으면 "🚫 제공된 문서에서 해당 정보를 찾을 수 없습니다."라고 출력하십시오.
8. 관련 있으면 모든 사항을 상세설명에서 모두 설명하세요.

## ⚖️ 정보 인용 우선순위 (Information Hierarchy)
문서 내용을 분석할 때 다음 순서대로 가중치를 두십시오:
1. **1순위 (Final Decision)**: '검토결과', '결론', '개선방안', '최종안', '적용방안', '개선', '향후계획' 섹션의 내용
2. **2순위 (Detailed Specs)**: '세부 기준', '적용 기준', '설계 기준' 등의 구체적 수치
3. **3순위 (Supporting Info)**: '現', '검토배경', '현황', '문제점', '사례조사' (이는 설명의 보조 자료로만 활용)
⚠️ **주의**: '현황'이나 '사례조사'에 나온 수치를 최종 기준으로 착각하여 답변하지 마십시오.

## 📝 출처 표기 (필수 형식)
**반드시 아래 형식을 정확히 따르세요:**
[챕터 | 제목 | 문서코드 | 날짜]

**예시:**
[설계행정 | 1-1 특정공법 심의대상 선정절차 개선방안 | 설계처-1036 | 2017.03.30]
[구조물공 | 3-2 제설염해 방지를 위한 콘크리트 구조물 표면보호재 적용 방안 | 구조물처-3819 | 2024.12.17]


## 🧠 답변 생성 프로세스

1. **질문 유형 판별**: 단순 조회? 비교/전환? 
2. **표 열 확인**: "현재" vs "적용(안)" 구분
3. **정답 추출**: "적용(안)" 열 또는 "검토결과" 섹션에서
4. **비교 질문이면**: 양쪽 조건 + 비교 + 결론
5. **표 단순화**: 복잡하면 요약 표 + 설명
6. **조건 명시**: 설계속도별 등 차이가 있으면 모두 표기
"""
 
    output_format = """ 
## 📋 출력 형식 (3단계 답변)

---
### 📖 용어 설명 (Terminology)
- **(매우 중요)** 반드시 **'사용자 질문'에 포함된 전문 용어**나, 답변 이해에 필수적인 **핵심 키워드**만 골라서 설명하십시오.
- ⚠️ **주의**: 질문에 없거나 관련 없는 일반적인 용어(예: BIM, 스마트건설, 4차산업 등)를 습관적으로 넣지 마십시오.
- **원문 우선**: 검색된 문서 안에 해당 용어의 '정의(Definition)'가 있다면, **문서의 문장을 그대로 인용**하여 적으십시오. (없을 때만 지식 활용)

---
### 📌 간단 요약 (개략 이해용)
0. '現', '검토배경', '현황', '문제점', '사례조사', '기존' 등은 참조용으로만(절대 결론성으로 이용하지 마세요) 사용하세요. 최종 결과는 개선(안), 변경(안) 등입니다.

**핵심 답변** (상세 답변 기반으로 핵심만 제시, 표는 제시하지 않음)
- 질문에 대한 **최종 기준(적용안/개선안)** 값 제시
- 비교/전환 질문이면: 기존 조건 → 적용 기준 → 결론 순서
- 조건별(설계속도, 지형 등) 값이 다르면 범위 또는 대표값 명시

[출처] : [설계행정 | 1-1 드론라이다 통합측량 확대방안 | 설계처-181 | 2024.01.16]

---
### 📖 상세 설명 (심화 이해용)
- 가독성을 위해 단락별 2줄 띄우기
- 문장별 한줄 띄우고, 출처 표시

0. '現', '검토배경', '현황', '문제점', '사례조사', '기존' 등은 참조용으로만(절대 결론성으로 이용하지 마세요) 사용하세요. 최종 결과는 개선(안), 변경(안) 등입니다.

1. 매우 상세하게 답변하세요.
- 배경, 목적, 예외사항, 관련 규정까지 포함
- 각 항목마다 출처 명기:`[챕터 | 제목 | 문서코드 | 날짜]` (구분되게, 줄바꿈해서서)

**검토결과 및 적용기준** (최우선)
- 문서의 '검토결과', '개선방안', '적용(안)' 열 내용 상세 기술
- 구체적 수치, 조건, 예외사항 포함

**배경 및 목적** (예시)
드론라이다 측량기술이 도입되면서 정확한 측량 품질 확보를 위한 명확한 기준 필요성이 대두되었습니다. 2023년 시범운영을 통해 실제 데이터를 수집하고, 이를 기반으로 2024년 구체적 수치 기준을 확립하였습니다.
([설계행정 | 1-1 드론라이다 통합측량 확대방안 | 설계처-181 | 2024.01.16])

**적용 기준 상세** (예시)
- 점밀도: **최소 400pts/㎡ 이상**
- 측정 방법: 전체 측량 구역의 평균 점밀도 산출
- 검증 절차:
  1. 시험비행 촬영 실시
  2. 점밀도 측정 및 감독 확인
  3. 기준 충족시 본 촬영 진행
  4. 기준 미달시 장비/촬영조건 변경 후 재촬영

**시범운영 결과 데이터:**
- 원본 유지, 절대 변경하지 마세요. (표 그대로 출력)


([설계행정 | 1-1 드론라이다 통합측량 확대방안 | 설계처-181 | 2024.01.16])

**연도별 변경사항** (예시)

**2017년:**
- 명확한 점밀도 기준 없음
- 일반적인 측량 정확도 기준만 존재
([설계행정 | 2-5 측량 업무처리 기준 | 설계처-892 | 2017.05.20])

**2024년 (개정):**
- **구체적 수치 기준 신설**: 400pts/㎡
- 시범운영 데이터 기반 기준 수립
- 사전 검증 절차 의무화
([설계행정 | 1-1 드론라이다 통합측량 확대방안 | 설계처-181 | 2024.01.16])

**변경 이유:** (예시시)
드론라이다 기술의 본격 도입으로 객관적이고 명확한 품질 기준이 필요해졌으며, 2023년 시범사업 결과 최소 400pts 이상이 적정하다고 판단되었습니다.

**예외 사항 및 특이사항**
- 과업지시서에 점밀도 기준을 명시해야 함
- 기준 미달시 무조건 재촬영 (예외 없음)
- 장비 성능이 기준을 만족하지 못하면 사용 불가
([설계행정 | 1-1 드론라이다 통합측량 확대방안 | 설계처-181 | 2024.01.16])

**관련 규정 및 참조**
- 「드론 활용 측량 업무처리 지침」(국토교통부)
- 「공간정보 구축 작업규정」
- 「측량·수로조사 및 지적에 관한 법률」
([설계행정 | 1-1 드론라이다 통합측량 확대방안 | 설계처-181 | 2024.01.16])

---

### 📚 참조 문서 목록
(연도별 정리)
---
""" 
    return base_prompt + output_format

# 유저 프롬프트
def get_user_prompt(query: str, context: str) -> str:
    return f"""## 🔍 사용자 질문 
        {query} 
        
        ## 📚 참조 문서 (RAG 컨텍스트) 
        아래 문서들은 **여러 연도(2014~2024)**에 걸쳐 있을 수 있습니다.
        **모든 관련 문서를 종합**하여 답변하세요.

        {context} 
        
        ## 📝 지시사항 (필수 준수)
        0. 제발, 검토, 현황, 문제점, 분석 등은 최종 결과가 아닙니다(이것은 참조용으로만). 반드시 고려해주세요. 최종 결과는 개선(안) 등입니다.
        - 현황, 現, 실태, 문제점, 기존 등은 참조용으로만(절대 결론성으로 이용하지 마세요) 사용하세요. 최종 결과는 개선(안) 등입니다.
        - 표 구조를 정확하게 읽으세요. 수치데이타 등(현재 vs 개선안 비교 시 등... 비교 되는 것을 면밀하게 분석해서 이에 대한 명확한 설명을 하세요)
        - 표는 절대 변경하지 마세요. 원본 그대로 출력하세요.

        1. **질문 키워드 확인**: 질문의 핵심 키워드가 문서에 있는지 확인

        2. **유사 개념도 활용**: 직접 언급이 없어도 유사한 개념, 관련 규정이 있으면 활용

        3. **연도별 종합 및 추이 분석**:
        - 여러 연도 문서가 있으면 **모두 참조**
        - **연도순으로 정리**: 2017년 → 2020년 → 2024년
        - 계수나 기준이 변경되었으면 **변화 추이를 명확히 표시**
        - 변경 이유나 배경도 함께 설명

        4. **부분 관련 정보도 제공**: 
        - 질문과 완전히 일치하지 않아도 **참고가 될 정보는 제공**
        - "직접적인 기준은 없으나, 관련 규정은..." 형식으로

        5. **정확한 인용**: 
        - 수치, 기준, 조건은 정확히 인용 (수식, 첨자 포함)
        - 원본 표현 그대로 유지

        6. **표 처리**: 
        - 표 내용을 인용했으면 반드시, 반드시 표 원본을 그대로 표시하세요 (상세 설명에, 반드시, 꼭)
        - 표는 절대, 절대, 절대대 변경하지 마세요. 원본 그대로 출력하세요.
        - 표가 있으면 **마크다운 표 형식 그대로** 포함
        - `<br>`, `&nbsp;` 등은 일반 텍스트로 변환
        - 표 앞뒤로 빈 줄 추가
        - 수치나 구조 절대 변경 금지

        7. **출처 명시 (필수 형식)**: 
        ```
        [챕터 | 제목 | 문서코드 | 날짜]
        ```
        - 각 정보마다 반드시 출처 표기
        - 챕터, 제목, 코드, 날짜 **모두 필수**
        - 구분자는 `|` (파이프) 사용

        8. **2단계 답변 구성**:
        - **1단계 (간단 요약)**: 핵심만 간결하게, 하지만 중요한 내용은 모두 포함
        - **2단계 (상세 설명)**: 배경, 목적, 예외사항, 관련 규정까지 포함

        9. **"없음" 최소화**: 
        - 정말로 전혀 관련 없을 때만 "찾을 수 없다"고 답변
        - 부분적이라도 관련 있으면 제공

        10. **연도별 문서 목록**:
            - 마지막에 참조 문서를 **연도별로 그룹핑**하여 정리
        """ 
                

# =========================
# 텍스트 전처리 함수
# =========================
def clean_text_for_display(text: str) -> str:
    """출처 표시용 텍스트 전처리"""
    if not text:
        return ""
    
    text = re.sub(r'<br\s*/?>', '\n', text, flags=re.IGNORECASE)
    text = re.sub(r'<sup>(.*?)</sup>', r'^(\1)', text)
    text = re.sub(r'<sub>(.*?)</sub>', r'_(\1)', text)
    
    html_entities = {
        '&nbsp;': ' ', '&lt;': '<', '&gt;': '>',
        '&amp;': '&', '&quot;': '"', '&#39;': "'",
        '&ndash;': '–', '&mdash;': '—',
    }
    for entity, char in html_entities.items():
        text = text.replace(entity, char)
    
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()


def extract_search_keywords(query: str) -> str:
    """검색 쿼리에서 핵심 키워드 추출"""
    stopwords = {
        '은', '는', '이', '가', '을', '를', '의', '에', '에서', '로', '으로',
        '와', '과', '도', '만', '까지', '부터', '마다', '처럼', '같이',
        '어떤', '무엇', '어떻게', '왜', '언제', '어디', '누가', '뭐',
        '알려줘', '알려주세요', '설명해줘', '설명해주세요', '뭐야', '뭔가요',
        '하는', '되는', '있는', '없는', '해야', '할', '한', '된', '인',
        '그', '저', '이', '것', '수', '등', '및', '또는', '그리고',
        '대해', '관해', '관한', '대한', '어떻게', '어떤', '무슨',
    }
    
    # cleaned = re.sub(r'[^\w\s가-힣]', ' ', query)
    # 🔥 보존할 기호: - . / ~ → ( ) %
    cleaned = re.sub(r"[^0-9A-Za-z가-힣\s\-\./~→()%]", " ", query)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    tokens = cleaned.split()
    keywords = [t for t in tokens if t not in stopwords and len(t) > 1]
    
    if len(keywords) < 2:
        return query
    
    return ' '.join(keywords)


# =========================
# Pinecone RAG 매니저
# =========================
class PineconeRAG:
    def __init__(self):
        self.client: OpenAI = None
        self.pc: Pinecone = None
        self.index = None
        self.namespace_map = {}
        
    def init_clients(self):
        """OpenAI와 Pinecone 클라이언트 초기화"""
        # api_key_openai = os.getenv("OPENAI_API_KEY")
        # api_key_pinecone = os.getenv("PINECONE_API_KEY")

        # 배포용 (둘 다 지원)
        def get_api_key(key_name: str) -> str:
            """Streamlit Secrets 또는 환경변수에서 API 키 가져오기"""
            # 1. Streamlit Secrets (배포 환경)
            try:
                import streamlit as st
                if key_name in st.secrets:
                    return st.secrets[key_name]
            except:
                pass
            
            # 2. 환경변수 (로컬 환경)
            return os.getenv(key_name, "")

        # 사용
        api_key_openai = get_api_key("OPENAI_API_KEY")
        api_key_pinecone = get_api_key("PINECONE_API_KEY")
        
        if not api_key_openai:
            st.error("❌ OPENAI_API_KEY가 없습니다!")
            return False
        if not api_key_pinecone:
            st.error("❌ PINECONE_API_KEY가 없습니다!")
            return False
        
        try:
            self.client = OpenAI(api_key=api_key_openai)
            self.pc = Pinecone(api_key=api_key_pinecone)
            self.index = self.pc.Index(INDEX_NAME)
            self._build_namespace_map()
            return True
        except Exception as e:
            st.error(f"❌ 클라이언트 초기화 실패: {e}")
            return False
    
    def _build_namespace_map(self):
        """실제 존재하는 namespace 조회 및 매핑"""
        try:
            stats = self.index.describe_index_stats()
            namespaces = stats.get("namespaces", {})
            
           # 🔥 자동으로 2014~2030년 모두 매칭
            known_folders = []
            for year in range(2014, 2031):
                known_folders.append(f"설계실무지침/{year}")
                known_folders.append(f"설계실무지침_{year}")
                
            for folder in known_folders:
                hash_val = hashlib.md5(folder.encode('utf-8')).hexdigest()
                if hash_val in namespaces:
                    self.namespace_map[hash_val] = folder
            
            for ns in namespaces.keys():
                if ns and ns not in self.namespace_map:
                    self.namespace_map[ns] = ns
                    
        except Exception as e:
            print(f"Namespace 매핑 실패: {e}")
    
    def get_namespaces(self) -> Dict[str, str]:
        """사용 가능한 namespace 목록"""
        try:
            stats = self.index.describe_index_stats()
            namespaces = stats.get("namespaces", {})
            
            result = {}
            for ns, info in namespaces.items():
                count = info.get("vector_count", 0)
                display_name = self.namespace_map.get(ns, ns)
                if ns == "":
                    display_name = "(기본)"
                result[f"{display_name} ({count:,}개)"] = ns
            
            return result
        except Exception as e:
            return {"(전체)": ""}
    
    def get_embedding(self, text: str) -> List[float]:
        """텍스트를 임베딩 벡터로 변환"""
        response = self.client.embeddings.create(
            input=text[:8000],
            model=EMBEDDING_MODEL,
            dimensions=EMBEDDING_DIM
        )
        return response.data[0].embedding
    
    def search(self, query: str, top_k: int = 10, 
               namespaces: List[str] = None, filters: Dict = None,
               use_keyword_extraction: bool = True) -> List[Dict]:
        """
        🔥 [검색 로직 최적화 - Diversity & Filtering]
        1. 필터링: 연도(20xx) 및 부서(OO처) 자동 추출 및 적용
        2. 점수 보정: 키워드 정확도 기반 Re-ranking
        3. 다양성 보장: 각 연도별 상위 2개 문서는 점수가 낮더라도 우선 확보 (LLM 비교 분석용)
        """
        import re
        from collections import defaultdict
        
        # ------------------------------------------------------------
        # 1. 자동 필터링 (연도 & 부서 추출)
        # ------------------------------------------------------------
        if filters is None:
            filters = {}

        # (1) 연도 추출 (2000~2030)
        year_match = re.search(r'(20[0-3]\d)', query)
        # 사용자가 사이드바에서 연도를 지정하지 않았을 때만 쿼리 기반 필터 적용
        if year_match and "year" not in filters:
            filters["year"] = int(year_match.group(1))
            print(f"🕵️‍♂️ [Auto-Filter] 연도 감지: {filters['year']}")

        # (2) 부서 추출 (설계처, 구조물처 등 'OO처' 패턴)
        # 이미 필터에 없으면 쿼리에서 찾아서 넣음
        if "dept" not in filters:
            dept_match = re.search(r'([가-힣]+처)', query)
            if dept_match:
                filters["dept"] = dept_match.group(1)
                print(f"🕵️‍♂️ [Auto-Filter] 부서 감지: {filters['dept']}")

        # ------------------------------------------------------------
        # 2. Pinecone 벡터 검색 (Wide Retrieval)
        # ------------------------------------------------------------
        search_query = extract_search_keywords(query) if use_keyword_extraction else query
        query_vector = self.get_embedding(search_query)
        
        raw_results = []
        # 다양한 연도를 확보하기 위해 top_k보다 훨씬 많이 가져옴 (최소 5배)
        fetch_k = max(top_k * 5, 50) 
        
        if not namespaces:
            try:
                stats = self.index.describe_index_stats()
                namespaces = list(stats.get("namespaces", {}).keys()) or [""]
            except: namespaces = [""]
        
        for ns in namespaces:
            try:
                search_params = {
                    "vector": query_vector,
                    "top_k": fetch_k,
                    "include_metadata": True
                }
                if ns: search_params["namespace"] = ns
                if filters: search_params["filter"] = filters # 연도/부서 필터 적용
                
                results = self.index.query(**search_params)
                
                for match in results.get("matches", []):
                    raw_results.append({
                        "id": match["id"],
                        "score": match["score"],
                        "namespace": ns,
                        "metadata": match.get("metadata", {})
                    })
            except: continue

        # ------------------------------------------------------------
        # 3. 키워드 기반 점수 부스팅 (Lexical Re-ranking)
        # ------------------------------------------------------------
        clean_q = re.sub(r"[^\w\s]", " ", query)
        query_words = [w for w in clean_q.split() if len(w) >= 2]
        
        # '개선', '적용', '표' 관련 키워드는 정답일 확률이 높으므로 추가 가산점
        bonus_keywords = ["개선", "적용", "검토", "결과", "표"]

        for doc in raw_results:
            title = doc["metadata"].get("title", "")
            text = doc["metadata"].get("text", "")
            full_text = (title * 2) + " " + text
            
            match_score = 0
            
            # (1) 질문 단어 매칭
            for word in query_words:
                if word in full_text: match_score += 0.05
            
            # (2) 구문 매칭 (강력)
            if len(query_words) >= 2:
                for i in range(len(query_words)-1):
                    phrase = f"{query_words[i]} {query_words[i+1]}"
                    if phrase in full_text: match_score += 0.3
            
            # (3) 정답 시그널 보너스
            for bk in bonus_keywords:
                if bk in full_text: match_score += 0.05

            doc["score"] += match_score
            doc["keyword_matches"] = int(match_score * 10)

        # ------------------------------------------------------------
        # 4. 🔥 [핵심] 연도별 쿼터제 적용 (Diversity Strategy)
        # ------------------------------------------------------------
        # 문서를 연도별로 그룹화
        docs_by_year = defaultdict(list)
        for doc in raw_results:
            y = doc["metadata"].get("year", "Unknown")
            docs_by_year[y].append(doc)
        
        final_results = []
        selected_ids = set()

        # (1) 각 연도별로 점수가 가장 높은 상위 2개 무조건 확보
        # 연도 역순(최신순)으로 순회
        sorted_years = sorted(docs_by_year.keys(), reverse=True)
        
        for year in sorted_years:
            # 해당 연도 문서들을 점수순 정렬
            docs_by_year[year].sort(key=lambda x: x["score"], reverse=True)
            
            # 상위 2개 추출 (있으면)
            top_2_docs = docs_by_year[year][:2]
            for doc in top_2_docs:
                final_results.append(doc)
                selected_ids.add(doc["id"])
        
        # (2) 남은 공간(top_k) 채우기
        # 전체 리스트를 다시 점수순으로 정렬하여, 아직 선택 안 된 고득점 문서 추가
        raw_results.sort(key=lambda x: x["score"], reverse=True)
        
        for doc in raw_results:
            if len(final_results) >= top_k:
                break
            if doc["id"] not in selected_ids:
                final_results.append(doc)
                selected_ids.add(doc["id"])
        
        # (3) 최종 정렬 (LLM에게는 점수 높은 순서대로 주는 것이 좋음)
        final_results.sort(key=lambda x: x["score"], reverse=True)
        
        return final_results
    
    def build_context(self, results: List[Dict], max_chunks: int = 10) -> str:
        """검색 결과를 LLM 컨텍스트로 변환"""
        context_parts = []
        
        for i, doc in enumerate(results[:max_chunks], 1):
            meta = doc["metadata"]
            score = doc["score"]
            
            code = meta.get("code", "N/A")
            date = meta.get("date", "N/A")
            title = meta.get("title", "제목 없음")
            dept = meta.get("dept", "N/A")
            year = meta.get("year", "N/A")
            category = meta.get("category", "N/A")
            
            raw_text = meta.get("text", "")
            text = clean_text_for_display(raw_text)
            
            context_parts.append(
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"📄 [문서 {i}] 유사도: {score:.4f}\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"• 코드: {code}\n"
                f"• 날짜: {date}\n"
                f"• 부서: {dept}\n"
                f"• 연도: {year}\n"
                f"• 제목: {title}\n"
                f"• 분류: {category}\n"
                f"\n[본문 내용]\n{text}\n"
            )
        
        return "\n\n".join(context_parts)
    
    def generate_response_streaming(self, query: str, context: str, 
                                     model: str, placeholder) -> str:
        """스트리밍 방식으로 LLM 응답 생성"""
        
        system_prompt = get_system_prompt()        
        user_prompt = get_user_prompt(query, context)

        try:
            stream = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                # 🔥🔥🔥 [핵심 수정: 일관성 강제 설정] 🔥🔥🔥
                temperature=0.0,  # 창의성 0 (가장 확률 높은 단어만 선택)
                top_p=0.1,        # 확률 분포 꼬리 자르기 (이상한 단어 선택 방지)
                seed=12345,       # 랜덤 시드 고정 (항상 같은 결과를 내도록 강제)
                stream=True
            )
            
            response_text = ""
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                    response_text += chunk.choices[0].delta.content
                    placeholder.markdown(response_text + "▌")
            
            placeholder.markdown(response_text)
            return response_text
            
        except Exception as e:
            placeholder.error(f"❌ LLM 호출 오류: {str(e)}")
            return ""
    
    def get_index_stats(self) -> Dict:
        """인덱스 통계 조회"""
        try:
            return self.index.describe_index_stats()
        except Exception as e:
            return {"error": str(e)}


# =========================
# 참조 문서 렌더링
# =========================
def render_source_card(doc: Dict, rank: int, msg_idx: int = 0):
    """참조 문서 카드 렌더링"""
    meta = doc["metadata"]
    score = doc["score"]
    
    code = meta.get("code", "N/A")
    title = meta.get("title", "제목 없음")
    date = meta.get("date", "")
    dept = meta.get("dept", "")
    year = meta.get("year", "")
    category = meta.get("category", "")
    
    raw_text = meta.get("text", "")
    cleaned_text = clean_text_for_display(raw_text)
    
    preview_length = 500
    preview_text = cleaned_text[:preview_length]
    has_more = len(cleaned_text) > preview_length
    
    unique_key = f"m{msg_idx}_r{rank}_{doc['id'][:8]}"
    
    # 🔥 키워드 매칭 정보 표시
    keyword_info = ""
    if "keyword_matches" in doc and doc["keyword_matches"] > 0:
        keyword_info = f" 🔍+{doc['keyword_matches']}"
    
    with st.expander(f"📄 [{rank}] {code} - {title} (유사도: {score:.4f}){keyword_info}", expanded=False):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"**코드:** `{code}`")
        with col2:
            st.markdown(f"**날짜:** {date}")
        with col3:
            st.markdown(f"**부서:** {dept}")
        with col4:
            st.markdown(f"**연도:** {year}")
        
        if category and category != "N/A":
            st.markdown(f"**분류:** {category}")
        
        st.markdown("---")
        st.markdown("**📝 본문 내용:**")
        st.markdown(preview_text)
        
        if has_more:
            show_full_key = f"show_full_{unique_key}"
            if show_full_key not in st.session_state:
                st.session_state[show_full_key] = False
            
            if st.button(f"📖 전체 보기", key=f"btn_{unique_key}"):
                st.session_state[show_full_key] = not st.session_state[show_full_key]
            
            if st.session_state[show_full_key]:
                st.markdown("---")
                st.markdown("**[전체 내용]**")
                st.markdown(cleaned_text)


def render_source_summary(results: List[Dict]):
    """참조 문서 요약 표시"""
    if not results:
        return
    
    st.markdown("**📚 참조 문서 목록:**")
    
    summary_lines = []
    for i, doc in enumerate(results, 1):
        meta = doc["metadata"]
        code = meta.get("code", "N/A")
        title = meta.get("title", "제목 없음")
        date = meta.get("date", "")
        score = doc["score"]
        
        keyword_info = ""
        if "keyword_matches" in doc and doc["keyword_matches"] > 0:
            keyword_info = f" 🔍+{doc['keyword_matches']}"
        
        summary_lines.append(f"{i}. `{code}` - {title} ({date}) [유사도: {score:.3f}]{keyword_info}")
    
    st.markdown("\n".join(summary_lines))


# =========================
# 자동 스크롤 함수
# =========================
def scroll_to_bottom():
    """JavaScript를 사용하여 페이지 하단으로 스크롤"""
    st.markdown(
        """
        <script>
            window.scrollTo({
                top: document.body.scrollHeight,
                behavior: 'smooth'
            });
        </script>
        """,
        unsafe_allow_html=True
    )


# =========================
# 메인 앱
# =========================
def main():
    st.title("🏗️ 설계실무지침 AI 검색")
    st.caption("Pinecone RAG + GPT 기반 질의응답 시스템 v3 + Lexical Re-rank")
    
    # 세션 상태 초기화
    if "rag" not in st.session_state:
        st.session_state.rag = PineconeRAG()
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "initialized" not in st.session_state:
        st.session_state.initialized = False
    if "pending_query" not in st.session_state:
        st.session_state.pending_query = None
    
    rag = st.session_state.rag
    
    # 🔥 자동 초기화 (앱 시작시 한 번만)
    if not st.session_state.initialized:
        with st.spinner("🔌 API 자동 연결 중..."):
            if rag.init_clients():
                st.session_state.initialized = True
                st.success("✅ API 연결 완료!")
            else:
                st.error("❌ API 연결 실패. 환경변수를 확인하세요.")
    
    # =========================
    # 사이드바
    # =========================
    with st.sidebar:
        st.header("⚙️ 설정")
        
        # 연결 상태
        if st.session_state.initialized:
            st.success("✅ API 연결됨")
            
            # 재연결 버튼
            if st.button("🔄 재연결"):
                with st.spinner("재연결 중..."):
                    if rag.init_clients():
                        st.success("✅ 재연결 성공!")
                    else:
                        st.error("❌ 재연결 실패")
        else:
            st.error("❌ API 미연결")
            if st.button("🔌 수동 연결", type="primary"):
                with st.spinner("연결 중..."):
                    if rag.init_clients():
                        st.session_state.initialized = True
                        st.success("✅ 연결 성공!")
                    else:
                        st.error("❌ 연결 실패")
        
        if st.session_state.initialized:
            st.divider()
            
            # 인덱스 정보
            stats = rag.get_index_stats()
            if "error" not in stats:
                total_vectors = stats.get("total_vector_count", 0)
                st.metric("📊 총 벡터 수", f"{total_vectors:,}")
                
                namespaces = stats.get("namespaces", {})
                if namespaces:
                    with st.expander("📁 Namespace 상세"):
                        for ns, info in namespaces.items():
                            display_name = rag.namespace_map.get(ns, ns) or "(기본)"
                            count = info.get("vector_count", 0)
                            st.text(f"{display_name}: {count:,}개")
            
            st.divider()
            
            # Namespace 선택
            st.subheader("📁 Namespace")
            ns_options = rag.get_namespaces()
            
            select_all = st.checkbox("전체 선택", value=True)
            
            if select_all:
                selected_namespaces = list(ns_options.values())
                st.info(f"✅ 모든 namespace에서 검색 ({len(selected_namespaces)}개)")
            else:
                selected_displays = st.multiselect(
                    "검색할 namespace",
                    options=list(ns_options.keys()),
                    default=list(ns_options.keys())[:1]
                )
                selected_namespaces = [ns_options[d] for d in selected_displays]
            
            st.divider()
            
            # LLM 설정
            st.subheader("🤖 LLM 설정")
            model_options = {
                "GPT-4o-mini (기본, 저비용)": "gpt-4o-mini",
                "GPT-4o (고품질)": "gpt-4o",
                "GPT-4-turbo": "gpt-4-turbo",
            }
            selected_model_name = st.selectbox("모델", list(model_options.keys()))
            selected_model = model_options[selected_model_name]
            
            custom_model = st.text_input("커스텀 모델명", placeholder="예: gpt-4.5-preview")
            if custom_model.strip():
                selected_model = custom_model.strip()
            
            st.divider()
            
           
            # 검색 설정
            st.subheader("🔍 검색 설정")
            top_k = st.slider("검색 결과 수", 3, 30, 10,
                             help="최종 반환 문서 수 (내부적으로 5배 많이 검색)")
            context_chunks = st.slider("LLM 컨텍스트 문서 수", 3, 30, 10,
                                       help="LLM에 전달할 최대 문서 수")
            
            use_keyword_extraction = st.checkbox("키워드 추출 사용", value=True,
                                                  help="불용어 제거로 검색 품질 향상")
            
            st.divider()
            
            # 필터 설정
            st.subheader("🏷️ 필터 (선택)")
            filter_dept = st.text_input("부서", placeholder="예: 설계처")
            filter_year = st.number_input("연도 (0=전체)", min_value=0, max_value=2030, value=0)
            
            st.divider()
            
            # 대화 초기화
            if st.button("🗑️ 대화 초기화"):
                st.session_state.messages = []
                st.session_state.pending_query = None
                st.rerun()
            
       
    # -------------------------
    # 예제 질문 데이터 (연도별)
    # -------------------------
    EXAMPLES_BY_YEAR = {
        2024: [
            "라이다 측량을 다시 해야하는 경우는?",
            "설계 하자책임기간 관리는 어떤 부서?",
            "주민설명회에서 BIM 활용 방법은?",
            "타당성 및 기본설계 기간은?",
            "내리막 좌측 곡선부 곡선반경은?",
            "가도, 가교 설치 기준을 알려줘",
            "지하차도 배수 수방체계 표준안은?",
            "제설염해 위험구간을 알려줘",
            "나들목 중앙분리대 방호등급은?",
        ],
        2023: [
            "고속도로 건설 관련 부담금은?",
            "하이패스 나들목 설계 기간은?",
            "안전관리비 간접공사비 적용은?",
            "배수성 포장 적용 대상 구간은?",
            "여굴량 산출기준을 알려줘",
            "터널 폐수처리시설 계상기준은?",
            "설계 안전성 검토 방법은?",
            "설계단계별 주민설명회 시기는?",
            "지적중첩도 작성 의뢰 시기는?",
            "확장구간 내 제한속도는?",
            "점검 승강시설 설치 기준은?",
        ],
        2017: [
            "경관설계 대상 시설물을 알려줘",
            "축중차로를 스마트톨링 차로로 전환시 설계기준은?",
            "고속도로 설계시 설계강우강도를 어떻게 적용해야 하는지 알려줘",
            "교량 고정식 점검시설 설치기준을 알려줘",
            "교면 배수구 설치간격 기준을 알려줘",
            "교량하부 가드휀스 설치 기준은?",
            "하이브리드 거더의 정의를 알려줘",
            "콘크리트 포장 줄눈 설치 기준을 알려줘",
            "터널 요철포장 설치 기준을 알려줘",
        ],
    }

    # 탭 순서(원하는 연도만 넣으면 됨)
    YEARS = [2024, 2023, 2022, 2021, 2020, 2019, 2018, 2017]

    st.divider()
    st.sidebar.markdown("### 💡 예제 질문")

    for year in [2024, 2023, 2022, 2021, 2020, 2019, 2018, 2017]:
        with st.sidebar.expander(f"📅 {year}년", expanded=(year == 2017)):
            examples = EXAMPLES_BY_YEAR.get(year, [])
            c1, c2 = st.columns(2)
            for i, q in enumerate(examples):
                with (c1 if i % 2 == 0 else c2):
                    if st.button(q, key=f"ex_{year}_{i}", use_container_width=True):
                        st.session_state.pending_query = q


        
    # =========================
    # 메인 영역
    # =========================
    if not st.session_state.initialized:
        st.info("⏳ API 연결 중입니다. 잠시만 기다려주세요...")
        st.stop()
    
    # =========================
    # 대화 히스토리 표시
    # =========================
    for msg_idx, msg in enumerate(st.session_state.messages):
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.markdown(msg["content"])
        else:
            with st.chat_message("assistant"):
                st.markdown(msg["content"])
                if "sources" in msg and msg["sources"]:
                    with st.expander("📚 참조 문서", expanded=False):
                        render_source_summary(msg["sources"])
                        st.markdown("---")
                        for i, doc in enumerate(msg["sources"], 1):
                            render_source_card(doc, i, msg_idx=msg_idx)
    
    # =========================
    # 🔥 입력 처리 (pending_query 우선)
    # =========================
    query = None
    
    # 1. pending_query가 있으면 사용
    if st.session_state.pending_query:
        query = st.session_state.pending_query
        st.session_state.pending_query = None
    
    # 2. chat_input 항상 표시
    chat_query = st.chat_input("질문을 입력하세요...")
    if chat_query and not query:
        query = chat_query
    
    # =========================
    # 질의응답 처리
    # =========================
    if query:
        # 사용자 메시지 추가
        st.session_state.messages.append({"role": "user", "content": query})
        
        with st.chat_message("user"):
            st.markdown(query)
        
        # 어시스턴트 응답
        with st.chat_message("assistant"):
            # 필터 구성
            filters = {}
            if filter_dept:
                filters["dept"] = filter_dept
            if filter_year > 0:
                filters["year"] = filter_year
            
            # 검색
            with st.spinner("🔍 문서 검색 중... (50개 검색 → 재정렬)"):
                t0 = time.time()
                results = rag.search(
                    query=query,
                    top_k=top_k,
                    namespaces=selected_namespaces,
                    filters=filters if filters else None,
                    use_keyword_extraction=use_keyword_extraction
                )
                search_time = time.time() - t0
            
            if not results:
                st.warning("⚠️ 관련 문서를 찾을 수 없습니다.")
                response = "검색 결과가 없습니다. 다른 질문을 시도해주세요."
                used_sources = []
            else:
                # 🔥 키워드 매칭 정보 표시
                keyword_boosted = sum(1 for doc in results if doc.get("keyword_matches", 0) > 0)
                st.caption(f"✅ {len(results)}개 문서 검색 완료 ({search_time:.2f}초) | 키워드 부스팅: {keyword_boosted}개")
                
                # 컨텍스트 구성
                context = rag.build_context(results, max_chunks=context_chunks)
                
                # LLM 응답 생성
                placeholder = st.empty()
                response = rag.generate_response_streaming(
                    query, context, selected_model, placeholder,
                )
                
                # 참조 문서 표시
                with st.expander("📚 참조 문서", expanded=False):
                    render_source_summary(results)
                    st.markdown("---")
                    current_msg_idx = len(st.session_state.messages)
                    for i, doc in enumerate(results, 1):
                        render_source_card(doc, i, msg_idx=current_msg_idx)
                
                used_sources = results
        
        # 메시지 저장
        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "sources": used_sources
        })
        
        # 🔥 자동 스크롤 (답변 후)
        scroll_to_bottom()
        
        # 🔥 즉시 리렌더링 (입력창 유지)
        st.rerun()


# =========================
# 실행
# =========================
if __name__ == "__main__":
    if not OPENAI_AVAILABLE:
        st.error("❌ openai 패키지 미설치")
        st.code("pip install openai python-dotenv")
        st.stop()
    if not PINECONE_AVAILABLE:
        st.error("❌ pinecone 패키지 미설치")
        st.code("pip install pinecone")
        st.stop()
    
    main()