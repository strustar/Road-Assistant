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

# 🔥 추가
try:
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    from pinecone import Pinecone
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False

# 🔥 Cohere Reranker
try:
    import cohere
    COHERE_AVAILABLE = True
except ImportError:
    COHERE_AVAILABLE = False

load_dotenv()

# =========================
# 페이지 설정
# =========================
st.set_page_config(
    page_title="설계실무지침 AI 검색",
    page_icon="🏗️",
    # layout="wide",  — CSS로 중앙 폭 조정
    initial_sidebar_state="expanded"
)

# =========================
# 스타일 + 자동 스크롤
# =========================
st.markdown("""
<style>
  /* ===== 중앙 영역 폭 확대 (Claude 스타일) ===== */
  .block-container {
    max-width: 900px !important;  /* 기본 730px → 900px */
    padding-left: 2rem !important;
    padding-right: 2rem !important;
  }
  
  /* ===== 글자 크기 조정 ===== */
  .stMarkdown, .stChatMessage {
    font-size: 0.95rem;
    line-height: 1.7;
  }
  
  /* ===== 전체 레이아웃 ===== */
  .stButton > button { width: 100%; }
  
  /* ===== 검색/형제 배지 ===== */
  .badge-search {
    display: inline-block;
    background: #2196f3;
    color: white;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 0.75rem;
    font-weight: 600;
  }
  .badge-sibling {
    display: inline-block;
    background: #ff9800;
    color: white;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 0.75rem;
    font-weight: 600;
  }
  .badge-llm {
    display: inline-block;
    background: #4caf50;
    color: white;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 0.75rem;
    font-weight: 600;
  }
  
  /* ===== 검색 결과 상단 요약 바 ===== */
  .search-summary-bar {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    color: white;
    padding: 12px 16px;
    border-radius: 10px;
    margin: 8px 0;
    display: flex;
    gap: 16px;
    flex-wrap: wrap;
    align-items: center;
    font-size: 0.85rem;
  }
  .search-summary-bar .stat {
    display: flex;
    align-items: center;
    gap: 4px;
  }
  .search-summary-bar .stat-num {
    font-weight: 700;
    font-size: 1.1rem;
  }
  
  /* ===== 문서 그룹 카드 ===== */
  .doc-group-card {
    background: #f8f9fa;
    border: 1px solid #e0e0e0;
    border-left: 4px solid #2196f3;
    border-radius: 8px;
    padding: 12px 16px;
    margin: 8px 0;
  }
  .doc-group-card.has-sibling {
    border-left-color: #ff9800;
  }
  .doc-group-title {
    font-weight: 600;
    font-size: 0.95rem;
    color: #1a1a2e;
  }
  .doc-group-meta {
    font-size: 0.8rem;
    color: #666;
    margin-top: 4px;
  }
  .chunk-pills {
    display: flex;
    flex-wrap: wrap;
    gap: 4px;
    margin-top: 8px;
  }
  .chunk-pill {
    display: inline-flex;
    align-items: center;
    gap: 3px;
    padding: 2px 10px;
    border-radius: 16px;
    font-size: 0.75rem;
    font-family: monospace;
    font-weight: 500;
  }
  .chunk-pill.search { background: #e3f2fd; color: #1565c0; border: 1px solid #90caf9; }
  .chunk-pill.sibling { background: #fff3e0; color: #e65100; border: 1px solid #ffcc80; }
  .chunk-pill.llm-yes { border-width: 2px; }
  .chunk-pill.llm-no { opacity: 0.5; }

  /* ===== 사이드바 개선 ===== */
  section[data-testid="stSidebar"] .stMetric {
    background: #f0f2f6;
    border-radius: 8px;
    padding: 8px 12px;
  }
  
  /* ===== 출처 표기 컬러 ===== */
  .source-ref {
    color: #1565c0;
    font-size: 0.85rem;
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
# 시스템 프롬프트 (1회 호출 — 용어 + 요약)
# =========================
def get_system_prompt() -> str:
    return """당신은 한국도로공사 '설계실무지침' 전문 수석 엔지니어입니다. 
제공된 RAG 컨텍스트를 분석하여 질문에 대한 정확한 답변을 제공하십시오.

## 표 출력 규칙
1. 표는 **원본 구조 그대로** Markdown 표로 재현 (절대 변경 금지)
2. 다중 헤더는 최대한 유사하게 표현
3. **모든 수치, 괄호, 단위를 정확히** 유지
4. **취소선(~~) 절대 사용 금지**: "기존: A → 개선: B" 형식
5. **<br> 태그 사용 금지**: 줄바꿈은 실제 엔터 사용

## 절대 규칙
- '現', '현황', '문제점', '기존' 등은 참조용. 최종 결과는 **개선(안), 변경(안)**.
- **RAG 컨텍스트만 사용** (외부 지식 금지)
- **검토결과(결론) 최우선**: '현황'과 '검토결과'가 상충하면 **검토결과**가 정답
- 없으면 "🚫 해당 정보를 찾을 수 없습니다."

## 정보 인용 우선순위
1순위: '검토결과', '결론', '개선방안', '적용방안'
2순위: '세부 기준', '설계 기준' 등 구체적 수치
3순위: '현황', '사례조사' (보조 자료)

## 출처 표기 (필수)
[챕터 | 제목 | 문서코드 | 날짜]

## 📋 출력 형식

### 📖 용어 설명
- 질문에 포함된 전문 용어만 간단히 설명 (1~3개)

---
### 📌 핵심 답변
- **최종 기준(적용안/개선안)** 값을 명확하게 제시
- 관련 표가 있으면 원본 그대로 포함하고 간단 설명 추가
- 비교 질문이면: 기존 → 개선 → 결론
- 연도별 변경이 있으면 간결하게 추이 정리
- 조건별(설계속도, 지형 등) 값이 다르면 명시
- 배경/목적/예외사항도 간결하게 포함
- 각 항목마다 인라인 출처: ([챕터 | 제목 | 코드 | 날짜])
- ⚠️ 참조 문서 목록은 출력하지 마세요 (UI에서 별도 표시)
"""

def get_user_prompt(query: str, context: str) -> str:
    return f"""## 🔍 사용자 질문 
{query} 

## 📚 참조 문서 (RAG 컨텍스트) 
아래 문서들은 **여러 연도(2014~2024)**에 걸쳐 있을 수 있습니다.

{context} 

## 📝 지시사항
- 현황/문제점은 참조용, 최종 결과는 개선(안)
- 표는 원본 그대로 출력
- 여러 연도 문서가 있으면 연도순 정리
- 각 정보마다 출처 표기: [챕터 | 제목 | 코드 | 날짜]
- 참조 문서 목록은 출력하지 마세요
""" 
                

# =========================
# 텍스트 전처리 함수
# =========================
def clean_text_for_display(text: str) -> str:
    """출처 표시용 텍스트 전처리 — <br> 및 HTML 태그 완전 제거"""
    if not text:
        return ""
    
    # 🔥 <br> 변환 (다양한 패턴)
    text = re.sub(r'<br\s*/?>', '\n', text, flags=re.IGNORECASE)
    
    # <sup>, <sub> 변환
    text = re.sub(r'<sup>(.*?)</sup>', r'^(\1)', text)
    text = re.sub(r'<sub>(.*?)</sub>', r'_(\1)', text)
    
    # HTML 엔티티 변환
    html_entities = {
        '&nbsp;': ' ', '&lt;': '<', '&gt;': '>',
        '&amp;': '&', '&quot;': '"', '&#39;': "'",
        '&ndash;': '–', '&mdash;': '—',
    }
    for entity, char in html_entities.items():
        text = text.replace(entity, char)
    
    # 🔥 나머지 HTML 태그 모두 제거
    text = re.sub(r'<[^>]+>', '', text)
    
    # 과도한 줄바꿈 정리
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
        self.client: OpenAI = None  # 임베딩용
        self.anthropic_client: Anthropic = None  # 🔥 추가 (LLM용)
        self.cohere_client = None  # 🔥 Reranker용
        self.pc: Pinecone = None
        self.index = None
        self.namespace_map = {}
        
    def init_clients(self):
        """OpenAI, Anthropic, Pinecone 클라이언트 초기화"""
        
        # API 키 가져오기 함수
        def get_api_key(key_name: str) -> str:
            """Streamlit Secrets 또는 환경변수에서 API 키 가져오기"""
            # 1. Streamlit Secrets (배포 환경)
            try:
                if key_name in st.secrets:
                    return st.secrets[key_name]
            except:
                pass
            
            # 2. 환경변수 (로컬 환경)
            return os.getenv(key_name, "")
        
        # API 키들 가져오기
        api_key_openai = get_api_key("OPENAI_API_KEY")
        api_key_anthropic = get_api_key("ANTHROPIC_API_KEY")
        api_key_pinecone = get_api_key("PINECONE_API_KEY")
        api_key_cohere = get_api_key("COHERE_API_KEY")  # 🔥 Reranker
        
        # 🔥 디버깅: 키 존재 확인
        print(f"🔑 OpenAI Key: {bool(api_key_openai)}")
        print(f"🔑 Anthropic Key: {bool(api_key_anthropic)}")
        print(f"🔑 Pinecone Key: {bool(api_key_pinecone)}")
        print(f"🔑 Cohere Key: {bool(api_key_cohere)}")
        
        # API 키 검증
        if not api_key_openai:
            st.error("❌ OPENAI_API_KEY가 없습니다!")
            return False
        if not api_key_anthropic:
            st.error("❌ ANTHROPIC_API_KEY가 없습니다!")
            return False
        if not api_key_pinecone:
            st.error("❌ PINECONE_API_KEY가 없습니다!")
            return False
        
        # 클라이언트 초기화
        try:
            # OpenAI 초기화
            self.client = OpenAI(api_key=api_key_openai)
            print("✅ OpenAI 클라이언트 초기화 성공")
            
            # 🔥 Anthropic 초기화
            self.anthropic_client = Anthropic(api_key=api_key_anthropic)
            print("✅ Anthropic 클라이언트 초기화 성공")
            
            # Pinecone 초기화
            self.pc = Pinecone(api_key=api_key_pinecone)
            self.index = self.pc.Index(INDEX_NAME)
            print("✅ Pinecone 클라이언트 초기화 성공")
            
            # 🔥 Cohere Reranker 초기화 (선택적 - 없어도 동작)
            if COHERE_AVAILABLE and api_key_cohere:
                self.cohere_client = cohere.ClientV2(api_key=api_key_cohere)
                print("✅ Cohere Reranker 초기화 성공")
            else:
                print("⚠️ Cohere 미설정 - 키워드 부스팅으로 대체")
            
            self._build_namespace_map()
            
            return True
            
        except Exception as e:
            st.error(f"❌ 클라이언트 초기화 실패: {e}")
            print(f"상세 에러: {e}")
            import traceback
            print(traceback.format_exc())
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
        # ⚠️ "단부처리", "처리", "처분" 등 오탐 방지: 처 뒤에 한글이 오면 부서가 아님
        if "dept" not in filters:
            dept_match = re.search(r'([가-힣]+처)(?![가-힣])', query)
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
                
                # Pinecone SDK 버전 호환 (dict / object 모두 대응)
                matches = getattr(results, 'matches', None) or results.get("matches", [])
                for match in matches:
                    match_id = getattr(match, 'id', None) or match.get("id", "")
                    match_score = getattr(match, 'score', 0) or match.get("score", 0)
                    match_meta = getattr(match, 'metadata', {}) or match.get("metadata", {})
                    # metadata가 dict가 아닐 수 있으므로 변환
                    if not isinstance(match_meta, dict):
                        match_meta = dict(match_meta) if match_meta else {}
                    raw_results.append({
                        "id": match_id,
                        "score": match_score,
                        "namespace": ns,
                        "metadata": match_meta
                    })
            except Exception as e:
                print(f"⚠️ [Search Error] namespace={ns}, error={e}")
                continue

        # 🔥 디버깅 로그
        print(f"🔍 [Search Debug] query='{query}', search_query='{search_query}'")
        print(f"🔍 [Search Debug] filters={filters}, namespaces_count={len(namespaces)}")
        print(f"🔍 [Search Debug] raw_results_count={len(raw_results)}")
        if raw_results:
            print(f"🔍 [Search Debug] top3: {[(r['score'], r['metadata'].get('title','')[:30]) for r in sorted(raw_results, key=lambda x: x['score'], reverse=True)[:3]]}")

        # ------------------------------------------------------------
        # 3. 🔥 Cohere Reranker 또는 키워드 부스팅 (Fallback)
        # ------------------------------------------------------------
        if self.cohere_client and raw_results:
            try:
                # (A) Reranker 적용
                docs_text = [doc["metadata"].get("text", "")[:1000] for doc in raw_results]
                
                rerank_resp = self.cohere_client.rerank(
                    model="rerank-multilingual-v3.0",
                    query=query,
                    documents=docs_text,
                    top_n=min(len(raw_results), top_k * 3)  # 넉넉하게 확보
                )
                
                # rerank 결과로 점수 교체
                reranked = []
                for r in rerank_resp.results:
                    doc = raw_results[r.index].copy()
                    doc["original_score"] = doc["score"]
                    doc["score"] = r.relevance_score  # Cohere 점수로 교체
                    doc["keyword_matches"] = 0  # reranker 사용시 불필요
                    reranked.append(doc)
                
                raw_results = reranked
                print(f"🔄 [Reranker] {len(reranked)}개 재정렬 완료")
                if reranked:
                    top3_info = [(f"{r['score']:.4f}", r['metadata'].get('title','')[:30]) for r in reranked[:3]]
                    print(f"🔄 [Reranker] top3: {top3_info}")
                
            except Exception as e:
                print(f"⚠️ [Reranker Error] {e} → 키워드 부스팅으로 대체")
                # Fallback: 기존 키워드 부스팅
                self._keyword_boosting(raw_results, query)
        else:
            # Cohere 미설정시 기존 키워드 부스팅
            self._keyword_boosting(raw_results, query)

        # ------------------------------------------------------------
        # 4. 🔥 형제 청크 확장 (Sibling Expansion)
        #    - 같은 문서(예: p6-2)의 다른 파트를 자동 포함
        #    - 검토결론 등 핵심 청크 누락 방지
        # ------------------------------------------------------------
        # 상위 top_k개에 'search' 태그
        for doc in raw_results[:top_k]:
            doc["source_type"] = "search"
        
        found_prefixes = set()
        for doc in raw_results[:top_k]:
            cid = doc["metadata"].get("chunk_id", "")
            if "_p" in cid:
                prefix = cid.rsplit("_p", 1)[0]
                found_prefixes.add(prefix)
        
        # 전체 raw_results에서 형제 청크 찾기
        sibling_docs = []
        top_ids = {doc["id"] for doc in raw_results[:top_k]}
        
        for doc in raw_results[top_k:]:
            cid = doc["metadata"].get("chunk_id", "")
            if "_p" in cid:
                prefix = cid.rsplit("_p", 1)[0]
                if prefix in found_prefixes and doc["id"] not in top_ids:
                    doc["source_type"] = "sibling"
                    sibling_docs.append(doc)
                    top_ids.add(doc["id"])
        
        if sibling_docs:
            print(f"🔗 [Sibling] {len(sibling_docs)}개 형제 청크 추가: {[d['metadata'].get('chunk_id','') for d in sibling_docs]}")

        # ------------------------------------------------------------
        # 5. 최종 결과 조합
        # ------------------------------------------------------------
        final_results = raw_results[:top_k] + sibling_docs
        
        # 중복 제거
        seen_ids = set()
        deduped = []
        for doc in final_results:
            if doc["id"] not in seen_ids:
                seen_ids.add(doc["id"])
                deduped.append(doc)
        final_results = deduped
        
        # 최종 정렬: 같은 문서는 파트 순서대로 모으기
        def sort_key(doc):
            cid = doc["metadata"].get("chunk_id", "")
            if "_p" in cid:
                prefix, part = cid.rsplit("_p", 1)
                try:
                    part_num = int(part)
                except:
                    part_num = 0
            else:
                prefix = cid
                part_num = 0
            return (-doc["score"], prefix, part_num)
        
        final_results.sort(key=sort_key)
        
        search_count = sum(1 for d in final_results if d.get("source_type") == "search")
        sibling_count = len(sibling_docs)
        print(f"📊 [Final] {len(final_results)}개 반환 (검색: {search_count}, 형제: {sibling_count})")
        
        return final_results
    
    def _keyword_boosting(self, raw_results: List[Dict], query: str):
        """키워드 기반 점수 부스팅 (Cohere 미사용시 Fallback)"""
        clean_q = re.sub(r"[^\w\s]", " ", query)
        query_words = [w for w in clean_q.split() if len(w) >= 2]
        
        bonus_keywords = ["개선", "적용", "검토", "결과", "표"]

        for doc in raw_results:
            title = doc["metadata"].get("title", "")
            text = doc["metadata"].get("text", "")
            full_text = (title * 2) + " " + text
            
            match_score = 0
            
            for word in query_words:
                if word in full_text: match_score += 0.05
            
            if len(query_words) >= 2:
                for i in range(len(query_words)-1):
                    phrase = f"{query_words[i]} {query_words[i+1]}"
                    if phrase in full_text: match_score += 0.3
            
            for bk in bonus_keywords:
                if bk in full_text: match_score += 0.05

            doc["score"] += match_score
            doc["keyword_matches"] = int(match_score * 10)
        
        raw_results.sort(key=lambda x: x["score"], reverse=True)
    
    def build_context(self, results: List[Dict], max_chunks: int = 15) -> str:
        """검색 결과를 LLM 컨텍스트로 변환 (br 태그 완전 제거)"""
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
            chunk_id = meta.get("chunk_id", "")
            source_type = doc.get("source_type", "search")
            tag = "🔍검색" if source_type == "search" else "🔗형제"
            
            raw_text = meta.get("text", "")
            # 🔥 <br> 및 HTML 태그 완전 제거
            text = clean_text_for_display(raw_text)
            
            context_parts.append(
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"📄 [문서 {i}] ({tag}) {chunk_id} | 유사도: {score:.4f}\n"
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
        """GPT 또는 Claude로 스트리밍 응답 생성"""
        
        system_prompt = get_system_prompt()
        user_prompt = get_user_prompt(query, context)

        try:
            response_text = ""
            
            if model.startswith("claude"):
                with self.anthropic_client.messages.stream(
                    model=model,
                    max_tokens=10000,
                    temperature=0.0,
                    system=system_prompt,
                    messages=[
                        {"role": "user", "content": user_prompt}
                    ]
                ) as stream:
                    for text in stream.text_stream:
                        response_text += text
                        placeholder.markdown(response_text + "▌")
            
            else:
                stream = self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.0,
                    stream=True
                )
                
                for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                        response_text += chunk.choices[0].delta.content
                        placeholder.markdown(response_text + "▌")
            
            # 🔥 LLM 응답에서 <br> 태그 제거
            response_text = re.sub(r'<br\s*/?>', '\n', response_text, flags=re.IGNORECASE)
            
            placeholder.markdown(response_text)
            return response_text
            
        except Exception as e:
            import traceback
            error_msg = traceback.format_exc()
            placeholder.error(f"❌ LLM 호출 오류: {str(e)}")
            print(f"상세 에러:\n{error_msg}")
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
    """참조 문서 카드 렌더링 (컬러 배지)"""
    meta = doc["metadata"]
    score = doc["score"]
    
    code = meta.get("code", "N/A")
    title = meta.get("title", "제목 없음")
    date = meta.get("date", "")
    dept = meta.get("dept", "")
    year = meta.get("year", "")
    chunk_id = meta.get("chunk_id", "")
    source_type = doc.get("source_type", "search")
    
    type_icon = "🔍" if source_type == "search" else "🔗"
    type_label = "검색" if source_type == "search" else "형제확장"
    
    raw_text = meta.get("text", "")
    cleaned_text = clean_text_for_display(raw_text)
    
    preview_length = 500
    preview_text = cleaned_text[:preview_length]
    has_more = len(cleaned_text) > preview_length
    
    unique_key = f"m{msg_idx}_r{rank}_{doc['id'][:8]}"
    
    with st.expander(
        f"{type_icon} [{rank}] {chunk_id} — {title} | {code} ({date}) | {score:.3f}",
        expanded=False
    ):
        # 배지 HTML
        badge_class = "badge-search" if source_type == "search" else "badge-sibling"
        st.markdown(
            f'<span class="{badge_class}">{type_label}</span> '
            f'<span class="badge-llm">LLM 전달</span> '
            f'&nbsp; 부서: {dept} | 연도: {year} | 유사도: {score:.4f}',
            unsafe_allow_html=True
        )
        
        st.markdown("---")
        st.markdown(preview_text)
        
        if has_more:
            show_full_key = f"show_full_{unique_key}"
            if show_full_key not in st.session_state:
                st.session_state[show_full_key] = False
            
            if st.button(f"📖 전체 보기", key=f"btn_{unique_key}"):
                st.session_state[show_full_key] = not st.session_state[show_full_key]
            
            if st.session_state[show_full_key]:
                st.markdown("---")
                st.markdown(cleaned_text)


def render_source_summary(results: List[Dict], max_chunks: int = 15):
    """참조 문서 요약 — 문서별 그룹핑, 컬러 배지, LLM 전달 여부"""
    if not results:
        return
    
    search_count = sum(1 for d in results if d.get("source_type") == "search")
    sibling_count = sum(1 for d in results if d.get("source_type") == "sibling")
    llm_count = min(len(results), max_chunks)
    
    # 상단 요약 바
    st.markdown(f"""
    <div class="search-summary-bar">
        <div class="stat">🔍 <span class="stat-num">{search_count}</span> 검색</div>
        <div class="stat">🔗 <span class="stat-num">{sibling_count}</span> 형제 확장</div>
        <div class="stat">📤 <span class="stat-num">{llm_count}</span> LLM 전달</div>
        <div class="stat">📄 <span class="stat-num">{len(results)}</span> 총 문서</div>
    </div>
    """, unsafe_allow_html=True)
    
    # 문서별 그룹핑
    from collections import OrderedDict
    doc_groups = OrderedDict()
    
    for i, doc in enumerate(results, 1):
        meta = doc["metadata"]
        code = meta.get("code", "N/A")
        title = meta.get("title", "제목 없음")
        group_key = f"{code}|{title}"
        
        if group_key not in doc_groups:
            doc_groups[group_key] = {
                "code": code,
                "title": title,
                "date": meta.get("date", ""),
                "dept": meta.get("dept", ""),
                "chunks": [],
                "best_score": 0,
                "has_sibling": False,
            }
        
        chunk_id = meta.get("chunk_id", "")
        source_type = doc.get("source_type", "search")
        is_llm = i <= max_chunks
        score = doc["score"]
        
        if source_type == "sibling":
            doc_groups[group_key]["has_sibling"] = True
        
        doc_groups[group_key]["chunks"].append({
            "chunk_id": chunk_id,
            "source_type": source_type,
            "is_llm": is_llm,
            "score": score,
        })
        
        if score > doc_groups[group_key]["best_score"]:
            doc_groups[group_key]["best_score"] = score
    
    # 문서별 카드 출력
    for gkey, group in doc_groups.items():
        chunks = group["chunks"]
        
        # 청크 pills HTML
        pills_html = ""
        for c in chunks:
            css_type = "search" if c["source_type"] == "search" else "sibling"
            css_llm = "llm-yes" if c["is_llm"] else "llm-no"
            icon = "🔍" if c["source_type"] == "search" else "🔗"
            llm_icon = "✅" if c["is_llm"] else ""
            pills_html += f'<span class="chunk-pill {css_type} {css_llm}">{llm_icon}{icon} {c["chunk_id"]}</span>'
        
        border_class = "has-sibling" if group["has_sibling"] else ""
        
        st.markdown(f"""
        <div class="doc-group-card {border_class}">
            <div class="doc-group-title">{group['code']} — {group['title']}</div>
            <div class="doc-group-meta">
                📅 {group['date']} &nbsp;|&nbsp; 🏢 {group['dept']} &nbsp;|&nbsp; 
                ⭐ 최고 유사도: {group['best_score']:.3f} &nbsp;|&nbsp; 
                📦 청크 {len(chunks)}개
            </div>
            <div class="chunk-pills">{pills_html}</div>
        </div>
        """, unsafe_allow_html=True)


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
    st.caption("Pinecone RAG + Claude/GPT + Reranker + 형제 청크 확장")
    
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
                "⭐ Sonnet 4.6 — $3/$15 | 한국어최고 | RAG최적": "claude-sonnet-4-6",
                "💰 GPT-5.2 — $1.75/$14 | 가성비왕 | 영어강점": "gpt-5.2",
                "🪙 Haiku 4.5 — $1/$5 | 한국어양호 | 빠름": "claude-haiku-4-5-20251001",
                "🪙 GPT-5 mini — $0.25/$2 | 초저가 | 단순RAG": "gpt-5-mini",
                "🆓 GPT-5 nano — $0.05/$0.4 | 최저가 | 분류전용": "gpt-5-nano",
                "🪙 GPT-4o mini — $0.15/$0.6 | 구형저가 | 안정적": "gpt-4o-mini",
            }
            selected_model_name = st.selectbox("모델", list(model_options.keys()))
            selected_model = model_options[selected_model_name]
            
            custom_model = st.text_input("커스텀 모델명", placeholder="예: claude-sonnet-4-20250514")
            if custom_model.strip():
                selected_model = custom_model.strip()
            
            # 💰 비용·성능 요약
            st.caption("📊 **100명×10건/일 월비용 추정**")
            st.markdown("""
            <div style="font-size:0.72rem;line-height:1.7;color:#999;">
            ⭐ <b>Sonnet 4.6</b> ~₩67만 | 한국어·지시따르기·할루시네이션 최소<br>
            💰 <b>GPT-5.2</b> ~₩43만 | Sonnet급 성능, 42%저렴<br>
            🪙 <b>Haiku 4.5</b> ~₩24만 | 간단 RAG 충분, 한국어 양호<br>
            🪙 <b>GPT-5 mini</b> ~₩9만 | 영어 중심, 한국어 보통<br>
            🆓 <b>GPT-5 nano</b> ~₩2만 | 분류·라우팅 전용<br>
            ─────────<br>
            <span style="color:#4caf50;">💡 한국어 설계지침 RAG → Sonnet 4.6 추천<br>
            💡 예산 절감 → Haiku 4.5가 차선책</span>
            </div>
            """, unsafe_allow_html=True)
            
            st.divider()
            
           
            # 검색 설정
            st.subheader("🔍 검색 설정")
            top_k = st.slider("검색 결과 수", 3, 30, 15,
                             help="최종 반환 문서 수 (+ 형제 청크 자동 확장)")
            context_chunks = st.slider("LLM 컨텍스트 문서 수", 3, 30, 15,
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
                    with st.expander(f"📚 참조 문서 ({len(msg['sources'])}개)", expanded=False):
                        render_source_summary(msg["sources"], max_chunks=context_chunks)
                        st.markdown("---")
                        for i, doc in enumerate(msg["sources"][:context_chunks], 1):
                            render_source_card(doc, i, msg_idx=msg_idx)
    
    # =========================
    # 입력 처리
    # =========================
    query = None
    
    if st.session_state.pending_query:
        query = st.session_state.pending_query
        st.session_state.pending_query = None
    
    chat_query = st.chat_input("질문을 입력하세요...")
    if chat_query and not query:
        query = chat_query
    
    # =========================
    # 질의응답 처리
    # =========================
    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        
        with st.chat_message("user"):
            st.markdown(query)
        
        with st.chat_message("assistant"):
            filters = {}
            if filter_dept:
                filters["dept"] = filter_dept
            if filter_year > 0:
                filters["year"] = filter_year
            
            reranker_label = "Reranker" if getattr(rag, 'cohere_client', None) else "키워드 부스팅"
            with st.spinner("🔍 문서 검색 중..."):
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
                search_count = sum(1 for d in results if d.get("source_type") == "search")
                sibling_count = sum(1 for d in results if d.get("source_type") == "sibling")
                llm_count = min(len(results), context_chunks)
                st.caption(
                    f"✅ 총 {len(results)}개 ({search_time:.2f}초) | "
                    f"🔍{search_count} + 🔗{sibling_count} | "
                    f"📤LLM: {llm_count}개 | {reranker_label}"
                )
                
                context = rag.build_context(results, max_chunks=context_chunks)
                
                placeholder = st.empty()
                response = rag.generate_response_streaming(
                    query, context, selected_model, placeholder,
                )
                
                # 참조 문서 (context_chunks개 통일)
                with st.expander(f"📚 참조 문서 ({len(results)}개)", expanded=False):
                    render_source_summary(results, max_chunks=context_chunks)
                    st.markdown("---")
                    current_msg_idx = len(st.session_state.messages)
                    for i, doc in enumerate(results[:context_chunks], 1):
                        render_source_card(doc, i, msg_idx=current_msg_idx)
                
                used_sources = results
        
        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "sources": used_sources,
        })
        
        scroll_to_bottom()
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