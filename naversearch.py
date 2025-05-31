import streamlit as st
import requests
import os
from dotenv import load_dotenv
import openai
from openai import OpenAI
import json 
from collections import Counter
import re
import time
import functools

# .env 파일에서 환경 변수 로드
load_dotenv()

# API 자격 증명 로드 (환경 변수 또는 .env 파일 사용)
NAVER_CLIENT_ID = os.getenv("NAVER_CLIENT_ID")
NAVER_CLIENT_SECRET = os.getenv("NAVER_CLIENT_SECRET")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- 설정 파일 관리 (prompt_config.json) ---
CONFIG_FILE = "prompt_config.json"

# 기본 프롬프트 템플릿
DEFAULT_PROMPT_TEMPLATE = """
SEO 최적화된 블로그 포스트를 작성해 주세요. 독자가 흥미를 느끼고 정보를 얻을 수 있도록 다음 가이드라인을 엄격히 준수하세요.

**키워드:** {keyword}
**대상 독자:** {target_audience}

**콘텐츠 구조 (마크다운 형식으로 작성):**
1.  **매력적인 제목 (H1)**: 키워드를 포함하고, 독자의 클릭을 유도하는 강력한 제목을 만드세요.
2.  **키워드 중심 서론**: {keyword}의 중요성, 이 글에서 다룰 내용 등을 명확하고 흥미롭게 제시하세요. 독자의 문제점을 언급하고 해결책을 제시하는 방식으로 시작합니다.
3.  **상세 본문 (하위 섹션)**: 
    -   최소 3개 이상의 H2 소제목으로 섹션을 나누세요.
    -   각 H2 섹션 아래에는 H3 소제목을 활용하여 내용을 더욱 세분화할 수 있습니다.
    -   각 문단은 100-150단어 내외로 간결하게 작성하고, {keyword} 및 관련 키워드를 자연스럽게 통합하세요.
    -   독자에게 유용한 정보, 구체적인 팁, 실제 사례 등을 포함하세요.
    -   (예시: "{keyword}란 무엇인가?", "{keyword}를 잘 활용하는 5가지 팁", "{keyword} 시 주의할 점")
4.  **실용적인 결론**: 본문의 내용을 요약하고, 독자가 취할 수 있는 다음 행동이나 얻을 수 있는 이점을 강조하세요.
5.  **Q&A 넣기**: {keyword}와 관련된 자주 묻는 질문 2~3개와 답변을 추가하여 독자의 궁금증을 해소하고 체류 시간을 늘리세요.
6.  **콜 투 액션 (CTA)**: 독자가 특정 행동(예: 관련 서비스 이용, 추가 정보 검색, 댓글 작성 등)을 유도하는 문구를 추가하세요.
7.  **태그 삽입 (쉼표로 연결)**: 글의 내용을 대표하는 관련 태그를 5~10개 정도 쉼표로 연결하여 마지막에 제시하세요.

**최적화 요구사항:**
-   키워드 '{keyword}' 및 관련 확장 키워드를 콘텐츠 전반에 자연스럽게 통합하되, 스터핑은 절대 금지.
-   문단당 100-150단어 (약 200-300자) 내외로 작성.
-   명확하고 간결한 문장 구조를 사용하고, 쉽게 이해할 수 있는 어휘 선택.
-   전문성과 신뢰감을 전달하는 어조를 유지.
-   읽기 쉽도록 목록(리스트), 굵은 글씨, 강조 등을 적절히 사용.
-   이미지/미디어는 텍스트로 '![이미지 설명](이미지_URL_또는_placeholder)' 형태로 표현하고, 이미지 설명을 SEO 친화적으로 작성.

**금지사항:**
-   키워드 스터핑 금지.
-   과도한 전문 용어 사용 자제 (대상 독자에 맞게 조절).
-   중복 콘텐츠 방지 (새로운 관점과 정보 제공).
-   '글을 생성했습니다'와 같은 메타 발언 금지. 오직 블로그 글 내용만 출력합니다.
"""
DEFAULT_MODEL_NAME = "gpt-4o"

def load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"prompt_template": DEFAULT_PROMPT_TEMPLATE, "openai_model_name": DEFAULT_MODEL_NAME}

def save_config(config_data):
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(config_data, f, ensure_ascii=False, indent=4)

# 앱 시작 시 설정 로드
app_config = load_config()
if 'custom_prompt_template' not in st.session_state:
    st.session_state.custom_prompt_template = app_config["prompt_template"]
if 'openai_model_name' not in st.session_state:
    st.session_state.openai_model_name = app_config["openai_model_name"]


# --- OpenAI API 키 설정 및 클라이언트 초기화 ---
client = None 
if OPENAI_API_KEY:
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
    except Exception as e:
        st.error(f"OpenAI 클라이언트 초기화 오류: {e}. .env 파일의 OPENAI_API_KEY를 확인해주세요.")
        client = None 
else:
    st.error("OpenAI API 키가 설정되지 않았습니다. .env 파일을 확인해주세요.")


# --- 1. 네이버 블로그 검색 기능 ---
@st.cache_data(ttl=3600)
def search_naver_blogs(keyword: str, display: int = 30) -> list:
    """
    네이버 블로그 검색 API를 사용하여 상위 N개의 블로그 포스트를 검색합니다.
    Args:
        keyword (str): 검색할 키워드.
        display (int): 검색 결과로 가져올 포스트 수 (최대 100).
    Returns:
        list: 각 포스트의 제목, URL, 요약(description)을 포함하는 딕셔너리 리스트.
    """
    if not NAVER_CLIENT_ID or not NAVER_CLIENT_SECRET:
        st.error("네이버 클라이언트 ID 또는 시크릿이 설정되지 않았습니다. .env 파일을 확인해주세요.")
        return []

    url = "https://openapi.naver.com/v1/search/blog.json"
    headers = {
        "X-Naver-Client-Id": NAVER_CLIENT_ID,
        "X-Naver-Client-Secret": NAVER_CLIENT_SECRET
    }
    params = {
        "query": keyword,
        "display": min(max(1, display), 100),
        "sort": "sim"
    }

    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        search_results = response.json()

        posts = []
        for item in search_results.get("items", []):
            posts.append({
                "title": item.get("title", "").replace("<b>", "").replace("</b>", ""),
                "link": item.get("link", ""),
                "description": item.get("description", "").replace("<b>", "").replace("</b>", "")
            })
        return posts
    except requests.exceptions.RequestException as e:
        st.error(f"네이버 블로그 검색 중 오류 발생: {e}")
        return []
    except json.JSONDecodeError:
        st.error("네이버 API 응답을 디코딩하는 데 실패했습니다. 응답 형식을 확인하세요.")
        return []

# --- 2. SEO 최적화 분석 (제목 리스트 기반) ---
def analyze_blog_titles(titles: list) -> dict:
    """
    주어진 블로그 제목 리스트를 분석하여 SEO 최적화 관점의 특징을 추출합니다.
    Args:
        titles (list): 분석할 블로그 제목 문자열 리스트.
    Returns:
        dict: 5가지 항목별 분석 결과와 제안 제목 10개.
    """
    if not titles:
        return {
            "structural_features": "분석할 제목이 없습니다.",
            "core_keywords_expressions": "분석할 제목이 없습니다.",
            "composition_patterns": "분석할 제목이 없습니다.",
            "attention_techniques": "분석할 제목이 없습니다.",
            "seo_optimization_features": "분석할 제목이 없습니다.",
            "new_titles": []
        }

    # 1. 제목의 구조적 특징
    total_length = sum(len(title) for title in titles)
    avg_length = total_length / len(titles) if titles else 0
    punctuation_count = Counter(char for title in titles for char in title if char in "?!.")
    tone_analysis = Counter()
    for title in titles:
        if '?' in title:
            tone_analysis['질문형'] += 1
        elif '!' in title:
            tone_analysis['감탄형'] += 1
        else:
            tone_analysis['서술형'] += 1

    structural_features = f"""
    - 평균 제목 길이: 약 {avg_length:.1f}자
    - 문장부호 사용 (상위 3개): {', '.join(f'{p}: {c}' for p, c in punctuation_count.most_common(3))}
    - 어투 분석: {', '.join(f'{t}: {c}개' for t, c in tone_analysis.most_common())}
    - 전반적으로 간결하거나 핵심 정보를 명확히 제시하는 경향이 있습니다.
    """

    # 2. 자주 사용되는 핵심 키워드와 표현
    all_words = []
    for title in titles:
        cleaned_title = re.sub(r'[^가-힣a-zA-Z0-9\s]', ' ', title)
        all_words.extend(cleaned_title.split())

    stopwords = {'은', '는', '이', '가', '을', '를', '에', '에서', '와', '과', '의', '더', '좀', '수', '할', '있는', '입니다', '합니다', '을까', '것', '으로', '들'}
    filtered_words = [word for word in all_words if len(word) > 1 and word not in stopwords]
    word_counts = Counter(filtered_words)

    bigrams = Counter()
    for title in titles:
        cleaned_title = re.sub(r'[^가-힣a-zA-Z0-9\s]', ' ', title)
        words = cleaned_title.split()
        for i in range(len(words) - 1):
            if words[i] not in stopwords and words[i+1] not in stopwords:
                bigrams[(words[i], words[i+1])] += 1

    core_keywords_expressions = f"""
    - 자주 사용되는 핵심 키워드 (상위 10개): {', '.join(f'{w}: {c}' for w, c in word_counts.most_common(10))}
    - 자주 사용되는 표현 (상위 5개): {', '.join(f'{" ".join(exp)}: {c}' for exp, c in bigrams.most_common(5))}
    """

    # 3. 제목 구성의 패턴
    patterns = Counter()
    for title in titles:
        if re.search(r'\d+[가지|개|방법|단계|팁]|TOP\s*\d+|베스트\s*\d+', title, re.IGNORECASE):
            patterns['리스트/순위형'] += 1
        if '?' in title or '무엇일까' in title or '어떻게' in title or '방법은' in title:
            patterns['질문형'] += 1
        if '후기' in title or '내돈내산' in title or '경험' in title or '솔직' in title:
            patterns['후기/경험형'] += 1
        if '꿀팁' in title or '필수템' in title or '정리' in title or '완벽가이드' in title:
            patterns['정보/가이드형'] += 1
        if '!' in title or '놀라운' in title or '최고의' in title or '강력추천' in title:
            patterns['감탄/강조형'] += 1
        
        top_keywords_for_pattern_check = [word for word, count in word_counts.most_common(5)]
        if any(kw in title for kw in top_keywords_for_pattern_check):
            patterns['키워드 선두 배치'] += 1
        else:
            patterns['일반 서술형'] += 1
    
    composition_patterns = f"""
    - 가장 흔한 패턴: {', '.join(f'{p}: {c}개' for p, c in patterns.most_common(3))}
    - 리스트형, 질문형, 정보/가이드형 제목이 정보 전달과 호기심 유발에 많이 활용됩니다.
    """

    # 4. 독자의 관심을 끌기 위한 기법
    attention_methods = Counter()
    for title in titles:
        if re.search(r'\d+', title):
            attention_methods['숫자 활용'] += 1
        if any(k in title for k in ['꿀팁', '필수', '진짜', '놀라운', '효과적인', '인생템']):
            attention_methods['가치/감성적 표현'] += 1
        if any(k in title for k in ['지금', '즉시', '놓치지']):
            attention_methods['긴급성/시의성'] += 1
        if any(k in title for k in ['비밀', '숨겨진', '궁금증', '파헤치기']):
            attention_methods['호기심 자극'] += 1
        if any(k in title for k in ['초보', '초보자', '왕초보', '완전정복']):
            attention_methods['타겟 명확화'] += 1
    
    attention_techniques = f"""
    - 숫자 활용 ({attention_methods['숫자 활용']}회): 정보의 명확성과 구체성을 제공합니다. (예: '5가지 꿀팁')
    - 가치/감성적 표현 ({attention_methods['가치/감성적 표현']}회): 독자의 문제 해결이나 욕구를 자극합니다. (예: '인생템', '효과적인')
    - 호기심 자극 ({attention_methods['호기심 자극']}회): 미지의 정보에 대한 궁금증을 유발합니다. (예: '숨겨진 비밀')
    - 타겟 명확화 ({attention_methods['타겟 명확화']}회): 특정 독자층에게 '이 글은 당신을 위한 것!'임을 어필합니다.
    """

    # 5. 제목의 SEO 최적화 특징
    seo_features = Counter()
    for title in titles:
        first_word_keywords = [word for word, count in word_counts.most_common(5)]
        if title.split() and any(kw in title.split()[0] for kw in first_word_keywords):
            seo_features['키워드 전면 배치'] += 1
        if len(title) >= 15 and len(title) <= 30:
             seo_features['적정 길이 유지'] += 1
        if any(k in title for k in ['방법', '추천', '종류', '정리', '가이드']):
            seo_features['정보성/탐색 의도 반영'] += 1
        if any(k in title for k in ['가격', '구매', '최저가', '비교']):
            seo_features['거래성 의도 반영'] += 1

    seo_optimization_features = f"""
    - 키워드 배치: {seo_features['키워드 전면 배치']}개의 제목에서 핵심 키워드가 제목 초반에 배치되어 검색 엔진에 노출될 확률을 높입니다.
    - 검색 의도 반영: 정보성 키워드 ('방법', '추천' 등)가 많아 사용자의 검색 의도를 명확하게 반영합니다.
    - 제목 길이: {seo_features['적정 길이 유지']}개의 제목이 네이버 SEO에 유리한 15~30자 이내의 적정 길이를 유지하고 있습니다.
    - 구체성: 제목에 숫자, 특정 명사 등이 포함되어 검색 사용자의 질문에 대한 구체적인 답변을 암시합니다.
    """

    # 6. 새로운 블로그 글 제목 10개 (AI 생성)
    combined_analysis_summary = f"""
    이전 블로그 제목들의 분석 결과는 다음과 같습니다:
    - 구조적 특징: 평균 길이 {avg_length:.1f}자, {', '.join(f'{t}형' for t, c in tone_analysis.most_common(1))} 어투가 흔함.
    - 핵심 키워드/표현: '{", ".join(w for w, c in word_counts.most_common(5))}' 등이 자주 사용됨.
    - 패턴: {', '.join(f'{p}형' for p, c in patterns.most_common(1))}이 흔함 (예: 리스트형, 질문형, 정보가이드형).
    - 관심 유도 기법: 숫자 활용, 가치/감성적 표현, 호기심 자극, 타겟 명확화 등이 효과적.
    - SEO 특징: 키워드 전면 배치, 검색 의도 반영 (정보성 위주), 적정 길이 유지가 중요.
    """

    new_titles_prompt = f"""
    위 분석 결과를 참고하여, 기존과 다른 신선한 구조, 패턴, 키워드, SEO 관점을 반영하여 블로그 글 제목 10개를 창의적으로 제안해 주세요.
    제안하는 제목은 기존 제목들의 특징을 활용하되, 더욱 매력적이고 검색 엔진 최적화에 유리하도록 만들어주세요.
    제목의 핵심 키워드는 사용자에게 입력받은 키워드 '{titles[0].split()[0] if titles else "새로운 정보"}'를 자연스럽게 포함하거나, 이와 관련된 확장 키워드를 활용해주세요.
    각 제목은 숫자를 포함하거나, 질문형, 가이드형, 감탄형 등 다양한 패턴을 조합하여 작성해 주세요.
    결과는 번호가 매겨진 리스트 형태로만 제공해주세요.
    """
    
    new_titles_list = []
    if client:
        try:
            new_titles_response = client.chat.completions.create( 
                model=st.session_state.openai_model_name, 
                messages=[
                    {"role": "system", "content": "당신은 SEO 전문가이자 창의적인 카피라이터입니다."},
                    {"role": "user", "content": combined_analysis_summary + new_titles_prompt}
                ],
                temperature=0.7,
                max_tokens=500
            )
            new_titles_text = new_titles_response.choices[0].message.content.strip()
            new_titles_list = [line.strip() for line in new_titles_text.split('\n') if line.strip() and re.match(r'^\d+\.', line)]
        except openai.APIError as e: 
            st.error(f"AI 제목 생성 중 API 오류 발생: {e}")
        except Exception as e:
            st.error(f"AI 제목 생성 중 알 수 없는 오류 발생: {e}")
    else:
        st.warning("OpenAI 클라이언트가 초기화되지 않아 AI 제목 생성을 건너뛸 수 없습니다. API 키를 확인해주세요.")
    
    return {
        "structural_features": structural_features,
        "core_keywords_expressions": core_keywords_expressions,
        "composition_patterns": composition_patterns,
        "attention_techniques": attention_techniques,
        "seo_optimization_features": seo_optimization_features,
        "new_titles": new_titles_list
    }

# --- 4. AI 블로그 글 생성 프롬프트 구조 ---
def generate_seo_optimized_content(keyword: str, analysis_results: dict, target_audience: str = "일반 대중") -> str:
    """
    SEO 최적화된 블로그 포스트를 AI로 생성합니다.
    Args:
        keyword (str): 블로그 포스트의 주요 키워드.
        analysis_results (dict): 제목 분석 결과 (대상 독자 추론 등에 활용).
        target_audience (str): 대상 독자 설명.
    Returns:
        str: 생성된 블로그 포스트 내용.
    """
    # 분석 결과를 바탕으로 대상 독자 추론 (예시)
    if "초보" in keyword or "입문" in keyword:
        target_audience = "관련 분야 초보자 및 입문자"
    elif "전문가" in keyword or "고급" in keyword:
        target_audience = "관련 분야 전문가 및 심화 학습자"
    elif "후기" in keyword or "내돈내산" in keyword:
        target_audience = "제품/서비스 구매를 고려하는 소비자"
    else:
        target_audience = "일반 대중"

    if not client:
        st.warning("OpenAI 클라이언트가 초기화되지 않아 AI 글 생성을 건너뛸 수 없습니다. API 키를 확인해주세요.")
        return "OpenAI 클라이언트가 설정되지 않아 글을 생성할 수 없습니다."

    # 동적으로 로드된 프롬프트 템플릿 사용
    prompt = st.session_state.custom_prompt_template.format(
        keyword=keyword,
        target_audience=target_audience
    )
    
    try:
        response = client.chat.completions.create(
            model=st.session_state.openai_model_name, 
            messages=[
                {"role": "system", "content": "당신은 네이버 블로그 SEO 전문가이자 콘텐츠 마케터입니다. 주어진 키워드와 가이드라인에 따라 독자의 클릭을 유도하고 검색 엔진에 최적화된 고품질 블로그 포스트를 작성합니다."},
                {"role": "user", "content": prompt} 
            ],
            temperature=0.8,
            max_tokens=3000
        )
        return response.choices[0].message.content
    except openai.APIError as e: 
        st.error(f"AI 글 생성 중 API 오류 발생: {e}")
        return "AI 글 생성 중 API 오류가 발생했습니다."
    except Exception as e:
        st.error(f"AI 글 생성 중 알 수 없는 오류 발생: {e}")
        return "AI 글 생성 중 알 수 없는 오류가 발생했습니다."

# --- Streamlit 웹 인터페이스 ---
st.set_page_config(
    page_title="Inbecs: 네이버 블로그 SEO & AI Writer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 헤더 ---
st.container()
col1, col2 = st.columns([1, 6])
with col1:
    # 로고 이미지 플레이스홀더 (여기에 실제 로고 이미지 경로를 넣을 수 있습니다)
    # st.image("path/to/your/logo.png", width=50) 
    st.markdown(" ") # 공간 확보용
with col2:
    st.markdown("<h1 style='text-align: left; color: #1DB954;'>Inbecs</h1>", unsafe_allow_html=True)
st.markdown("---")

# --- 사이드바 메뉴 ---
st.sidebar.header("메뉴")
page_selection = st.sidebar.radio("원하는 기능을 선택하세요:", ["블로그 글 생성", "설정 및 지침 수정"])
st.sidebar.markdown("---")
st.sidebar.info("이 도구는 네이버 블로그 검색 API와 OpenAI GPT-4o를 활용하여 블로그 글 제목을 분석하고 SEO 최적화된 블로그 콘텐츠를 생성합니다.")


# --- 메인 콘텐츠 영역 ---
if page_selection == "블로그 글 생성":
    st.title("🚀 네이버 블로그 SEO & AI 블로그 글 생성기")
    
    with st.sidebar:
        st.header("블로그 글 생성 설정")
        keyword_input = st.text_input("검색할 키워드를 입력하세요:", "강남 맛집", key="main_keyword_input")
        display_count = st.slider("네이버 블로그 검색 결과 개수:", min_value=1, max_value=100, value=30, step=1, key="main_display_count")
        
        # "검색 및 분석 시작" 버튼을 누르면 초기화 및 분석 시작
        if st.button("🔍 검색 및 분석 시작", key="main_search_button"):
            st.session_state.run_analysis = True
            st.session_state.keyword = keyword_input
            st.session_state.display_count = display_count
            st.session_state.selected_blog_title = None
            st.session_state.generated_content = None # 이전 생성된 글 초기화
            st.session_state.trigger_generation_flag = False # 글 생성 트리거 초기화
            st.session_state.title_analysis_results = None # 분석 결과 초기화
            st.session_state.generated_status = {} # 각 제목별 생성 여부 초기화
            st.rerun() 

    # 세션 상태 초기화 및 관리
    if 'run_analysis' not in st.session_state: st.session_state.run_analysis = False
    if 'selected_blog_title' not in st.session_state: st.session_state.selected_blog_title = None
    if 'generated_content' not in st.session_state: st.session_state.generated_content = None
    if 'trigger_generation_flag' not in st.session_state: st.session_state.trigger_generation_flag = False # 새로운 글 생성 트리거
    if 'display_count' not in st.session_state: st.session_state.display_count = 30
    if 'title_analysis_results' not in st.session_state: st.session_state.title_analysis_results = None
    if 'generated_status' not in st.session_state: st.session_state.generated_status = {}

    # 수동 제목 입력 섹션
    st.markdown("---")
    st.subheader("📝 수동 제목으로 블로그 글 생성")
    manual_title_input = st.text_input("직접 블로그 제목을 입력하세요:", "", key="manual_title_input")
    if st.button("수동 제목으로 글 생성", key="manual_generate_button"):
        if manual_title_input:
            st.session_state.selected_blog_title = manual_title_input
            st.session_state.trigger_generation_flag = True # 글 생성 트리거 설정
            st.session_state.generated_content = None # 이전 글 내용 초기화
            st.session_state.run_analysis = False # 분석 결과 표시 섹션 비활성화
            st.rerun()
        else:
            st.warning("수동으로 생성할 제목을 입력해주세요.")

    st.markdown("---")


    if st.session_state.run_analysis:
        st.markdown(f"## '{st.session_state.keyword}' 키워드 분석 결과")
        
        # 1. 네이버 블로그 검색 결과 표시
        st.subheader(f"📊 네이버 블로그 검색 결과 (상위 {st.session_state.display_count}개)")
        with st.spinner("네이버 블로그 검색 중..."):
            naver_posts = search_naver_blogs(st.session_state.keyword, st.session_state.display_count)
        
        if naver_posts:
            st.write(f"총 {len(naver_posts)}개 포스트를 찾았습니다.")
            titles_for_analysis = [post["title"] for post in naver_posts]

            for i, post in enumerate(naver_posts):
                # URL을 제목에 링크로 걸어 표시하고 URL 텍스트는 제거
                st.markdown(f"**{i+1}. [{post['title']}]({post['link']})**") 
                st.write(f"요약: {post['description'][:100]}...")
            
            st.markdown("---")
            
            # 2. SEO 최적화 제목 분석 (백그라운드에서 실행, 화면에는 표시 안 함)
            if st.session_state.title_analysis_results is None: # 이미 분석 결과가 없으면 새로 분석
                with st.spinner("AI가 제목 특징을 분석 중... (이 결과는 백그라운드에서 사용됩니다.)"):
                    st.session_state.title_analysis_results = analyze_blog_titles(titles_for_analysis)
            
            st.markdown("---")
            
            st.subheader("✨ 새로운 블로그 글 제목 10개 제안 (마음에 드는 제목을 클릭하세요!)")
            if st.session_state.title_analysis_results and st.session_state.title_analysis_results["new_titles"]:
                col_idx, col_title_button, col_checkbox = st.columns([0.5, 4, 1])
                with col_idx: st.markdown("**#**")
                with col_title_button: st.markdown("**제안 제목**")
                with col_checkbox: st.markdown("**생성 여부**")
                st.markdown("---")

                for i, title_with_num in enumerate(st.session_state.title_analysis_results["new_titles"]):
                    # 제목에서 번호 제거 (예: "1. 멋진 블로그 제목" -> "멋진 블로그 제목")
                    clean_title = re.sub(r'^\d+\.\s*', '', title_with_num).strip()

                    col_idx, col_title_button, col_checkbox = st.columns([0.5, 4, 1])
                    with col_idx:
                        st.write(f"{i+1}.")
                    with col_title_button:
                        if st.button(clean_title, key=f"title_btn_{i}"):
                            st.session_state.selected_blog_title = clean_title # 번호 제거된 제목 저장
                            st.session_state.trigger_generation_flag = True # 글 생성 트리거 설정
                            st.session_state.generated_content = None # 이전 글 내용 초기화
                            # st.session_state.run_analysis는 True로 유지하여 제목 목록이 계속 보이게 함
                            st.rerun() 
                    with col_checkbox:
                        is_generated = st.session_state.generated_status.get(clean_title, False) # 번호 제거된 제목으로 상태 확인
                        st.checkbox("생성 완료", value=is_generated, disabled=True, key=f"checkbox_{i}")
            else:
                st.warning("새로운 제목을 생성하는 데 실패했거나 OpenAI API 키가 올바르지 않습니다.")
        else:
            st.warning("네이버 블로그 검색 결과가 없거나 오류가 발생했습니다. 키워드를 변경하여 다시 시도해 주세요.")
    
    # 블로그 글 생성 로직 (trigger_generation_flag가 True일 때만 실행)
    if st.session_state.trigger_generation_flag and st.session_state.selected_blog_title:
        st.markdown("---")
        st.subheader(f"✍️ AI 기반 SEO 최적화 블로그 글 생성: '{st.session_state.selected_blog_title}'")
        
        with st.spinner(f"'{st.session_state.selected_blog_title}' 제목으로 블로그 글을 작성 중... 잠시 기다려 주세요."):
            # AI에게는 번호 없는 제목을 전달
            generated_content = generate_seo_optimized_content(
                st.session_state.selected_blog_title, 
                st.session_state.title_analysis_results # analysis_results는 여기서도 활용 가능
            )
            st.session_state.generated_content = generated_content # 생성된 글을 세션 상태에 저장
        
        # 글 생성이 완료되면 트리거 플래그 바로 해제 (자동 재실행 방지)
        st.session_state.trigger_generation_flag = False 

    # 생성된 글 표시 및 저장/삭제 버튼 (generated_content가 있을 때만 표시)
    if st.session_state.generated_content:
        st.markdown("---")
        st.subheader("📰 생성된 블로그 글")
        st.markdown(st.session_state.generated_content)

        # 해당 제목에 대해 글이 생성되었음을 상태에 기록
        if st.session_state.selected_blog_title: # 선택된 제목이 있을 때만 기록
            st.session_state.generated_status[st.session_state.selected_blog_title] = True

        st.markdown("---")
        st.subheader("📁 생성된 블로그 글 저장 및 삭제")
        
        # 파일명 제안 (한글, 특수문자 고려)
        default_filename = re.sub(r'[^가-힣a-zA-Z0-9ㄱ-ㅎㅏ-ㅣ]', '_', st.session_state.selected_blog_title if st.session_state.selected_blog_title else "블로그_글")
        default_filename = re.sub(r'_+', '_', default_filename).strip('_') + ".md"
        
        if not default_filename.strip(".md"):
            default_filename = "블로그_글.md"

        download_filename = st.text_input("저장할 파일 이름을 입력하세요 (확장자 포함):", default_filename, key="download_filename_input")
        
        col_save, col_delete = st.columns(2)
        with col_save:
            st.download_button(
                label=f"'{download_filename}' 파일로 저장",
                data=st.session_state.generated_content.encode('utf-8'),
                file_name=download_filename,
                mime="text/markdown",
                key="download_button"
            )
        with col_delete:
            if st.button("생성된 글 삭제", key="delete_generated_content_button"):
                st.session_state.generated_content = None # 글 내용 삭제
                st.session_state.selected_blog_title = None # 선택된 제목 초기화
                # st.session_state.generated_status[clean_title] = False # 필요시 체크박스 해제 (하지만 보통 생성된건 유지)
                st.success("생성된 블로그 글이 화면에서 삭제되었습니다.")
                st.rerun() # 화면 업데이트
        
        st.info("다운로드 버튼을 클릭하면 브라우저에서 파일을 저장할 수 있습니다. 특정 폴더를 직접 지정하는 기능은 웹 앱의 보안 제약 상 제공되지 않습니다.")

    # Reset selected_blog_title after potential generation to prevent persistent selection causing issues
    # This reset happens after the generation/display block, ensuring it's processed first.
    # It might be cleared by the delete button or new search already.
    # if st.session_state.selected_blog_title and not st.session_state.trigger_generation_flag:
    #    st.session_state.selected_blog_title = None


elif page_selection == "설정 및 지침 수정":
    st.title("⚙️ 설정 및 지침 수정")
    st.markdown("---")

    st.subheader("AI 블로그 글 생성 지침 (프롬프트 템플릿)")
    st.info("AI가 블로그 글을 생성할 때 사용되는 기본 가이드라인입니다. 필요에 따라 수정하여 AI의 응답 스타일이나 포함될 내용을 조절할 수 있습니다. `{keyword}`와 `{target_audience}`는 자동으로 채워지는 변수입니다.")
    
    edited_prompt = st.text_area(
        "프롬프트 템플릿 수정:",
        st.session_state.custom_prompt_template,
        height=500,
        key="prompt_editor"
    )

    col_save_prompt, col_reset_prompt = st.columns(2)
    with col_save_prompt:
        if st.button("지침 저장", key="save_prompt_button"):
            st.session_state.custom_prompt_template = edited_prompt
            app_config["prompt_template"] = edited_prompt
            save_config(app_config)
            st.success("새로운 지침이 저장되었습니다!")
            st.rerun()
    with col_reset_prompt:
        if st.button("기본 지침으로 복원", key="reset_prompt_button"):
            st.session_state.custom_prompt_template = DEFAULT_PROMPT_TEMPLATE
            app_config["prompt_template"] = DEFAULT_PROMPT_TEMPLATE
            save_config(app_config)
            st.warning("지침이 기본값으로 복원되었습니다!")
            st.rerun()
    
    st.markdown("---")
    st.subheader("AI 모델 설정")
    st.info("OpenAI API를 사용할 모델을 선택/입력합니다. 성능과 비용에 영향을 미칩니다. (예: `gpt-4o`, `gpt-3.5-turbo`)")

    model_options = ["gpt-4o", "gpt-3.5-turbo", "text-davinci-003", "davinci-002"] # 예시 모델 목록
    
    selected_model_name = st.selectbox(
        "OpenAI 모델 선택:",
        options=model_options,
        index=model_options.index(st.session_state.openai_model_name) if st.session_state.openai_model_name in model_options else 0,
        key="openai_model_selector"
    )
    
    # 사용자가 직접 입력할 수 있는 필드 추가 (선택지 외의 모델을 입력하고 싶을 경우)
    custom_model_name = st.text_input(
        "또는 직접 모델 이름 입력 (선택지 외 모델):", 
        st.session_state.openai_model_name if st.session_state.openai_model_name not in model_options else "",
        key="custom_openai_model_input"
    )

    if custom_model_name:
        st.session_state.openai_model_name = custom_model_name
    else:
        st.session_state.openai_model_name = selected_model_name

    if st.button("모델 설정 저장", key="save_model_button"):
        app_config["openai_model_name"] = st.session_state.openai_model_name
        save_config(app_config)
        st.success(f"AI 모델이 '{st.session_state.openai_model_name}'으로 설정되었습니다!")
        st.rerun()

    # "다른 AI API 연동 안내" 섹션은 요청에 따라 제거했습니다.
    # st.markdown("---")
    # st.subheader("다른 AI API 연동 안내")
    # st.warning("""
    # 현재 이 애플리케이션은 OpenAI API에만 연동되어 있습니다. 
    # Google Gemini, Anthropic Claude와 같은 다른 AI 서비스의 API를 연동하려면 각 서비스의 SDK를 설치하고, 
    # 해당 API의 호출 방식에 맞춰 코드 구조를 크게 변경해야 합니다. 
    # 이는 단순히 API 키를 변경하는 것을 넘어선 복잡한 개발 작업이므로, 현재 버전에서는 지원되지 않습니다.
    # """)

# --- 푸터 ---
st.markdown("---")
st.markdown(f"<p style='text-align: center; color: gray;'>Copyright © {time.strftime('%Y')} Inbecs. All rights reserved.</p>", unsafe_allow_html=True)