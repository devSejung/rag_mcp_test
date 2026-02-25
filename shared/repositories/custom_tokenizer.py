import re
from typing import List

# 불필요한 마크다운 메타데이터성 키워드 배제
STOPWORDS = {'summary', 'description', 'comment', 'comments', 'assignee', 'reporter', 'status', 'resolution', 'label', 'labels'}

def tokenize(text: str) -> List[str]:
    """
    SOC Platform 특화 커스텀 토크나이저
    1. 특수문자 보존 및 후행 구두점 제거
    2. 문맥 단절을 인지하는 영문/숫자 N-gram (Bi-gram, Tri-gram) 생성
    3. 한글 토큰 분리 및 독립성 유지
    """
    if not text:
        return []

    # 1. 문맥 단절 구간(Boundary)을 기준으로 텍스트를 먼저 분리 (구(Phrase) 단위)
    # 쉼표(,), 세미콜론(;), 줄바꿈(\n), 그리고 뒤에 공백이 따르는 콜론(: )이나 마침표(. )를 단절선으로 본다.
    # 괄호 등도 단절의 기준으로 추가할 수 있으나, 일단 명시된 대로 적용.
    phrases = re.split(r'[,;\n]+|:\s+|\.\s+|\(|\)', text)
    
    final_tokens = []
    
    # 영문/숫자/특수문자 조합 또는 한글 덩어리를 찾는 정규식
    # 한글: [가-힣]+
    # 영/숫자/특수기호: [a-zA-Z0-9_\-\.\:\/\\\+\#]+ (단어를 구성하는 모든 허용 기호)
    pattern = re.compile(r'[가-힣]+|[a-zA-Z0-9_\-\.\:\/\\\+\#]+')

    for phrase in phrases:
        if not phrase.strip():
            continue
            
        raw_tokens = pattern.findall(phrase)
        phrase_eng_tokens = []
        
        for rt in raw_tokens:
            is_korean = bool(re.match(r'^[가-힣]+$', rt))
            
            if is_korean:
                # 한글은 후처리 없이 독립 토큰으로 추가 (순수 특수기호만 있는 한글 덩어리는 없으므로 안전)
                final_tokens.append(rt)
                # 한글이 등장하면 연속된 영문 N-gram의 흐름도 끊어줌 (노이즈 방지)
                if phrase_eng_tokens:
                    final_tokens.extend(_generate_ngrams(phrase_eng_tokens))
                    phrase_eng_tokens = []
            else:
                # 규칙 1: 영문/숫자 토큰 후행 구두점 제거 (:, ., ,)
                clean_t = rt.rstrip(':.,')
                
                if clean_t:
                    clean_t = clean_t.lower()
                    
                    # 1. 불용어(Stopwords) 체크
                    if clean_t in STOPWORDS:
                        continue
                        
                    # 2. 순수 특수기호 찌꺼기 체크 (알파벳이나 숫자가 전혀 포함되어 있지 않는 문자는 제거, 예: ###, ***, ---)
                    if not re.search(r'[a-z0-9가-힣]', clean_t):
                        continue
                        
                    # 최종 유효 토큰 등록
                    final_tokens.append(clean_t) # 기본 토큰 (Unigram)
                    phrase_eng_tokens.append(clean_t)
                    
        # 구(Phrase)가 끝났을 때 쌓여있는 영문 토큰들로 N-gram 생성
        if phrase_eng_tokens:
            final_tokens.extend(_generate_ngrams(phrase_eng_tokens))
            
    # 중복 제거 및 리스트 반환 (Set은 순서를 보장하지 않으므로, 순서 유지를 원하면 dict.fromkeys 사용)
    return list(dict.fromkeys(final_tokens))

def _generate_ngrams(tokens: List[str]) -> List[str]:
    """연속된 영문/숫자 토큰 리스트에서 Bi-gram, Tri-gram 생성"""
    ngrams = []
    n = len(tokens)
    
    # Bi-gram (2개 묶음)
    for i in range(n - 1):
        ngrams.append(f"{tokens[i]} {tokens[i+1]}")
        
    # Tri-gram (3개 묶음)
    for i in range(n - 2):
        ngrams.append(f"{tokens[i]} {tokens[i+1]} {tokens[i+2]}")
        
    return ngrams

if __name__ == "__main__":
    # Test Cases
    cases = [
        "vwm: valid window margin",
        "레지스터 0x4000_8000 접근 시 Timeout 발생 (SOC-1234)",
        "main.c:125 에서 init_dram_ctrl() 호출 실패, error, 확인 요망",
        "drivers/soc/ 경로의 SOC_Platform\\ 빌드 C++ C# 테스트",
        "DDR 초기화 불량 확인 (ERR_05)",
        "### Description\n이슈 정리. \n**summary**: test 수행 중 --- 발생",
        "comment: code refactoring."
    ]
    
    for c in cases:
        print(f"\\nInput: {c}")
        print(f"Tokens: {tokenize(c)}")
