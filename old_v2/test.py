import requests
from urllib.parse import quote

def get_naver_examples(word):
    session = requests.Session()

    # 1) 쿠키 확보 (NNB 생성)
    session.get("https://ko.dict.naver.com")

    # 2) 실제 Chrome AJAX 요청 헤더 그대로
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
        ),
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
        "Referer": "https://ko.dict.naver.com/",
        "X-Requested-With": "XMLHttpRequest",

        # 핵심: 안 넣으면 차단
        "Sec-Fetch-Site": "same-origin",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Dest": "empty",
    }

    encoded = quote(word)
    url = f"https://ko.dict.naver.com/api3/search?query={encoded}&range=example"

    r = session.get(url, headers=headers)

    # 차단 페이지인지 검사
    if r.text.strip().startswith("<!DOCTYPE html>"):
        print("❌ 네이버 차단됨 → 헤더 또는 쿠키 부족")
        print(r.text[:500])
        return []

    # JSON 파싱
    data = r.json()

    examples = []
    for item in data.get("searchResultMap", {}).get("searchResultList", []):
        if item.get("type") == "EXAMPLE":
            for ex in item.get("searchResultList", []):
                examples.append({
                    "kor": ex.get("exampleHtml", "").replace("<b>", "").replace("</b>", ""),
                    "eng": ex.get("translation", "")
                })

    return examples


# 테스트
if __name__ == "__main__":
    result = get_naver_examples("체면")
    print(result)
