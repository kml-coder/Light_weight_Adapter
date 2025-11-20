import requests
import time
import html
from bs4 import BeautifulSoup

def clean_html(html_text):
    """HTML에서 텍스트만 추출"""
    soup = BeautifulSoup(html_text, "html.parser")
    return soup.get_text(" ", strip=True)



def extract_text_from_paragraph(paragraph):
    if isinstance(paragraph, list):
        return "".join(seg.get("text", "") for seg in paragraph).strip()

    if isinstance(paragraph, str):
        real_html = html.unescape(paragraph)
        soup = BeautifulSoup(real_html, "html.parser")
        return soup.get_text(" ", strip=True)

    return ""


def get_naver_examples(query, max_pages=10, save_file="examples_raw.txt", save_clean="examples_clean.txt"):
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/123.0.0.0 Safari/537.36",
        "Referer": "https://ko.dict.naver.com/"
    }

    raw_results = []
    clean_results = []

    for page in range(1, max_pages + 1):
        url = f"https://ko.dict.naver.com/api3/koko/search?query={query}&range=example&page={page}"
        print(f"\n=== 📄 요청: page {page} ===")
        print(url)

        # 1. HTTP 요청
        try:
            res = requests.get(url, headers=headers, timeout=5)
        except Exception as e:
            print(f"❌ 요청 실패: {e}")
            continue

        if res.status_code != 200:
            print(f"❌ 상태 코드 오류: {res.status_code}")
            continue

        # 2. JSON 파싱
        try:
            data = res.json()
        except:
            print("❌ JSON 파싱 실패 (HTML일 가능성)")
            print(res.text[:300])
            continue

        # 3. 예문 데이터 추출
        try:
            items = data["searchResultMap"]["searchResultListMap"]["EXAMPLE"]["items"]
        except:
            print("⚠️ 데이터 없음 → 마지막 페이지일 가능성")
            break

        # 4. 예문 저장
        for item in items:
            html = item.get("paragraph", "")
            text = extract_text_from_paragraph(html)
            raw_results.append(text)
            clean_results.append(clean_html(text))

        print(f"✔ 페이지 {page} 수집: {len(items)}개")

        time.sleep(0.3)

    # ---------------------------
    # 파일로 저장
    # ---------------------------
    with open(save_file, "w", encoding="utf-8") as f:
        for line in raw_results:
            f.write(line + "\n")

    with open(save_clean, "w", encoding="utf-8") as f:
        for line in clean_results:
            f.write(line + "\n")

    print(f"\n💾 저장 완료:")
    print(f"- Raw HTML 저장: {save_file}")
    print(f"- Clean Text 저장: {save_clean}")
    print(f"총 {len(clean_results)}개 예문")


if __name__ == "__main__":
    get_naver_examples("체면", max_pages=12)
