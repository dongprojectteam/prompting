import os
from dotenv import load_dotenv
import google.generativeai as genai
import datetime
import json
from collections import Counter

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    print("GOOGLE_API_KEY 환경 변수를 찾을 수 없습니다.")
    exit()
genai.configure(api_key=api_key)

MODEL_NAME = "gemini-2.0-flash"  # 시뮬레이션을 위한 모델 - VAC와 동일하게 사용용
DEBUG_MODE = True
CONTEXT_FILE = "conversation_context.json"
WATCH_HISTORY_FILE = "watch_history.json"
CONTENTS_FILE = "contents.json"

CHATBOT_ROLE_INSTRUCTION = """

너는 사용자의 콘텐츠 시청 습관, 선호도, 현재 상황 및 감정 상태를 파악하여 맞춤형 콘텐츠를 추천하는 AI 비서다.
너는 대화의 전제, 규칙, 세계관을 정의하는 system 역할을 수행하며, Gemini Flash 모델의 특성을 고려하여 설계되었다.
사용자의 질문에 대해 이전 대화 내용(user-model 쌍으로 구성), 요약본, 그리고 사용자의 현재 상황(시간, 요일, 질문에 나타난 감정 등)을 종합적으로 참고하여 답변한다.
답변은 항상 명확하고 핵심적인 정보를 전달해야 하며 (위 30글자 규칙 엄수), 사용자의 지시를 충실히 따른다.
컨텐츠를 추천할 땐 항상 10개의 컨텐츠를 추천한다.

**[프로그램 목적 및 핵심 원칙 (모든 답변은 30글자 이내)]**

1.  **문맥(Context) 유지:** 이전 대화 및 요약을 반영, 일관성을 유지한다.
2.  **정확하고 적절한 응답 생성:** 사용자 의도 파악, 관련성 높은 정보로 추천한다.
3.  **중복 표현 방지:** 이전 추천/표현의 불필요한 반복을 피한다.
4.  **항상 컨텐츠 추천 및 관련도 우선 제공:** 추천 기회를 탐색, 관련도 높은 콘텐츠를 우선 제시한다.
5.  **추천 이유 명시 (사용자 신뢰도 향상):** 추천 이유(장르 선호, 유사 작품 등)를 핵심만 간결히 제공한다.
6.  **감정 상태 및 현재 상황 반영 맞춤형 추천:** 사용자 감정/상황 고려, 분위기에 맞는 콘텐츠 추천한다.

**[콘텐츠 추천 시 세부 지침 (모든 답변은 30글자 이내)]**

사용자가 콘텐츠 추천을 원하거나 관련된 대화를 할 때, 위의 핵심 원칙을 바탕으로 다음 지침을 따른다:

1.  **추천 다양성, 신선도 및 개수:**
    * 다양한 장르, 테마, 분위기 조합 및 신/구작 균형을 고려한다.
    * 항상 10개 새 콘텐츠 추천 목표. (각 추천 내용은 매우 간결해야 함)

2.  **입력 유형별 추천 전략:**
    * **취향 기반 추천 (일반적/특정 콘텐츠 질문):**
        * 사용자 선호 프로그램(언급, 이력 등) 기반, 장르/분위기/출연진 등 분석 후 새 콘텐츠 추천.
        * 특정 콘텐츠 질문 시, 유사 요소(분위기, 장르 등) 중심으로 추천. 맥락상 관련 높으면 재언급 가능(남용 금지).
    * **상황 기반 추천 (시간 관련 질문, 감정 표현 등):**
        * 명확한 선호 정보 부족 시, '현재 시각', '요일' 등 활용. (예: 저녁엔 드라마, 주말엔 예능 핵심 추천)
        * 감지된 감정(예: 지루함, 신남)에 맞춰 분위기 맞는 콘텐츠 추천.

3.  **선제적 추천 및 정보 활용:**
    * 추가 질문 반복보다, 주어진 정보(대화 이력, 요약, 현재 상황 등)로 먼저 추천.

4.  **답변의 상세 수준 (30글자 제한 엄수):**
    * 모든 답변은 핵심 정보만 간결하게 전달한다. 콘텐츠 추천 시에도 추천 이유를 포함한 핵심 내용을 30글자 이내로 요약하여 제공한다.
"""

try:
    chat_model = genai.GenerativeModel(MODEL_NAME, system_instruction=CHATBOT_ROLE_INSTRUCTION)
    summarizer_model = genai.GenerativeModel(MODEL_NAME)
except Exception as e:
    print(f"모델 ({MODEL_NAME}) 초기화 중 오류 발생: {e}")
    exit()

def load_json_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_enriched_watch_history():
    raw_watch_data = load_json_file(WATCH_HISTORY_FILE)
    contents = load_json_file(CONTENTS_FILE)
    content_map = {content["content_id"]: content for content in contents}

    enriched_history = []
    for user, history_entries in raw_watch_data.items():
        for entry in history_entries:
            content_id = entry.get("content_id")
            content_info = content_map.get(content_id, {})
            merged_entry = {
                "user": user,
                **entry,
                **content_info  # 콘텐츠 정보 포함
            }
            enriched_history.append(merged_entry)

    if DEBUG_MODE:
        print("[과거 시청 이력]")
        for record in enriched_history:
            print(record)

    return enriched_history



def save_context_to_file(filepath, history_data, summary_data, tc_data):
    context_data = {
        "history": history_data,
        "current_summary": summary_data,
        "turn_count": tc_data,
    }
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(context_data, f, ensure_ascii=False, indent=4)

def load_context_from_file(filepath):
    try:
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                context_data = json.load(f)
                return (
                    context_data.get("history", []),
                    context_data.get("current_summary", ""),
                    context_data.get("turn_count", 0),
                )
    except Exception as e:
        print(f"[시스템] 대화 내용 불러오기 중 오류 발생: {e}")
    return [], "", 0

def summarize_conversation_history(full_conv_history):
    if not full_conv_history or len(full_conv_history) < 5:
        return ""
    summary_instruction = "다음 대화 내용을 요약해줘."
    formatted = "".join([f"사용자: {e['user']}\nGemini: {e['model']}\n" for e in full_conv_history])
    prompt = f"{summary_instruction}\n---\n{formatted}\n---"
    try:
        response = summarizer_model.generate_content([{"role": "user", "parts": [prompt]}])
        return response.text.strip()
    except Exception as e:
        print(f"요약 오류: {e}")
        return ""

def build_chat_messages(user_input_text, conv_history, summary_text, enriched_watch_history):
    messages_for_api = []

    # 최근 대화 5턴 포함
    for entry in conv_history[-5:]:
        messages_for_api.append({"role": "user", "parts": [entry["user"]]})
        messages_for_api.append({"role": "model", "parts": [entry["model"]]})

    # 현재 시각 포함
    current_turn_user_prompt_parts = []
    now = datetime.datetime.now()
    current_time_str = now.strftime("%Y년 %m월 %d일 %A %p %I:%M")
    current_turn_user_prompt_parts.append(f"참고: 현재 시각은 {current_time_str} 입니다.")

    if enriched_watch_history:
        try:
            watched_titles = [entry["title"] for entry in enriched_watch_history if "title" in entry][:20]
            watched_text = ", ".join(watched_titles)
            current_turn_user_prompt_parts.append(f"과거 시청 이력: {watched_text}")

            genre_list = []
            for entry in enriched_watch_history:
                genre = entry.get("genre")
                if isinstance(genre, list):
                    genre_list.extend(genre)
                elif isinstance(genre, str):
                    genre_list.append(genre)

            genre_counter = Counter(genre_list)
            top_genres = [f"{genre}({count})" for genre, count in genre_counter.most_common(3)]
            genre_text = ", ".join(top_genres)
            current_turn_user_prompt_parts.append(f"선호 장르: {genre_text}")

        except Exception as e:
            print(f"[경고] 시청 이력 처리 오류: {e}")

    if summary_text:
        current_turn_user_prompt_parts.append(f"이전 대화 요약: {summary_text}")

    current_turn_user_prompt_parts.append(f"사용자 질문:\n{user_input_text}")

    final_user_content = "\n\n".join(current_turn_user_prompt_parts).strip()
    
    if DEBUG_MODE:
        print(f"[과거를 바탕으로한 기록] : {final_user_content}")
    messages_for_api.append({"role": "user", "parts": [final_user_content]})
    return messages_for_api



history, current_summary, turn_count = load_context_from_file(CONTEXT_FILE)
enriched_watch_history = load_enriched_watch_history()

print(f"{MODEL_NAME} 초기화 완료. 이전 대화 수: {len(history)}")

try:
    while True:
        user_input = input("사용자: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            save_context_to_file(CONTEXT_FILE, history, current_summary, turn_count)
            break

        turn_count += 1
        needs_summary = (
            turn_count == 10 or (turn_count > 10 and (turn_count - 10) % 15 == 0)
        )

        if needs_summary:
            print("[시스템] 대화 요약 중...")
            new_summary = summarize_conversation_history(history)
            if new_summary:
                current_summary = new_summary

        messages = build_chat_messages(user_input, history, current_summary, enriched_watch_history)

        if DEBUG_MODE:
            for i, msg in enumerate(messages):
                print(f"[{i+1}] {msg['role']}: {msg['parts'][0][:100]}...")

        try:
            response = chat_model.generate_content(messages)
            reply = response.text.strip()
            print("Gemini:", reply)
            history.append({"user": user_input, "model": reply})
        except Exception as e:
            print(f"오류 발생: {e}")
except KeyboardInterrupt:
    print("\n[시스템] 종료 신호 감지됨. 저장 후 종료합니다.")
    save_context_to_file(CONTEXT_FILE, history, current_summary, turn_count)
