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

file = open("system_prompt.txt", "r")

CHATBOT_ROLE_INSTRUCTION = file.read()
file.close()

if DEBUG_MODE:
    print(CHATBOT_ROLE_INSTRUCTION)

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
    try:
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
    except Exception:
        return None

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

        messages = build_chat_messages(user_input, history, current_summary, load_enriched_watch_history())

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
