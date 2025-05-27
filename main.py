import os
from dotenv import load_dotenv
import google.generativeai as genai
import datetime
import json
from collections import Counter
import re # 정규표현식 모듈 추가

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    print("GOOGLE_API_KEY 환경 변수를 찾을 수 없습니다.")
    exit()
genai.configure(api_key=api_key)

MODEL_NAME = "gemini-2.0-flash"
DEBUG_MODE = True
CONTEXT_FILE = "conversation_context.json"
WATCH_HISTORY_FILE = "watch_history.json"
CONTENTS_FILE = "contents.json"

file = open("system_prompt.txt", "r", encoding="utf-8") # 인코딩 추가
CHATBOT_ROLE_INSTRUCTION = file.read()
file.close()

if DEBUG_MODE:
    print("--- [시스템 프롬프트] ---")
    print(CHATBOT_ROLE_INSTRUCTION)
    print("------------------------\n")

try:
    # JSON Schema 정의 (프롬프트와 일치하도록 구성)
    # google-generativeai 라이브러리의 types.Schema를 사용하여 명확하게 정의하는 것이 더 좋습니다.
    # 여기서는 dict 형태로 간략화하여 표현했습니다.
    json_schema = {
        "type": "object",
        "properties": {
            "recommendation_type": {"type": "string", "enum": ["media_content", "general_text"]},
            "recommendation_reason_summary": {"type": "string"},
            "recommended_contents": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "reason": {"type": "string"}
                    },
                    "required": ["title", "reason"]
                },
            },
            "response_text": {"type": "string"}
        },
        "required": ["recommendation_type"]
    }

    chat_model = genai.GenerativeModel(
        MODEL_NAME,
        system_instruction=CHATBOT_ROLE_INSTRUCTION,
        generation_config={
            "response_mime_type": "application/json",
            "response_schema": json_schema # JSON 스키마 적용
        }
    )
    summarizer_model = genai.GenerativeModel(MODEL_NAME)
except Exception as e:
    print(f"모델 ({MODEL_NAME}) 초기화 중 오류 발생: {e}")
    exit()

def load_json_file(file_path):
    """JSON 파일을 로드합니다."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"[경고] 파일이 없습니다: {file_path}")
        return {}
    except json.JSONDecodeError:
        print(f"[오류] JSON 디코딩 오류: {file_path}")
        return {}

def load_enriched_watch_history():
    """과거 시청 이력을 콘텐츠 정보와 함께 로드하여 풍부하게 만듭니다."""
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
            print("[과거 시청 이력 불러오기 완료]")
            #for record in enriched_history: # 너무 길면 주석 처리
            #    print(record)

        return enriched_history
    except Exception as e:
        print(f"[오류] 과거 시청 이력 로드 중 오류 발생: {e}")
        return None

def save_context_to_file(filepath, history_data, summary_data, tc_data):
    """대화 컨텍스트를 파일에 저장합니다."""
    context_data = {
        "history": history_data,
        "current_summary": summary_data,
        "turn_count": tc_data,
    }
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(context_data, f, ensure_ascii=False, indent=4)
    if DEBUG_MODE:
        print(f"[시스템] 대화 컨텍스트가 '{filepath}'에 저장되었습니다.")

def load_context_from_file(filepath):
    """파일에서 대화 컨텍스트를 로드합니다."""
    try:
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                context_data = json.load(f)
                if DEBUG_MODE:
                    print(f"[시스템] 대화 컨텍스트를 '{filepath}'에서 불러왔습니다.")
                return (
                    context_data.get("history", []),
                    context_data.get("current_summary", ""),
                    context_data.get("turn_count", 0),
                )
    except Exception as e:
        print(f"[시스템] 대화 내용 불러오기 중 오류 발생: {e}")
    if DEBUG_MODE:
        print("[시스템] 새로운 대화를 시작합니다. (이전 컨텍스트 없음)")
    return [], "", 0

def summarize_conversation_history(full_conv_history):
    """대화 이력을 요약합니다."""
    # JSON 응답도 텍스트로 변환하여 요약 모델에 전달
    formatted_history_for_summary = []
    for entry in full_conv_history:
        user_text = entry["user"]
        model_text = entry["model"]
        try:
            parsed_model_response = json.loads(model_text)
            if parsed_model_response.get("recommendation_type") == "media_content":
                summary_text = parsed_model_response.get("recommendation_reason_summary", "")
                titles = [item.get("title", "") for item in parsed_model_response.get("recommended_contents", [])]
                model_text = f"영상 추천: {summary_text}. 컨텐츠: {', '.join(titles[:3])}..." # 요약 모델에 너무 긴 내용 전달 방지
            elif parsed_model_response.get("recommendation_type") == "general_text":
                model_text = parsed_model_response.get("response_text", "")
        except json.JSONDecodeError:
            pass # JSON이 아니면 원본 텍스트 사용

        formatted_history_for_summary.append(f"사용자: {user_text}\nGemini: {model_text}")

    if not formatted_history_for_summary or len(formatted_history_for_summary) < 5:
        return ""
    
    summary_instruction = "다음 대화 내용을 요약해줘. 중요한 대화 주제와 사용자의 요청 사항을 중심으로 요약해."
    formatted = "\n".join(formatted_history_for_summary)
    prompt = f"{summary_instruction}\n---\n{formatted}\n---"
    try:
        response = summarizer_model.generate_content([{"role": "user", "parts": [prompt]}])
        return response.text.strip()
    except Exception as e:
        print(f"[오류] 요약 중 오류 발생: {e}")
        return ""

def format_model_response_for_history(model_response_str):
    """history에 저장된 모델 응답 (JSON)을 모델이 이해할 수 있는 텍스트로 변환합니다."""
    try:
        parsed_model_response = json.loads(model_response_str)
        if parsed_model_response.get("recommendation_type") == "media_content":
            summary = parsed_model_response.get("recommendation_reason_summary", "추천 이유 요약 없음")
            contents = parsed_model_response.get("recommended_contents", [])
            titles = [item.get("title", "제목 없음") for item in contents]
            
            # 모델에게 전달할 때 너무 길지 않게 간결하게 요약
            formatted_contents = ", ".join(titles[:5]) # 최대 5개 타이틀만 전달
            if len(titles) > 5:
                formatted_contents += " 등"
            
            return f"영상 컨텐츠 추천: {summary}. 추천 컨텐츠: {formatted_contents}"
        
        elif parsed_model_response.get("recommendation_type") == "general_text":
            return parsed_model_response.get("response_text", "이전 대화 응답입니다.")
        
        else:
            return f"[알 수 없는 타입] {model_response_str[:100]}..." # 알 수 없는 JSON이면 잘라서 전달
    except json.JSONDecodeError:
        return model_response_str # JSON이 아니면 원본 문자열 그대로 사용
    except Exception as e:
        if DEBUG_MODE:
            print(f"[경고] 모델 이력 포맷팅 중 오류 발생: {e} - 원본: {model_response_str[:100]}...")
        return f"[오류 발생 응답] {model_response_str[:100]}..."


def build_chat_messages(user_input_text, conv_history, summary_text, enriched_watch_history):
    """Gemini API에 보낼 메시지 리스트를 구성합니다."""
    messages_for_api = []

    # 현재 시각 포함
    current_turn_user_prompt_parts = []
    now = datetime.datetime.now()
    current_time_str = now.strftime("%Y년 %m월 %d일 %A %p %I:%M")
    current_turn_user_prompt_parts.append(f"참고: 현재 시각은 {current_time_str} 입니다.")

    # 과거 시청 이력 포함
    if enriched_watch_history:
        try:
            # 최근 본 영화 20개 (제목만)
            watched_titles = [entry["title"] for entry in enriched_watch_history if "title" in entry][:20]
            if watched_titles:
                watched_text = ", ".join(watched_titles)
                current_turn_user_prompt_parts.append(f"과거 시청 이력: {watched_text}")

            # 선호 장르 (가장 많이 본 3개 장르)
            genre_list = []
            for entry in enriched_watch_history:
                genre = entry.get("genre")
                if isinstance(genre, list):
                    genre_list.extend(genre)
                elif isinstance(genre, str):
                    genre_list.append(genre)

            genre_counter = Counter(genre_list)
            top_genres = [f"{genre}({count})" for genre, count in genre_counter.most_common(3)]
            if top_genres:
                genre_text = ", ".join(top_genres)
                current_turn_user_prompt_parts.append(f"선호 장르: {genre_text}")

        except Exception as e:
            print(f"[경고] 시청 이력 처리 오류: {e}")

    # 이전 대화 요약 포함
    if summary_text:
        messages_for_api.append({"role": "user", "parts": [f"이전 대화 요약: {summary_text}"]})

        # 최근 대화 5턴 포함 (모델에게 이전 대화 맥락 제공)
    for entry in conv_history[-5:]: # 최근 5턴만 사용 (토큰 제한 고려)
        messages_for_api.append({"role": "user", "parts": [entry["user"]]})
        # history에 JSON 문자열로 저장된 모델 응답을 텍스트로 변환하여 전달
        messages_for_api.append({"role": "model", "parts": [format_model_response_for_history(entry["model"])]})

    messages_for_api.append({"role": "user", "parts": [f"사용자 질문:\n{user_input_text}"]})

    print("\n\n\n 메시지 에이피아이")
    print(messages_for_api)
    print("\n\n\n")
    return messages_for_api


# --- 메인 실행 로직 ---
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
        # 10턴마다 또는 10턴 이후 15턴마다 요약
        needs_summary = (
            turn_count == 10 or (turn_count > 10 and (turn_count - 10) % 15 == 0)
        )

        if needs_summary:
            print("\n[시스템] 대화 요약 중...")
            new_summary = summarize_conversation_history(history)
            if new_summary:
                current_summary = new_summary
                if DEBUG_MODE:
                    print(f"[시스템] 새로운 대화 요약: {current_summary[:100]}...") # 요약 내용 미리보기

        # 모델에게 보낼 메시지 구성
        messages = build_chat_messages(user_input, history, current_summary, load_enriched_watch_history())

        try:
            response = chat_model.generate_content(messages)
            reply_json_str = response.text.strip() # 모델은 JSON 문자열을 반환

            if DEBUG_MODE:
                print(f"[Gemini Raw JSON 응답]: {reply_json_str}")

            try:
                reply_data = json.loads(reply_json_str)

                # 응답 데이터 유효성 검사
                if "recommendation_type" not in reply_data:
                    print(f"Gemini: [응답 형식 오류] 'recommendation_type' 필드가 누락되었습니다. 원본 응답:\n{reply_json_str}\n")
                    history.append({"user": user_input, "model": "모델 응답 형식 오류: " + reply_json_str}) # 오류도 history에 저장
                    continue

                if reply_data["recommendation_type"] == "media_content":
                    summary = reply_data.get("recommendation_reason_summary", "추천 이유를 요약할 수 없습니다.")
                    contents = reply_data.get("recommended_contents", [])

                    print("\n✨ Gemini: 당신을 위한 영상 미디어 컨텐츠 추천입니다! ✨")
                    print(f"**추천 이유**: {summary}")
                    print("\n--- 추천 목록 ---")
                    if not contents:
                        print("추천할 컨텐츠가 없습니다.")
                    else:
                        for i, item in enumerate(contents):
                            title = item.get("title", "제목 없음")
                            reason = item.get("reason", "사유 없음")
                            print(f"{i+1}. **{title}**: {reason}")
                            if i >= 9: break # 최대 10개 출력 제한
                    print("-----------------\n")

                    # history에는 모델의 원본 JSON 응답을 저장
                    history.append({"user": user_input, "model": reply_json_str})

                elif reply_data["recommendation_type"] == "general_text":
                    response_text = reply_data.get("response_text", "응답을 생성할 수 없습니다.")
                    print(f"Gemini: {response_text}\n")
                    # history에는 일반 텍스트 응답을 저장
                    history.append({"user": user_input, "model": json.dumps({"recommendation_type": "general_text", "response_text": response_text}, ensure_ascii=False)})

                else:
                    print(f"Gemini: [알 수 없는 추천 타입] 응답: {reply_json_str}\n")
                    history.append({"user": user_input, "model": reply_json_str})


            except json.JSONDecodeError:
                print(f"Gemini: [JSON 파싱 오류] 모델 응답이 유효한 JSON 형식이 아닙니다. 원본 응답:\n{reply_json_str}\n")
                history.append({"user": user_input, "model": "JSON 파싱 실패: " + reply_json_str})
            except Exception as e:
                print(f"Gemini: [응답 처리 중 알 수 없는 오류 발생] {e}. 원본 응답:\n{reply_json_str}\n")
                history.append({"user": user_input, "model": "응답 처리 오류: " + reply_json_str})

        except Exception as e:
            print(f"[오류] API 호출 중 오류 발생: {e}\n")
            history.append({"user": user_input, "model": f"API 호출 오류: {e}"}) # API 오류도 history에 기록

except KeyboardInterrupt:
    print("\n[시스템] 종료 신호 감지됨. 저장 후 종료합니다.")
    save_context_to_file(CONTEXT_FILE, history, current_summary, turn_count)