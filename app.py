import os
from dotenv import load_dotenv
import google.generativeai as genai
import datetime
import json
from collections import Counter
import re

from flask import Flask, render_template, request, jsonify

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    print("GOOGLE_API_KEY 환경 변수를 찾을 수 없습니다.")
    exit()
genai.configure(api_key=api_key)

MODEL_NAME = "gemini-2.0-flash"
DEBUG_MODE = True # Enable debug mode for the web app
CONTEXT_FILE = "conversation_context.json"
WATCH_HISTORY_FILE = "watch_history.json"
CONTENTS_FILE = "contents.json"

file = open("system_prompt.txt", "r", encoding="utf-8")
CHATBOT_ROLE_INSTRUCTION = file.read()
file.close()

try:
    json_schema = {
        "type": "object",
        "properties": {
            "recommendation_type": {"type": "string", "enum": ["media_content", "general_text", "nsfw_text"]},
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
            "response_schema": json_schema
        }
    )
    summarizer_model = genai.GenerativeModel(MODEL_NAME)
except Exception as e:
    print(f"모델 ({MODEL_NAME}) 초기화 중 오류 발생: {e}")
    exit()

app = Flask(__name__)

# Global variables for conversation context (for simplicity in this example)
history = []
current_summary = ""
turn_count = 0
debug_messages = [] # To store debug messages for the frontend

def log_debug(message):
    """Logs a debug message and adds it to the debug_messages list."""
    timestamp = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    full_message = f"{timestamp} {message}"
    debug_messages.append(full_message)
    if DEBUG_MODE:
        print(full_message)

def load_json_file(file_path):
    """JSON 파일을 로드합니다."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        log_debug(f"[경고] 파일이 없습니다: {file_path}")
        return {}
    except json.JSONDecodeError:
        log_debug(f"[오류] JSON 디코딩 오류: {file_path}")
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

        log_debug("[과거 시청 이력 불러오기 완료]")
        return enriched_history
    except Exception as e:
        log_debug(f"[오류] 과거 시청 이력 로드 중 오류 발생: {e}")
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
    log_debug(f"[시스템] 대화 컨텍스트가 '{filepath}'에 저장되었습니다.")

def load_context_from_file(filepath):
    """파일에서 대화 컨텍스트를 로드합니다."""
    try:
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                context_data = json.load(f)
                log_debug(f"[시스템] 대화 컨텍스트를 '{filepath}'에서 불러왔습니다.")
                return (
                    context_data.get("history", []),
                    context_data.get("current_summary", ""),
                    context_data.get("turn_count", 0),
                )
    except Exception as e:
        log_debug(f"[시스템] 대화 내용 불러오기 중 오류 발생: {e}")
    log_debug("[시스템] 새로운 대화를 시작합니다. (이전 컨텍스트 없음)")
    return [], "", 0

def summarize_conversation_history(full_conv_history):
    """대화 이력을 요약합니다."""
    formatted_history_for_summary = []
    for entry in full_conv_history:
        user_text = entry["user"]
        model_text = entry["model"]
        try:
            parsed_model_response = json.loads(model_text)
            if parsed_model_response.get("recommendation_type") == "media_content":
                summary_text = parsed_model_response.get("recommendation_reason_summary", "")
                titles = [item.get("title", "") for item in parsed_model_response.get("recommended_contents", [])]
                model_text = f"영상 추천: {summary_text}. 컨텐츠: {', '.join(titles[:3])}..."  # Limit titles for summary
            elif parsed_model_response.get("recommendation_type") == "general_text":
                model_text = parsed_model_response.get("response_text", "")
            elif parsed_model_response.get("recommendation_type") == "nsfw_text":
                 model_text = parsed_model_response.get("response_text", "정책 위반 응답") # For summary, just indicate it was a policy response
        except json.JSONDecodeError:
            pass

        formatted_history_for_summary.append(f"사용자: {user_text}\nVAC: {model_text}")

    if not formatted_history_for_summary or len(formatted_history_for_summary) < 5:
        return ""
    
    summary_instruction = "다음 대화 내용을 요약해줘. 중요한 대화 주제와 사용자의 요청 사항을 중심으로 요약해."
    formatted = "\n".join(formatted_history_for_summary)
    prompt = f"{summary_instruction}\n---\n{formatted}\n---"
    try:
        response = summarizer_model.generate_content([{"role": "user", "parts": [prompt]}])
        return response.text.strip()
    except Exception as e:
        log_debug(f"[오류] 요약 중 오류 발생: {e}")
        return ""

def format_model_response_for_history(model_response_str):
    """history에 저장된 모델 응답 (JSON)을 모델이 이해할 수 있는 텍스트로 변환합니다."""
    try:
        parsed_model_response = json.loads(model_response_str)
        if parsed_model_response.get("recommendation_type") == "media_content":
            summary = parsed_model_response.get("recommendation_reason_summary", "추천 이유 요약 없음")
            contents = parsed_model_response.get("recommended_contents", [])
            titles = [item.get("title", "제목 없음") for item in contents]
            
            formatted_contents = ", ".join(titles[:5])
            if len(titles) > 5:
                formatted_contents += " 등"
            
            return f"영상 컨텐츠 추천: {summary}. 추천 컨텐츠: {formatted_contents}"
        
        elif parsed_model_response.get("recommendation_type") == "general_text":
            return parsed_model_response.get("response_text", "이전 대화 응답입니다.")
        
        elif parsed_model_response.get("recommendation_type") == "nsfw_text":
            return parsed_model_response.get("response_text", "정책 위반 응답")
        
        else:
            return f"[알 수 없는 타입] {model_response_str[:100]}..."  # Truncate for brevity
    except json.JSONDecodeError:
        return model_response_str  # Return original if not JSON
    except Exception as e:
        log_debug(f"[경고] 모델 이력 포맷팅 중 오류 발생: {e} - 원본: {model_response_str[:100]}...")
        return f"[오류 발생 응답] {model_response_str[:100]}..."


def build_chat_messages(user_input_text, conv_history, summary_text, enriched_watch_history):
    """Gemini API에 보낼 메시지 리스트를 구성합니다."""
    messages_for_api = []

    current_turn_user_prompt_parts = []
    now = datetime.datetime.now()
    current_time_str = now.strftime("%Y년 %m월 %d일 %A %p %I:%M")
    current_turn_user_prompt_parts.append(f"참고: 현재 시각은 {current_time_str} 입니다.")

    if enriched_watch_history:
        try:
            watched_titles = [entry["title"] for entry in enriched_watch_history if "title" in entry][:20]
            if watched_titles:
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
            if top_genres:
                genre_text = ", ".join(top_genres)
                current_turn_user_prompt_parts.append(f"선호 장르: {genre_text}")

        except Exception as e:
            log_debug(f"[경고] 시청 이력 처리 오류: {e}")

    if summary_text:
        messages_for_api.append({"role": "user", "parts": [f"이전 대화 요약: {summary_text}"]})

    # Include the last 5 turns of conversation history in the prompt
    for entry in conv_history[-5:]:
        messages_for_api.append({"role": "user", "parts": [entry["user"]]})
        messages_for_api.append({"role": "model", "parts": [format_model_response_for_history(entry["model"])]})

    # Add the current user input as the last message
    current_user_prompt = "\n".join(current_turn_user_prompt_parts + [f"사용자 질문:\n{user_input_text}"])
    messages_for_api.append({"role": "user", "parts": [current_user_prompt]})
    
    return messages_for_api

@app.route("/")
def index():
    """Renders the main chat interface."""
    global history, current_summary, turn_count, debug_messages
    history, current_summary, turn_count = load_context_from_file(CONTEXT_FILE)
    debug_messages.clear() # Clear debug messages on page load
    log_debug(f"{MODEL_NAME} 초기화 완료. 이전 대화 수: {len(history)}")
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    """Handles chat requests from the frontend."""
    global history, current_summary, turn_count, debug_messages
    user_input = request.json.get("message").strip()

    if not user_input:
        return jsonify({"response_html": "<p>Please enter a message.</p>", "debug_log": debug_messages})

    turn_count += 1
    needs_summary = (
        turn_count == 10 or (turn_count > 10 and (turn_count - 10) % 15 == 0)
    )

    if needs_summary:
        log_debug("[시스템] 대화 요약 중...")
        new_summary = summarize_conversation_history(history)
        if new_summary:
            current_summary = new_summary
            log_debug(f"[시스템] 새로운 대화 요약: {current_summary[:100]}...")

    # Build messages for the API call
    messages = build_chat_messages(user_input, history, current_summary, load_enriched_watch_history())
    
    response_html = ""
    try:
        response = chat_model.generate_content(messages)
        reply_json_str = response.text.strip()
        log_debug(f"[Gemini Raw JSON 응답]: {reply_json_str}")

        try:
            reply_data = json.loads(reply_json_str)

            if "recommendation_type" not in reply_data:
                response_html = "<p class='error-message'>VAC: [응답 형식 오류] 'recommendation_type' 필드가 누락되었습니다.</p>"
                log_debug(f"VAC: [응답 형식 오류] 'recommendation_type' 필드가 누락되었습니다. 원본 응답:\n{reply_json_str}")
            elif reply_data["recommendation_type"] == "media_content":
                summary = reply_data.get("recommendation_reason_summary", "추천 이유를 요약할 수 없습니다.")
                contents = reply_data.get("recommended_contents", [])

                response_html += "<p class='vac-response'>✨ VAC: 당신을 위한 영상 미디어 컨텐츠 추천입니다!</p>"
                response_html += f"<p class='vac-response'><strong>추천 이유</strong>: {summary}</p>"
                response_html += "<div class='recommendation-list'>"
                if not contents:
                    response_html += "<p>추천할 컨텐츠가 없습니다.</p>"
                else:
                    for i, item in enumerate(contents):
                        title = item.get("title", "제목 없음")
                        reason = item.get("reason", "사유 없음")
                        response_html += f"<p>{i+1}. <strong>{title}</strong>: {reason}</p>"
                        if i >= 9: break # Limit to 10 for display
                response_html += "</div>"
                history.append({"user": user_input, "model": reply_json_str})

            elif reply_data["recommendation_type"] == "general_text":
                response_text = reply_data.get("response_text", "응답을 생성할 수 없습니다.")
                response_html += f"<p class='vac-response'>VAC: {response_text}</p>"
                history.append({"user": user_input, "model": json.dumps({"recommendation_type": "general_text", "response_text": response_text}, ensure_ascii=False)})

            elif reply_data["recommendation_type"] == "nsfw_text":
                response_text = reply_data.get("response_text", "말씀하신 내용에 당사의 정책에 위배될 수 있는 문구가 포함되어 있습니다. VAC는 여러분에게 도움을 주기 위해 존재하는 만큼 건전하고 올바른 사용을 통해 건실한 대화를 이어가면 좋겠습니다.")
                response_html += f"<p class='vac-response policy-warning'>VAC: {response_text}</p>"
                history.append({"user": user_input, "model": json.dumps({"recommendation_type": "nsfw_text", "response_text": response_text}, ensure_ascii=False)})

            else:
                response_html = f"<p class='error-message'>VAC: [알 수 없는 추천 타입] 응답: {reply_json_str}</p>"
                log_debug(f"VAC: [알 수 없는 추천 타입] 응답: {reply_json_str}")
                history.append({"user": user_input, "model": reply_json_str})

        except json.JSONDecodeError:
            response_html = f"<p class='error-message'>VAC: [JSON 파싱 오류] 모델 응답이 유효한 JSON 형식이 아닙니다. 원본 응답:\n{reply_json_str}</p>"
            log_debug(f"VAC: [JSON 파싱 오류] 모델 응답이 유효한 JSON 형식이 아닙니다. 원본 응답:\n{reply_json_str}")
        except Exception as e:
            response_html = f"<p class='error-message'>VAC: [응답 처리 중 알 수 없는 오류 발생] {e}</p>"
            log_debug(f"VAC: [응답 처리 중 알 수 없는 오류 발생] {e}. 원본 응답:\n{reply_json_str}")

    except Exception as e:
        response_html = f"<p class='error-message'>VAC: [API 호출 중 오류 발생] {e}</p>"
        log_debug(f"[오류] API 호출 중 오류 발생: {e}")
    
    # Save context after each successful turn
    save_context_to_file(CONTEXT_FILE, history, current_summary, turn_count)
    
    return jsonify({"response_html": response_html, "debug_log": debug_messages})

if __name__ == "__main__":
    history, current_summary, turn_count = load_context_from_file(CONTEXT_FILE)
    # app.run(debug=DEBUG_MODE)
    app.run(host='0.0.0.0', port=7777, debug=DEBUG_MODE)
