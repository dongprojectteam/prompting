import os
from dotenv import load_dotenv
import google.generativeai as genai
import datetime
import json

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    print("GOOGLE_API_KEY 환경 변수를 찾을 수 없습니다.")
    exit()
genai.configure(api_key=api_key)

MODEL_NAME = "gemini-2.0-flash"  # 시뮬레이션을 위한 모델 - VAC와 동일하게 사용용
DEBUG_MODE = True
CONTEXT_FILE = "conversation_context.json" # 컨텍스트 저장 파일명

# 챗봇의 기본 역할 정의 (콘텐츠 추천 로직 강화 및 일관성 확보)
CHATBOT_ROLE_INSTRUCTION = """너는 사용자의 콘텐츠 시청 습관과 선호도를 파악하여 맞춤형 콘텐츠를 추천하는 AI 비서다.
사용자의 질문에 대해 이전 대화 내용과 요약본을 참고하여 답변한다.
답변은 항상 명확하고 핵심적인 정보를 전달해야 하며, 사용자의 지시를 충실히 따른다.

특히 사용자가 콘텐츠 추천을 원하거나 관련된 대화를 할 때 다음 지침을 따른다:
1. 콘텐츠 추천 시에는 항상 10개의 새로운 콘텐츠를 추천하는 것을 목표로 한다.
2. 만약 이전 대화나 요약에서 사용자가 좋아한다고 언급한 프로그램이 있다면 (예: "나는 [X]를 좋아해", "[X] 재미있었어" 등), 해당 프로그램들의 장르, 분위기, 출연진, 제작진 등을 종합적으로 분석하여 사용자가 좋아할 만한 새로운 콘텐츠를 추천한다. 추천 시에는 어떤 점을 바탕으로 추천했는지 간략히 언급할 수 있다.
3. 사용자가 좋아한다고 언급한 프로그램에 대한 정보가 대화 내역이나 요약에서 명확히 찾아볼 수 없다면, 제공되는 '현재 시각' 정보를 참고하여 그 시간에 어울리는 콘텐츠를 추천한다. 예를 들어, 저녁 시간대에는 편안하게 볼 수 있는 드라마나 영화, 주말 오전에는 가벼운 예능 등을 추천할 수 있다.
4. 콘텐츠 추천 시, 사용자에게 추가적인 질문을 하기보다는 이미 주어진 정보(대화 이력, 요약, 현재 시각)를 최대한 활용하여 먼저 추천을 제시한다. 사용자가 나중에 더 구체적인 정보를 주면 그에 맞춰 추천을 조절할 수 있다.
5. 일반적인 답변은 간결하게 유지하되, 위와 같은 콘텐츠 추천의 경우에는 답변이 길어지더라도 충분한 정보를 제공한다.
"""

try:
    chat_model = genai.GenerativeModel(
        MODEL_NAME, system_instruction=CHATBOT_ROLE_INSTRUCTION
    )
    summarizer_model = genai.GenerativeModel(MODEL_NAME)
except Exception as e:
    print(f"모델 ({MODEL_NAME}) 초기화 중 오류 발생: {e}")
    print("올바른 모델 이름을 사용하고 있는지, API 키가 유효한지 확인해주세요.")
    exit()

def save_context_to_file(filepath, history_data, summary_data, tc_data):
    context_data = {
        "history": history_data,
        "current_summary": summary_data,
        "turn_count": tc_data
    }
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(context_data, f, ensure_ascii=False, indent=4)
        print(f"\n[시스템] 대화 내용이 '{filepath}'에 저장되었습니다.")
    except Exception as e:
        print(f"\n[시스템] 대화 내용 저장 중 오류 발생: {e}")

def load_context_from_file(filepath):
    try:
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                context_data = json.load(f)
                print(f"[시스템] '{filepath}'에서 이전 대화 내용을 불러왔습니다.")
                return context_data.get("history", []), \
                       context_data.get("current_summary", ""), \
                       context_data.get("turn_count", 0)
        else:
            print(f"[시스템] 저장된 대화 내용 파일('{filepath}')이 없습니다. 새 대화를 시작합니다.")
    except FileNotFoundError:
        print(f"[시스템] 저장된 대화 내용 파일('{filepath}')이 없습니다. 새 대화를 시작합니다.")
    except json.JSONDecodeError:
        print(f"[시스템] '{filepath}' 파일 형식이 올바르지 않습니다. 새 대화를 시작합니다.")
    except Exception as e:
        print(f"[시스템] 대화 내용 불러오기 중 오류 발생 ({type(e).__name__}): {e}. 새 대화를 시작합니다.")
    return [], "", 0 # history, current_summary, turn_count (기본값)

history, current_summary, turn_count = load_context_from_file(CONTEXT_FILE)

WINDOW_SIZE = 5
SUMMARIZE_THRESHOLD_INITIAL = 10
SUMMARIZE_THRESHOLD_RECURRING = 15


def summarize_conversation_history(full_conv_history):
    if (
        not full_conv_history
        or len(full_conv_history) < SUMMARIZE_THRESHOLD_INITIAL / 2
    ):
        return ""
    print(f"[정보] {len(full_conv_history)}개의 대화 턴에 대한 요약을 시도합니다.")
    summary_instruction = "다음 대화 내용을 이전 대화의 맥락을 파악할 수 있도록 핵심 위주로 간결하게 요약해줘. 대화 형식은 유지하지 말고, 요약된 내용만 전달해줘."
    formatted_history_for_summary = ""
    for entry in full_conv_history:
        formatted_history_for_summary += (
            f"사용자: {entry['user']}\nGemini: {entry['model']}\n"
        )
    prompt_for_summarizer = f"{summary_instruction}\n\n--- 대화 시작 ---\n{formatted_history_for_summary}\n--- 대화 끝 ---"
    try:
        response = summarizer_model.generate_content(
            [{"role": "user", "parts": [prompt_for_summarizer]}]
        )
        return response.text.strip()
    except Exception as e:
        print(f"요약 중 오류 발생: {e}")
        return ""


def build_chat_messages(user_input_text, conv_history, summary_text):
    messages_for_api = []
    for entry in conv_history[-WINDOW_SIZE:]:
        messages_for_api.append({"role": "user", "parts": [entry["user"]]})
        messages_for_api.append({"role": "model", "parts": [entry["model"]]})

    current_turn_user_prompt_parts = []
    now = datetime.datetime.now()
    current_time_str = now.strftime("%Y년 %m월 %d일 %A %p %I:%M")
    current_turn_user_prompt_parts.append(f"참고: 현재 시각은 {current_time_str} 입니다.")
    if summary_text:
        current_turn_user_prompt_parts.append(
            f"이전 대화의 요약본은 다음과 같아:\n{summary_text}"
        )
    current_turn_user_prompt_parts.append(f"사용자 질문:\n{user_input_text}")
    final_user_content = "\n\n".join(current_turn_user_prompt_parts).strip()
    messages_for_api.append({"role": "user", "parts": [final_user_content]})
    return messages_for_api


print(f"{MODEL_NAME} (챗봇 역할 일부: {CHATBOT_ROLE_INSTRUCTION[:50].replace('\n', ' ')}...)")
print("일반 답변은 간결하게, 콘텐츠 추천 시에는 상세하게 답변합니다.")
if history:
    print(f"이전 대화 {len(history)} 턴이 로드되었습니다. (현재 턴: {turn_count})")

try:
    while True:
        user_input = input("사용자: ")
        if len(user_input.strip()) == 0:
            continue
        
        if user_input.lower() in ("exit", "quit"):
            save_context_to_file(CONTEXT_FILE, history, current_summary, turn_count)
            break

        turn_count += 1

        try:
            needs_summary = False
            if turn_count == SUMMARIZE_THRESHOLD_INITIAL:
                needs_summary = True
            elif (
                turn_count > SUMMARIZE_THRESHOLD_INITIAL
                and (turn_count - SUMMARIZE_THRESHOLD_INITIAL)
                % SUMMARIZE_THRESHOLD_RECURRING
                == 0
            ):
                needs_summary = True

            if needs_summary and history:
                print("\n[시스템] 대화 내용 요약 중...")
                new_summary = summarize_conversation_history(history)
                if new_summary:
                    current_summary = new_summary
                    print(f"[시스템] 새 요약 (일부): {current_summary[:70].replace('\n', ' ')}...\n")
                else:
                    print("[시스템] 요약 생성에 실패했거나 내용이 없습니다.\n")

            messages_to_send = build_chat_messages(user_input, history, current_summary)

            if DEBUG_MODE:
                print("\n[DEBUG] Sending messages to chat_model:")
                for i, msg in enumerate(messages_to_send):
                    part_content = msg['parts'][0].replace('\n', '\\n')
                    print(
                        f"  {i+1}. Role: {msg['role']}, Parts: '{part_content[:100]}...'"
                    )
                print("\n")

            response = chat_model.generate_content(messages_to_send)
            reply = response.text.strip()

            print("Gemini:", reply)

            history.append({"user": user_input, "model": reply})

        except Exception as e:
            print(f"메인 처리 중 오류 발생: {e}")
            if hasattr(e, "response"):
                if hasattr(e.response, "prompt_feedback"):
                    print(f"API Prompt Feedback: {e.response.prompt_feedback}")
                if hasattr(e.response, "candidates"):
                    for candidate in e.response.candidates:
                        if hasattr(candidate, "finish_reason"):
                            print(f"Candidate Finish Reason: {candidate.finish_reason}")
                        if hasattr(candidate, "safety_ratings"):
                            print(f"Candidate Safety Ratings: {candidate.safety_ratings}")
except KeyboardInterrupt: # Ctrl+C 로 종료 시
    print("\n[시스템] 사용자에 의해 프로그램이 중단되었습니다.")
    save_context_to_file(CONTEXT_FILE, history, current_summary, turn_count)
finally: 
    # 어떤 식으로든 루프가 종료될 때 저장을 시도 (Ctrl+C 제외한 예외 발생 시 등)
    # 지금 중요한게 아니니 일단 넘어가자
    pass