import random
import datetime
import json
import os

WATCH_HISTORY_FILE = "watch_history.json"


def load_watch_history(filepath=WATCH_HISTORY_FILE):
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_watch_history(history, filepath=WATCH_HISTORY_FILE):
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def add_watch_log(
    user_id,
    content_id,
    watched_duration,
    feedback="like",
    time_context="저녁",
    emotion="보통",
):
    history = load_watch_history()
    log_entry = {
        "content_id": content_id,
        "watched_at": datetime.datetime.now().isoformat(),
        "feedback": feedback,
        "watched_duration": watched_duration,
        "context": {"time": time_context, "emotion": emotion},
    }

    if user_id not in history:
        history[user_id] = []

    history[user_id].append(log_entry)
    save_watch_history(history)

    print(
        f"[시스템] 사용자 {user_id}의 시청 이력이 저장되었습니다: 콘텐츠 {content_id}"
    )


def generate_random_watch_logs(user_id="user_001", count=20):
    # 콘텐츠 로드
    with open("contents.json", "r", encoding="utf-8") as f:
        contents = json.load(f)

    if len(contents) < count:
        print(f"콘텐츠가 {count}개 미만입니다. 먼저 contents.json을 채워주세요.")
        return

    selected_contents = random.sample(contents, count)

    feedback_options = ["like", "neutral", "dislike"]
    emotions = ["행복함", "지루함", "우울함", "피곤함", "기대됨"]
    time_contexts = ["아침", "점심", "저녁", "심야", "주말 낮", "주말 밤"]

    for content in selected_contents:
        watched_duration = random.randint(
            int(content["duration_min"] * 0.8), content["duration_min"]
        )

        add_watch_log(
            user_id=user_id,
            content_id=content["content_id"],
            watched_duration=watched_duration,
            feedback=random.choices(feedback_options, weights=[0.6, 0.3, 0.1])[0],
            time_context=random.choice(time_contexts),
            emotion=random.choice(emotions),
        )

if __name__ == "__main__":
    generate_random_watch_logs(user_id="김동혁", count=20)
