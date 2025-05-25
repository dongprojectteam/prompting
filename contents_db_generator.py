import json
import random

title_prefix = ["나의", "우리들의", "그날의", "마지막", "조용한", "뜨거운", "은밀한", "기묘한", "비밀의", "하루의"]
title_subject = ["기억", "여행", "복수", "사랑", "의사생활", "이야기", "범죄", "감정", "우정", "계획"]
title_suffix = ["일기", "리포트", "파일", "시즌", "연대기", "사건", "보고서", "극장판", "편지", "시"]

genres = [["드라마"], ["예능"], ["액션"], ["스릴러"], ["코미디"], ["범죄", "액션"], ["힐링", "예능"]]
moods = ["따뜻함", "긴장감", "편안함", "자극적", "감성적"]
tags_pool = ["우정", "복수", "추리", "사랑", "여행", "형사", "감정", "학폭", "직장", "한옥"]
platforms = ["넷플릭스", "티빙", "왓챠", "웨이브", "디즈니플러스"]
actors_pool = ["조정석", "송혜교", "마동석", "전미도", "이도현", "김태리", "남궁민", "정해인"]
directors_pool = ["신원호", "나영석", "안길호", "이응복", "박찬욱"]

contents = []
for i in range(200):
    genres_selected = random.choice(genres)
    moods_selected = random.sample(moods, 2)
    tags_selected = random.sample(tags_pool, 3)
    actors = random.sample(actors_pool, 2)
    director = random.choice(directors_pool)
    platform = random.choice(platforms)

    title = f"{random.choice(title_prefix)} {random.choice(title_subject)}의 {random.choice(title_suffix)}"

    contents.append({
        "content_id": f"c{i+1:03d}",
        "title": title,
        "genre": genres_selected,
        "mood": moods_selected,
        "actors": actors,
        "director": director,
        "platform": platform,
        "release_year": random.randint(2010, 2024),
        "tags": tags_selected,
        "duration_min": random.choice([60, 70, 90, 100, 120])
    })

with open("contents.json", "w", encoding="utf-8") as f:
    json.dump(contents, f, ensure_ascii=False, indent=2)

