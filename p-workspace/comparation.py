import mysql.connector
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import ttest_ind
import random
import json
import os

# ============================
# DB 설정
# ============================
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "onlyroot",
    "database": "comparation",
    "auth_plugin": 'mysql_native_password',
    "charset": 'utf8mb4'
}

# ============================
# 키워드 가중치 로드
# ============================
with open("keyword_weights.json", "r", encoding="utf-8") as f:
    weights_data = json.load(f)

AD_WEIGHTS = weights_data["AD_WEIGHTS"]
NON_AD_WEIGHTS = weights_data["NON_AD_WEIGHTS"]

# ============================
# 키워드 가중치 적용 임베딩 함수
# ============================
def compute_weighted_embeddings(reviews, model, weight_dict):
    embeddings = []
    for review in reviews:
        emb = model.encode(review, convert_to_numpy=True)
        weight = 1.0
        for kw, w in weight_dict.items():
            if kw in review:
                weight += w
        embeddings.append(emb * weight)
    return np.array(embeddings)

# ============================
# 파인튜닝
# ============================
def finetune_model(db_config, output_model_path="./finetuned_minilm_model", epochs=3):
    db = mysql.connector.connect(**db_config)
    cursor = db.cursor(dictionary=True)
    cursor.execute("SELECT id, cleaned_review, label FROM reviews WHERE cleaned_review IS NOT NULL")
    rows = cursor.fetchall()
    cursor.close()
    db.close()

    ads = [r for r in rows if r["label"] == 1]
    non_ads = [r for r in rows if r["label"] == 0]

    train_examples = []

    # 광고-광고 유사도=1.0
    for _ in range(len(ads)//2):
        a, b = random.sample(ads, 2)
        train_examples.append(InputExample(texts=[a["cleaned_review"], b["cleaned_review"]], label=1.0))

    # 비광고-비광고 유사도=1.0
    for _ in range(len(non_ads)//2):
        a, b = random.sample(non_ads, 2)
        train_examples.append(InputExample(texts=[a["cleaned_review"], b["cleaned_review"]], label=1.0))

    # 광고-비광고 유사도=0.0
    for _ in range(min(len(ads), len(non_ads))):
        a = random.choice(ads)
        b = random.choice(non_ads)
        train_examples.append(InputExample(texts=[a["cleaned_review"], b["cleaned_review"]], label=0.0))

    print(f"총 학습 샘플: {len(train_examples)} 개")

    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    train_loss = losses.CosineSimilarityLoss(model)

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        warmup_steps=int(len(train_dataloader) * 0.1),
        output_path=output_model_path
    )

    print(f"✅ 파인튜닝 완료! 모델 저장 경로: {output_model_path}")
    return model

# ============================
# 리뷰 벡터 DB 업데이트 (이미 있는 벡터 건너뛰기)
# ============================
def update_review_vectors(db_config, model):
    db = mysql.connector.connect(**db_config)
    cursor = db.cursor(dictionary=True)
    
    cursor.execute("SELECT id, cleaned_review FROM reviews WHERE review_vector IS NULL OR review_vector = ''")
    rows = cursor.fetchall()
    print(f"총 {len(rows)}개 리뷰 벡터 생성 예정")

    count = 0
    for row in rows:
        review_id = row["id"]
        text = (row["cleaned_review"] or "").strip()
        if not text:
            continue
        vector = model.encode(text).tolist()
        vector_json = json.dumps(vector, ensure_ascii=False)
        cursor.execute("UPDATE reviews SET review_vector = %s WHERE id = %s", (vector_json, review_id))
        count += 1
        if count % 100 == 0:
            db.commit()
            print(f"  - {count}개 커밋 완료")
    db.commit()
    cursor.close()
    db.close()
    print(f"✅ 벡터 생성 완료: 총 {count}개 업데이트 완료")

# ============================
# 키워드 가중치 적용 평균 유사도 계산
# ============================
def run_weighted_similarity_analysis_from_db(db_config, model):
    db = None
    try:
        db = mysql.connector.connect(**db_config)
        cursor = db.cursor(dictionary=True)
        cursor.execute("SELECT cleaned_review, label FROM reviews WHERE cleaned_review IS NOT NULL")
        rows = cursor.fetchall()
        cursor.close()

        data = pd.DataFrame(rows)
        data = data.dropna(subset=['cleaned_review', 'label'])
        data['label_text'] = data['label'].astype(str).replace({'1': '광고', '0': '비광고'})

        ads = data[data["label_text"] == "광고"]["cleaned_review"].astype(str).tolist()
        non_ads = data[data["label_text"] == "비광고"]["cleaned_review"].astype(str).tolist()

        print(f"✅ DB에서 데이터 로드 완료.")
        print(f"광고 리뷰 수: {len(ads)}개, 비광고 리뷰 수: {len(non_ads)}개")
        if len(ads) < 2 or len(non_ads) < 2:
            print("데이터 부족")
            return

    except Exception as e:
        print(f"❌ DB 오류: {e}")
        return
    finally:
        if db and db.is_connected():
            db.close()

    try:
        print("\n⏳ 파인튜닝 모델 로드 및 임베딩 + 키워드 가중치 적용")
        emb_ads = compute_weighted_embeddings(ads, model, AD_WEIGHTS)
        emb_non_ads = compute_weighted_embeddings(non_ads, model, NON_AD_WEIGHTS)

        mean_ads = np.mean(emb_ads, axis=0)
        mean_non_ads = np.mean(emb_non_ads, axis=0)

        sim_ads = cosine_similarity(emb_ads, [mean_ads])[:, 0]
        sim_non_ads = cosine_similarity(emb_non_ads, [mean_non_ads])[:, 0]

        mean_ads_score = np.mean(sim_ads)
        mean_non_ads_score = np.mean(sim_non_ads)
        std_ads = np.std(sim_ads)
        std_non_ads = np.std(sim_non_ads)

        print("\n📈 키워드 가중치 + 파인튜닝 후 평균 유사도")
        print(f"광고 리뷰 평균 유사도: {mean_ads_score:.3f} ± {std_ads:.3f}")
        print(f"비광고 리뷰 평균 유사도: {mean_non_ads_score:.3f} ± {std_non_ads:.3f}")
        print(f"차이 (광고 - 비광고): {mean_ads_score - mean_non_ads_score:.3f}")

        t_stat, p_value = ttest_ind(sim_ads, sim_non_ads, equal_var=False)
        print("\n🧮 Welch's t-test")
        print(f"t 통계량: {t_stat:.3f}, p-value: {p_value:.6f}")

        n1, n2 = len(sim_ads), len(sim_non_ads)
        s_pooled = np.sqrt(((n1-1)*std_ads**2 + (n2-1)*std_non_ads**2) / (n1 + n2 - 2))
        cohen_d = (mean_ads_score - mean_non_ads_score) / s_pooled
        print(f"Cohen's d: {cohen_d:.3f}")

    except Exception as e:
        print(f"❌ 유사도 분석 오류: {e}")

# ============================
# 실행
# ============================
if __name__ == "__main__":
    # 1️⃣ 파인튜닝 모델 학습 (이미 학습한 모델 있으면 skip 가능)
    if os.path.exists("./finetuned_minilm_model"):
        print("✅ 기존 파인튜닝 모델 로드")
        model = SentenceTransformer("./finetuned_minilm_model")
    else:
        model = finetune_model(DB_CONFIG, output_model_path="./finetuned_minilm_model", epochs=3)

    # 2️⃣ 벡터 업데이트
    update_review_vectors(DB_CONFIG, model)

    # 3️⃣ 키워드 가중치 적용 평균 유사도 계산
    run_weighted_similarity_analysis_from_db(DB_CONFIG, model)
