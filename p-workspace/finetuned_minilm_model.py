import mysql.connector
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import random
import json

# 1. DB 연결
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="onlyroot",
    database="review_similarity",
    auth_plugin='mysql_native_password'
)
cursor = db.cursor(dictionary=True)

# 2. 광고성/비광고성 리뷰 가져오기
cursor.execute("SELECT id, cleaned_review, label FROM reviews WHERE cleaned_review IS NOT NULL")
rows = cursor.fetchall()

ads = [r for r in rows if r["label"] == 1]   # 광고성 리뷰
non_ads = [r for r in rows if r["label"] == 0]  # 비광고성 리뷰

print(f"광고성 리뷰 {len(ads)}개, 비광고성 리뷰 {len(non_ads)}개 로드 완료")

# ✅ 3. 학습용 데이터셋 
train_examples = []

# 광고성-광고성 (유사도=1.0)
for _ in range(len(ads)//2):
    a, b = random.sample(ads, 2)
    train_examples.append(InputExample(texts=[a["cleaned_review"], b["cleaned_review"]], label=1.0))

# 비광고성-비광고성 (유사도=1.0)
for _ in range(len(non_ads)//2):
    a, b = random.sample(non_ads, 2)
    train_examples.append(InputExample(texts=[a["cleaned_review"], b["cleaned_review"]], label=1.0))

# 광고성-비광고성 (유사도=0.0)
for _ in range(min(len(ads), len(non_ads))):
    a = random.choice(ads)
    b = random.choice(non_ads)
    train_examples.append(InputExample(texts=[a["cleaned_review"], b["cleaned_review"]], label=0.0))

print(f"총 학습 샘플: {len(train_examples)} 개")

# 4. DataLoader 준비
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)

# 5. 기본 모델 로드
model = SentenceTransformer("all-MiniLM-L6-v2")

# 6. 학습 손실 함수 (Cosine Similarity Loss)
train_loss = losses.CosineSimilarityLoss(model)

# 7. 학습 시작
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=3,
    warmup_steps=int(len(train_dataloader) * 0.1),
    output_path="./finetuned_minilm_model"   # 결과 모델 저장 경로
)

print("✅ 파인튜닝 완료! ./finetuned_minilm_model 폴더에 저장됨")

cursor.close()
db.close()
