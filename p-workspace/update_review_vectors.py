import argparse
import mysql.connector
import json
from sentence_transformers import SentenceTransformer

def main():
    parser = argparse.ArgumentParser(description="Update missing review vectors. Optionally specify --category")
    parser.add_argument("--category", "-c", type=str, default=None, help="카테고리 이름 (예: '패션잡화'). 지정하지 않으면 전체 대상.")
    args = parser.parse_args()

    model = SentenceTransformer("./finetuned_minilm_model")

    db = mysql.connector.connect(
        host="localhost",
        user="root",
        password="onlyroot",
        database="review_similarity",
        auth_plugin='mysql_native_password',
        charset='utf8mb4'
    )
    cursor = db.cursor(dictionary=True)

    if args.category:
        cursor.execute("""
            SELECT id, cleaned_review 
            FROM reviews 
            WHERE (review_vector IS NULL OR review_vector = '')
              AND category = %s
        """, (args.category,))
    else:
        cursor.execute("""
            SELECT id, cleaned_review 
            FROM reviews 
            WHERE (review_vector IS NULL OR review_vector = '')
        """)
    rows = cursor.fetchall()
    print(f"총 {len(rows)}개 리뷰 벡터 생성 예정" + (f" (카테고리: {args.category})" if args.category else ""))

    batch = []
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

        # optional: commit periodically
        if count % 100 == 0:
            db.commit()
            print(f"  - {count}개 커밋 완료")

    db.commit()
    cursor.close()
    db.close()
    print(f"✅ 벡터 생성 완료: 총 {count}개 업데이트 완료")

if __name__ == "__main__":
    main()
