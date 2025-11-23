# AI 기반 광고성 리뷰 탐지 웹 서비스 ✨

[![React](https://img.shields.io/badge/React-61DAFB?style=for-the-badge&logo=react&logoColor=white)](https://reactjs.org/) 
[![Java](https://img.shields.io/badge/Java-ED8B00?style=for-the-badge&logo=java&logoColor=white)](https://www.java.com/) 
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/) 
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge)](https://fastapi.tiangolo.com/) 
[![MySQL](https://img.shields.io/badge/MySQL-4479A1?style=for-the-badge&logo=mysql&logoColor=white)](https://www.mysql.com/)  

---

## 📑 Table of Contents
1. [프로젝트 소개](#%F0%9F%93%96-프로젝트-소개)
2. [시연 GIF / 이미지](#%F0%9F%96%B8-시연-gif--이미지)
3. [기술 스택](#%E2%9A%99%EF%B8%8F-기술-스택)
4. [아키텍처 구조도](#%F0%9F%A7%A9-아키텍처-구조도)
5. [폴더 구조](#%F0%9F%93%82-폴더-구조)
6. [주요 기능](#%F0%9F%94%8D-주요-기능)
7. [설치 & 실행 방법](#%F0%9F%92%BB-설치--실행-방법)
8. [Contact](#%F0%9F%93%9E-contact)

---

## 📖 프로젝트 소개
소비자는 제품 구매 전 리뷰를 주요 판단 기준으로 삼지만,  
**광고성 리뷰의 확산**으로 인해 리뷰 신뢰도가 낮아지는 문제가 있습니다.  
이를 해결하기 위해 AI를 활용하여 리뷰의 **진위 여부를 자동 판별**하는 시스템을 개발하였습니다.

---

## 🖼 시연 GIF / 이미지
![서비스 시연 GIF](./assets/demo.gif)  
*사용자가 리뷰를 입력하면 광고성 여부를 자동 분석, 시각화, 요약까지 제공*

---

## ⚙️ 기술 스택
| 구분 | 사용 기술 |
|------|------------|
| **Frontend** | React, HTML, CSS, JavaScript |
| **Backend** | Spring Boot, Java |
| **AI Server** | FastAPI, Python, SentenceTransformer (all-MiniLM-L6-v2), Google Gemini API |
| **Database** | MySQL |
| **Visualization** | Recharts (Pie Chart) |
| **Infra / Tool** | GitHub, Google Cloud, JWT Authentication |

---

## 🧩 아키텍처 구조도

```text
[React]
   │  
   ▼
[Spring Boot] - 인증 처리, DB 조회, FastAPI 통신
   │  
   ▼
[FastAPI] - MiniLM 파인튜닝 + 코사인 유사도 계산 + Gemini 요약
   │  
   ▼
[MySQL] - 리뷰 데이터 저장 / 통계
```
---

---

### 🔄 소프트웨어 흐름도

```text
사용자 입력 (텍스트 or 이미지)
       ↓
리뷰 전송 (React -> Spring Boot)
       ↓
AI 분석 요청 (Spring Boot -> Fast API)
       ↓
유사도 + 가중치 계산
       ↓
결과 반환 및 시각화
       ↓
Gemini API를 통한 요약 제공
```

---

### 🔍 유사도 분석 과정
1. **데이터 수집**
   - 광고성 리뷰 800개, 비광고성 리뷰 567개 수집
3. **모델 사용**
   - `SentenceTransformer: all-MiniLM-L6-v2` 
5. **파인튜닝 & 문장 임베딩**
   - 광고성끼리 가깝게, 비광고성끼리 가깝게 학습
   - 광고성과 비광고성끼리는 멀게 학습
   - 리뷰를 384차원 벡터로 변환
7. **키워드 가중치 적용**
   - 광고 관련 단어에 +가중치, 비광고 관련 단어에 -가중치 적용
9. **통계 분석**
    - `p-value < 0.001`, `Cohen's d`를 통해 그룹간 차이 검증

📊 **결과 요약**
- 광고 리뷰는 문체·표현의 유사도가 높음
- 비광고 리뷰는 표현의 다양성이 큼
- 파인튜닝과 가중치 적용을 함으로써 통계적으로 유의미한 차이가 존재함 -> **정확한 판별 가능**
  

---

 ### 📈 프로젝트 성과
  - AI 탐지 정확도 **기존 대비 45% 향상**
  - Gemini 요약 결과를 통해 사용자 이해도 상승
  - 웹 페이지 내에서 분석·시각화·요약을 **한페이지 통합 구현**

---
