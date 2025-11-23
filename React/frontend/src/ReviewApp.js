import React, { useState, useEffect, useRef } from "react";
import { useNavigate } from "react-router-dom";
import { PieChart, Pie, Cell } from "recharts";
import "./App.css";
import axios from 'axios';

// ✅ 유사도 점수 차트 (변경 없음)
const SimilarityChart = ({ score }) => {
  const data = [
    { name: "유사도", value: score ?? 0 },
    { name: "나머지", value: 100 - (score ?? 0) },
  ];
  const COLORS = ["#e6f911ff", "#E0E0E0"];

  return (
    <div className="flex flex-col items-center justify-center my-4">
      <PieChart width={120} height={120}>
        <Pie
          data={data}
          cx="50%"
          cy="50%"
          innerRadius={40}
          outerRadius={60}
          dataKey="value"
          startAngle={90}
          endAngle={-270}
        >
          {data.map((entry, index) => (
            <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
          ))}
        </Pie>
      </PieChart>
      <div className="absolute text-center">
        <span className="text-xl font-bold">{(score ?? 0).toFixed(1)}%</span>
      </div>
    </div>
  );
};

// ✅ 긴 텍스트 접기/펼치기 (변경 없음)
const TruncatedTextAi = ({ text, maxLength }) => {
  const [isExpanded, setIsExpanded] = useState(false);
  const displayText = text ?? "";

  if (displayText.length <= maxLength) {
    return <span>{displayText}</span>;
  }

  return (
    <div>
      <span>{isExpanded ? displayText : `${displayText.substring(0, maxLength)}...`}</span>
      <button
        onClick={(e) => {
          e.stopPropagation();
          setIsExpanded(!isExpanded);
        }}
        className="toggle-text-button"
      >
        {isExpanded ? "간략히 보기" : "더 보기"}
      </button>
    </div>
  );
};

const TruncatedText = ({ text, maxLength }) => {
  const [isExpanded, setIsExpanded] = useState(false);
  const displayText = text ?? "";

  const truncated = displayText.length > maxLength ? displayText.substring(0, maxLength) + '...' : displayText;

  // •, ✔, 마침표 기준 줄바꿈
  const formattedText = (isExpanded ? displayText : truncated)
    .replace(/([•✔])/g, "\n$1")
    .replace(/(\. )/g, ".\n");

  return (
    <div>
      {formattedText.split("\n").map((line, idx) => (
        <p key={idx} style={{ margin: '4px 0' }}>{line}</p>
      ))}
      {displayText.length > maxLength && (
        <button
          onClick={(e) => { e.stopPropagation(); setIsExpanded(!isExpanded); }}
          className="toggle-text-button"
          style={{ color: "#8b8b8bff", background: "none", border: "none", cursor: "pointer", padding: 0 }}
        >
          {isExpanded ? "간략히 보기" : "더 보기"}
        </button>
      )}
    </div>
  );
};


// ✅ 메인 컴포넌트
function ReviewApp() {
  const [review, setReview] = useState("");
  const [placeholder, setPlaceholder] = useState("이미지 또는 텍스트 업로드");
  const [showSimilar, setShowSimilar] = useState(false);
  const [reviewsData, setReviewsData] = useState([]);
  const [loading, setLoading] = useState(false);
  const [selectedCategory, setSelectedCategory] = useState("");
  const [expandedItems, setExpandedItems] = useState({});
  const API_BASE = "http://localhost:8080";

  const navigate = useNavigate();
  const isLoggedIn = !!localStorage.getItem("token");
  const textareaRef = useRef(null);
  const [loadingText, setLoadingText] = useState("분석 중");
  const [feedbackCounts, setFeedbackCounts] = useState({ like: 0, dislike: 0 });

  // 🗑️ 제거된 상태: reviewText, result (경고 해결)
  // const [reviewText, setReviewText] = useState("");
  // const [result, setResult] = useState(null);

  // ✅ 추가된 상태: 붙여넣기된 이미지 URL
  const [pastedImage, setPastedImage] = useState(null);

  const categories = [
    "패션잡화", "식품건강", "뷰티", "생활주방",
    "유아동", "스포츠레저", "가전디지털", "문구오피스",
  ];

  useEffect(() => {
    if (loading) {
      let count = 0;
      const interval = setInterval(() => {
        count = (count + 1) % 3;
        setLoadingText("분석 중" + ".".repeat(count + 1));
      }, 500);

      return () => clearInterval(interval);
    } else {
      setLoadingText("분석 중");
    }
  }, [loading]);

  useEffect(() => {
    const storedCounts = localStorage.getItem("feedbackCounts");
    if (storedCounts) {
      try {
        const parsed = JSON.parse(storedCounts);
        setFeedbackCounts({
          like: Number(parsed.like) || 0,
          dislike: Number(parsed.dislike) || 0
        });
      } catch {
        setFeedbackCounts({ like: 0, dislike: 0 });
      }
    }

    const storedFeedback = localStorage.getItem("feedbackGiven");
    if (storedFeedback) {
      setFeedbackGiven(JSON.parse(storedFeedback));
    }
  }, []);

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
      textareaRef.current.style.height = `${textareaRef.current.scrollHeight}px`;
    }
  }, [review]); // review 상태가 변경될 때 textarea 높이 조절

  const handleLogout = () => {
    localStorage.removeItem("token");
    alert("로그아웃되었습니다.");
    navigate("/");
  };

  const handleCategoryClick = (name) => {
    setPlaceholder(name);
    setSelectedCategory(name);
  };

  // ✅ handleShowReview 함수에 textParam 추가 (이전 에러 해결)
  const handleShowReview = async (textParam) => {
    // textParam이 있으면 그것을 사용하고, 없으면 review 상태의 값을 사용
    const text = textParam || review;
    if (!text) {
      alert("리뷰 내용을 입력해주세요.");
      return;
    }

    setLoading(true);

    try {
      const token = localStorage.getItem("token");
      const headers = { "Content-Type": "application/json" };
      if (token) {
        headers['Authorization'] = `Bearer ${token}`;
      }

      console.log("전송하려는 카테고리:", selectedCategory);

      const response = await fetch(`${API_BASE}/review/check`, {
        method: "POST",
        headers: headers,
        body: JSON.stringify({ userReview: text, category: selectedCategory }),
      });

      if (!response.ok) {
        const msg = await response.text();
        throw new Error(`서버 응답 오류 (${response.status}): ${msg}`);
      }

      const data = await response.json();

      const newReview = {
        ...data,
        category: selectedCategory,
        timestamp: Date.now(),
      };

      setReviewsData(prevData => [newReview, ...prevData]);

      setShowSimilar(true);
      setExpandedItems({});
    } catch (error) {
      console.error("에러 발생:", error);
      alert("분석 중 오류가 발생했습니다.");
    } finally {
      setLoading(false);
    }
  };

  const [feedbackGiven, setFeedbackGiven] = useState({});

  const handleFeedback = async (e, isPositive, reviewData) => {
    e.stopPropagation();
    if (!reviewData) return;
    const key = reviewData.timestamp;
    if (feedbackGiven[key]) {
      alert("이미 피드백을 제출했습니다.");
      return;
    }
    try {
      const feedbackPayload = {
        review: reviewData["입력_리뷰"],
        score: reviewData["유사도_점수"],
        similarAdReview: reviewData["가장_유사한_광고_리뷰"],
        decision: reviewData["판단"],
        feedback: isPositive ? "추천" : "비추천",
      };

      const feedbackResponse = await fetch(`${API_BASE}/review/feedback`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(feedbackPayload),
      });

      if (!feedbackResponse.ok) {
        const errorText = await feedbackResponse.text();
        throw new Error(`서버 응답 오류 (${feedbackResponse.status}): ${errorText}`);
      }

      alert(isPositive ? "추천이 반영되었습니다." : "비추천이 반영되었습니다.");
      setFeedbackGiven(prev => {
        const newState = { ...prev, [reviewData.timestamp]: isPositive ? "like" : "dislike" };
        localStorage.setItem("feedbackGiven", JSON.stringify(newState));
        return newState;
      });
      setFeedbackCounts(prev => {
        const updated = {
          like: prev.like + (isPositive ? 1 : 0),
          dislike: prev.dislike + (!isPositive ? 1 : 0)
        };
        localStorage.setItem("feedbackCounts", JSON.stringify(updated));
        return updated;
      });

    } catch (error) {
      console.error("피드백 전송 오류:", error);
      alert("피드백 전송 중 오류가 발생했습니다.");

    }
  };

  const toggleReviewExpand = (index) => {
    setExpandedItems(prev => ({
      ...prev,
      [index]: !prev[index],
    }));
  };

  // ✅ 이미지 삭제 함수 추가
  const handleDeleteImage = () => {
    setPastedImage(null); // pastedImage 상태를 null로 설정하여 이미지 제거
    if (textareaRef.current) {
        textareaRef.current.focus(); // 텍스트 입력 영역으로 포커스 이동
    }
  };

  // ✅ handlePaste 함수 수정
  const handlePaste = async (event) => {
    const items = event.clipboardData.items;
    for (const item of items) {
      if (item.type.startsWith("image/")) {
        event.preventDefault();
        const blob = item.getAsFile();

        if (blob) {
          const reader = new FileReader();
          reader.onload = async (e) => {
            // ✅ 이미지를 DOM에 직접 추가하는 대신 pastedImage 상태에 저장
            setPastedImage(e.target.result);

            setLoading(true);
            try {
              const formData = new FormData();
              formData.append("image", blob, "pasted-image.png");

              const token = localStorage.getItem("token");
              const headers = {}; // Content-Type은 axios가 FormData 처리 시 자동으로 설정
              if (token) headers["Authorization"] = `Bearer ${token}`;

              const response = await axios.post(
                "http://localhost:8080/review/ocr",
                formData,
                { headers }
              );

              const extracted = response?.data?.extractedText?.trim();
              if (!extracted) {
                alert("텍스트를 인식하지 못했습니다.");
                // 텍스트 인식 실패 시 이미지도 함께 삭제할지 결정
                // setPastedImage(null);
                return;
              }

              // ✅ 1) OCR로 추출된 텍스트를 review 상태에 저장
              setReview(extracted);

              // ✅ 2) OCR 끝나면 즉시 분석 실행 (extracted 텍스트로)
              handleShowReview(extracted);

            } catch (error) {
              console.error("OCR 에러:", error);
              alert("이미지에서 텍스트를 추출하는 데 실패했습니다.");
              setPastedImage(null); // 에러 발생 시 이미지도 삭제
            } finally {
              setLoading(false);
            }
          };
          reader.readAsDataURL(blob);
        }
      }
    }
  };

  return (
    <div className="app">
      <header className="header-top">
        <div className="logo" onClick={() => navigate("/")} style={{ cursor: "pointer" }}>
          re:view
        </div>
        <nav className="nav-menu">
          {isLoggedIn ? (
            <a href="/" onClick={handleLogout}>
              <img src="/login.png" alt="Logout" />
              로그아웃
            </a>
          ) : (
            <a href="/login">
              <img src="/login.png" alt="Login" />
              로그인
            </a>
          )}
          <a href="/user/inquiry">
            <img src="/contact.png" alt="Contact" />
            문의하기
          </a>
        </nav>
      </header>

      <div style={{ display: "flex", paddingTop: "60px", minHeight: "100vh" }}>
        <div className="sidebar">
          <ul>
            <li className="category-title">
              카테고리
              <ul style={{ marginTop: "0.9rem", marginLeft: "0.01rem" }}>
                {categories.map(cat => (
                  <li
                    key={cat}
                    onClick={() => handleCategoryClick(cat)}
                    style={{
                      padding: "4px 0",
                      cursor: "pointer",
                      color: selectedCategory === cat ? "#00ffccff" : "white",
                      fontWeight: selectedCategory === cat ? "bold" : "normal",
                    }}
                  >
                    {cat}
                  </li>
                ))}
              </ul>
            </li>
            <li className="history-button" onClick={() => navigate("/user/history")}>
              History
            </li>
          </ul>
        </div>

        <div className="content">
            <p className="main-title">AI 기반 광고성 리뷰 탐지 웹 서비스</p>

            <div className="input-area-wrapper" style={{ width: '80%', maxWidth: '700px' }}> {/* width 추가 */}

            <div
              ref={textareaRef}
                contentEditable
                className="review-input editable"
                onInput={(e) => setReview(e.currentTarget.innerText)}
                onPaste={handlePaste}
                data-placeholder={placeholder}
              >
              {/* ✅ 이미지 미리보기 컨테이너를 contentEditable DIV 내부로 이동 */}
                {pastedImage && (
                  <div className="image-preview-container">
                    <img
                      src={pastedImage}
                      alt="붙여넣은 이미지"
                    />
                    <button
                      onClick={handleDeleteImage}
                      className="delete-image-btn"
                    >
                      X
                    </button>
                  </div>
                )}
              </div>

              <button
                className={`input-button ${loading ? "loading" : ""}`}
                onClick={() => handleShowReview(review)}
                disabled={loading}
              >
              {loading ? loadingText : "분석"}
              </button>
            </div>
          </div>


        <div className={`similar-review ${showSimilar ? "show" : ""}`}>
          <h3>분석 결과</h3>
          {Array.isArray(reviewsData) && reviewsData.filter(d => d != null).length > 0 ? (
            reviewsData.filter(d => d != null).map((data, index) => {
              const inputReview = data?.["입력_리뷰"] ?? "";
              const similarityScore = data?.["유사도_점수"] ?? 0;
              const mostSimilar = data?.["가장_유사한_광고_리뷰"] ?? "";
              const adKeywords = Array.isArray(data?.["광고_키워드"]) ? data["광고_키워드"] : [];
              const notAdKeywords = Array.isArray(data?.["비광고_키워드"]) ? data["비광고_키워드"] : [];
              const judgement = data?.["판단"] ?? "";
              const category = data?.category ?? "";

              const isLatest = index === 0;
              const isOpen = isLatest || expandedItems[index];

              return (
                <div
                  key={index}
                  className="review-result-item"
                  onClick={() => !isLatest && toggleReviewExpand(index)}
                  style={{
                    cursor: isLatest ? "default" : "pointer",
                    borderBottom: "1px solid #ddd",
                    paddingBottom: "12px",
                    marginBottom: "12px",
                  }}
                >
                  {!isOpen ? (
                    <div className="result-item-group">
                      <p><strong>입력 리뷰 :</strong></p>
                      <div className="result-item-content">
                        {inputReview.length > 20 ? inputReview.slice(0, 20) + "..." : inputReview}
                      </div>
                    </div>
                  ) : (
                    <div>
                      <div className="result-item-group review-header">
                        <div className="review-label">
                          <strong>입력 리뷰 :</strong>
                          <span className="category-inline">{category || ""}</span> 
                        </div>
                        <div className="result-item-content">
                          <TruncatedText text={inputReview} maxLength={100} />
                        </div>
                          {/* ✅ 요약문 표시 추가 */}
                        {data["요약문"] && (
                          <div className="result-summary-group">
                            <p className="summary-title">AI 요약</p>
                            <div className="summary-box">
                              <TruncatedTextAi text={data["요약문"]} maxLength={80} />
                            </div>
                          </div>
                        )}
                      </div>

                      <div className="result-item-group">
                        <p><strong>유사도 점수 :</strong></p>
                        <div className="chart-center-wrapper">
                          <SimilarityChart score={similarityScore} />
                        </div>
                      </div>

                      <div className="result-item-group">
                        <p><strong>가장 유사한 광고 리뷰 :</strong></p>
                        <div className="result-item-content">
                          <TruncatedText text={mostSimilar} maxLength={120} />
                        </div>
                      </div>

                      <div className="result-item-group">
                        <p><strong>광고 키워드 :</strong></p>
                        <div className="keyword-container">
                          {adKeywords.map((keyword, i) => (
                            <span key={i} className="keyword-tag">{keyword}</span>
                          ))}
                        </div>
                      </div>

                      <div className="result-item-group">
                        <p><strong>비광고 키워드 :</strong></p>
                        <div className="keyword-container">
                          {notAdKeywords.map((keyword, i) => (
                            <span key={i} className="keyword-tag">{keyword}</span>
                          ))}
                        </div>
                      </div>

                      <div className="result-item-group judgement-line">
                        <strong>판단 :</strong>
                        <span className={`result-judgement ${judgement.includes("광고") ? "ad" : "not-ad"}`}>
                          {judgement}
                        </span>
                      </div>

                      <button
                        onClick={(e) => handleFeedback(e, true, data)}
                        className={`feedback-btn ${feedbackGiven[data.timestamp] === "like" ? "clicked" : ""}`}
                      >
                        <img src="/feedback.png" alt="추천" className="feedback-icon" />
                        추천 <span className="count">{feedbackCounts.like}</span>
                      </button>

                      <button
                        onClick={(e) => handleFeedback(e, false, data)}
                        className={`feedback-btn ${feedbackGiven[data.timestamp] === "dislike" ? "clicked" : ""}`}
                      >
                        <img src="/feedback.png" alt="비추천" className="feedback-icon dislike-icon" />
                        비추천 <span className="count">{feedbackCounts.dislike}</span>
                      </button>
                    </div>
                  )}
                </div>
              );
            })
          ) : (
            <p className="no-result-text">아직 분석 결과가 없습니다.</p>
          )}
        </div>
      </div>
    </div>
  );
}

export default ReviewApp;