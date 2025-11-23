import React, { useState, useEffect, useRef } from "react";
import axios from "axios";
import { useNavigate } from "react-router-dom";
import "./Contact.css";

function ContactUser() {
  const [title, setTitle] = useState("");
  const [content, setContent] = useState("");
  const [inquiries, setInquiries] = useState([]);
  const navigate = useNavigate();
  const hasRun = useRef(false); // ✅ useEffect 첫 실행 방지용

  // 로그인된 사용자 이름 가져오기 (JWT에서 가져온다고 가정)
  const username = localStorage.getItem("username");

  useEffect(() => {
    if (hasRun.current) return; // ✅ 두 번째 실행 방지
    hasRun.current = true;
    const token = localStorage.getItem("token");

    // ✅ 로그인되지 않은 경우
    if (!token || !username) {
      alert("로그인이 필요한 서비스입니다.");
      navigate("/login");
      return;
    }

    // ✅ 로그인된 경우 문의 목록 불러오기
    axios
      .get(`/api/inquiries/user/${username}`, {
        headers: {
          Authorization: `Bearer ${token}`,
        },
      })
      .then((res) => setInquiries(res.data))
      .catch((err) => console.error(err));
  }, [username, navigate]);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!title || !content) return;

    const inquiry = { username, title, content };
    const token = localStorage.getItem("token");

    axios
      .post("/api/inquiries", inquiry, {
        headers: {
          Authorization: `Bearer ${token}`,
        },
      })
      .then((res) => {
        setInquiries([...inquiries, res.data]);
        setTitle("");
        setContent("");
      })
      .catch((err) => console.error(err));
  };

  const handleDelete = (id) => {
    if (window.confirm("정말로 이 문의를 삭제하시겠습니까?")) {
      const token = localStorage.getItem("token");

      axios
        .delete(`/api/inquiries/${id}?username=${username}`, {
          headers: {
            Authorization: `Bearer ${token}`,
          },
        })
        .then(() => {
          alert("문의가 삭제되었습니다.");
          setInquiries(inquiries.filter((inq) => inq.id !== id));
        })
        .catch((err) => {
          console.error(err);
          alert("삭제 권한이 없거나 오류가 발생했습니다.");
        });
    }
  };

  return (
    <div className="contact-container">
      {/* 문의 작성 */}
      <div className="contact-box">
        <h2>문의하기</h2>
        <form onSubmit={handleSubmit}>
          <label>제목</label>
          <input
            type="text"
            value={title}
            onChange={(e) => setTitle(e.target.value)}
          />

          <label>내용</label>
          <textarea
            value={content}
            onChange={(e) => setContent(e.target.value)}
          ></textarea>

          <button type="submit">작성하기</button>
        </form>
      </div>

      {/* 나의 문의 내역 */}
      <div className="inquiry-list">
        <h3>나의 문의 내역</h3>
        <ul>
          {inquiries.map((item, index) => (
            <li key={item.id}>
              <strong>
                {index + 1}. {item.title}
              </strong>
              {item.username === username && (
                <button
                  onClick={() => handleDelete(item.id)}
                  className="delete-btn-inline"
                  style={{ marginLeft: "10px" }}
                >
                  삭제
                </button>
              )}
              <br />
              문의: {item.content} <br />
              답변: {item.answer ? item.answer : "아직 답변이 없습니다."}
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}

export default ContactUser;
