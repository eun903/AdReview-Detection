import React, { useState } from "react";
import { Link } from "react-router-dom";
import "./FindAccount.css";

function FindAccount() {
  const [mode, setMode] = useState("id");
  const [email, setEmail] = useState("");
  const [username, setUsername] = useState("");
  const [message, setMessage] = useState("");

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const body = mode === "id" ? { email } : { email, username };
      const response = await fetch(`http://localhost:5000/find-${mode}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const data = await response.json();
      setMessage(data.message);
    } catch (err) {
      console.error(err);
      setMessage("서버 오류가 발생했습니다.");
    }
  };

  return (
    <div className="find-container">

      <div className="find-box">
        <h2>{mode === "id" ? "아이디 찾기" : "비밀번호 찾기"}</h2>

        <div className="tab-buttons">
          <button
            className={mode === "id" ? "active" : ""}
            onClick={() => setMode("id")}
          >
            아이디 찾기
          </button>
          <button
            className={mode === "password" ? "active" : ""}
            onClick={() => setMode("password")}
          >
            비밀번호 찾기
          </button>
        </div>

        {mode === "id" ? (
          <form onSubmit={handleSubmit}>
            <input
              type="email"
              placeholder="이메일 입력"
              required
              value={email}
              onChange={(e) => setEmail(e.target.value)}
            />
            <button type="submit">아이디 찾기</button>
          </form>
        ) : (
          <form onSubmit={handleSubmit}>
            <input
              type="text"
              placeholder="아이디 입력"
              required
              value={username}
              onChange={(e) => setUsername(e.target.value)}
            />
            <input
              type="email"
              placeholder="이메일 입력"
              required
              value={email}
              onChange={(e) => setEmail(e.target.value)}
            />
            <button type="submit">비밀번호 찾기</button>
          </form>
        )}

        {message && <p className="status-message">{message}</p>}

        <p className="back-link">
          <Link to="/login">로그인 화면으로 돌아가기</Link>
        </p>
      </div>
    </div>
  );
}

export default FindAccount;