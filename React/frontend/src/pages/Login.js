import React, { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import axios from 'axios';
import './Login.css';

function Login() {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const response = await axios.post('http://localhost:8080/api/auth/login', { username, password });

      console.log("로그인 응답:", JSON.stringify(response.data, null, 2)); // ✅ 구조 확인

      // 토큰과 role 저장
      localStorage.setItem('token', response.data.token);
      localStorage.setItem('role', response.data.role);
      localStorage.setItem('username', username); 


      alert('로그인에 성공했습니다.');

      // role에 따라 다른 페이지로 이동
      if (response.data.role === 'ROLE_ADMIN') {
        navigate('/admin');   // 관리자 전용 페이지
      } else {
        navigate('/review');  // 일반 회원 리뷰 분석 페이지
      }
    } catch (error) {
      console.error('Login failed:', error.response?.data || error.message);
      alert('로그인에 실패했습니다. 아이디와 비밀번호를 확인해주세요.');
    }
  };

  return (
    <div className="login-container">
      <div className="login-box">
        <h2 className="login-title">로그인</h2>
        <form onSubmit={handleSubmit} className="login-form">
          <div className="form-group">
            <input
              type="text"
              placeholder="아이디"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              required
              className="form-input"
            />
          </div>
          <div className="form-group">
            <input
              type="password"
              placeholder="비밀번호"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
              className="form-input"
            />
          </div>
          <button type="submit" className="login-button">로그인</button>
        </form>
        <div className="extra-links">
          <Link to="/register">회원가입</Link>
          <Link to="/findaccount">아이디/비밀번호 찾기</Link>
        </div>
      </div>
    </div>
  );
}

export default Login;
