import React from "react";
import { Outlet, Link, useNavigate } from "react-router-dom";
import "../App.css"

function UserLayout() {
  const navigate = useNavigate()
  const isLoggedIn = !!localStorage.getItem("token");

  const handleLogout = () => {
    localStorage.removeItem("token");
    alert("로그아웃되었습니다.");
    navigate("/");
  };

  return (
    <div className="app">
      <header className="header-top">
        <div
          className="logo"
          onClick={() => navigate("/")}
          style={{ cursor: "pointer" }}
        >
          re:view
        </div>        
        <h2>회원 전용 페이지</h2>
        <nav className="nav-menu">
          {isLoggedIn ? (
            <a href="/" onClick={handleLogout}>
              <img src="/login.png" alt="Logout" />
              로그아웃
            </a>
            ) : (
              <Link to="/login">
                <img src="/login.png" alt="Login" />
                로그인
              </Link>
            )}
          <Link to="/user/inquiry">
            <img src="/contact.png" alt="Inquiries" />
            문의하기
          </Link>
        </nav>
      </header>
      <main>
        <Outlet /> {/* 회원 하위 라우트 출력 */}
      </main>
    </div>
  );
}

export default UserLayout;
