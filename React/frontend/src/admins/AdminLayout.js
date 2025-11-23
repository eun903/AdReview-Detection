import React, { useState } from "react";
import { Outlet, Link, useNavigate } from "react-router-dom";
import "../App.css";

function AdminLayout() {
  const navigate = useNavigate();
  const isLoggedIn = !!localStorage.getItem("token");
  const [activeSection, setActiveSection] = useState(null);

  const handleLogout = () => {
    localStorage.removeItem("token");
    alert("로그아웃되었습니다.");
    navigate("/");
  };

  return (
    <div className="admin-app">
      <header className="header-top">
        <div
          className="logo"
          onClick={() => navigate("/")}
          style={{ cursor: "pointer" }}
        >
          re:view
        </div>
        <h2>관리자 페이지</h2>
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
        </nav>
      </header>

      <div className="admin-body">
        <aside className="admin-sidebar">
          <ul>
            <li>
              <a
                href="#users-section"
                onClick={() => setActiveSection("#users-section")}
                className={activeSection === "#users-section" ? "active" : ""}
              >
                Users
              </a>
            </li>
            <li>
              <a
                href="#inquiries-section"
                onClick={() => setActiveSection("#inquiries-section")}
                className={activeSection === "#inquiries-section" ? "active" : ""}
              >
                Inquiries
              </a>
            </li>
          </ul>
        </aside>

        <main className="admin-content">
          <Outlet />
        </main>
      </div>
    </div>
  );
}

export default AdminLayout;