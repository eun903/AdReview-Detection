import React from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";

import About from "./pages/About"; // 이거 어디서?

import Home from "./pages/Home";
import ReviewApp from "./ReviewApp";
import Register from "./pages/Register";
import Login from "./pages/Login";
import FindAccount from "./pages/FindAccount";
import ContactUser from "./pages/ContactUser";
//import ContactAdmin from "./pages/ContactAdmin";
import Dashboard from "./pages/Dashboard";
import Layout from "./layouts/Layout";
import Admin from "./admins/Admin";
import AdminLayout from "./admins/AdminLayout";
import UserLayout from "./users/UserLayout";
import HistoryPage from "./pages/HistoryPage";
import ResetPasswordPage from "./pages/ResetPasswordPage";

function App() {
  return (
    <Router>
      <Routes>
        <Route index element={<Home />} />
        <Route path="review" element={<ReviewApp />} />
        {/* 모든 페이지에 적용되는 공통 레이아웃 */}
        <Route path="/" element={<Layout />}>
          <Route path="login" element={<Login />} />
          <Route path="register" element={<Register />} />
          <Route path="findaccount" element={<FindAccount />} />
          <Route path="about" element={<About />} />
          <Route path="dashboard" element={<Dashboard />} />
          <Route path="reset-password" element={<ResetPasswordPage />} /> 
        </Route>

        {/* 회원 전용 레이아웃과 경로 */}
        <Route path="/user" element={<UserLayout />}>
          <Route path="inquiry" element={<ContactUser />} />
          <Route path="history" element={<HistoryPage />} />
        </Route>

        {/* 관리자 전용 레이아웃과 경로 */}
        <Route path="/admin" element={<AdminLayout />}>
          <Route index element={<Admin />}>
            {/* <Route path="contact" element={<ContactAdmin />} />
            <Route path="profile" element={<Admin />} />  */}
          </Route>
        </Route>
      </Routes>
    </Router>
  );
}

export default App;