import React from "react";
import { Routes, Route, Navigate } from "react-router-dom";
import UserLayout from "./UserLayout";

// 회원 페이지 컴포넌트 (예시)
import ReviewPage from "./ReviewPage";
import InquiryPage from "./InquiryPage";

function UserRoutes() {
  const role = localStorage.getItem("role");

  if (role !== "ROLE_USER") {
    return <Navigate to="/" replace />; // 권한 없으면 메인으로
  }

  return (
    <Routes>
      <Route path="/user" element={<UserLayout />}>
        <Route path="review" element={<ReviewPage />} />
        <Route path="inquiries" element={<InquiryPage />} />
      </Route>
    </Routes>
  );
}

export default UserRoutes;
