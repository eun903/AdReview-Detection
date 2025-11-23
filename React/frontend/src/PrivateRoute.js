import { Navigate } from "react-router-dom";

function PrivateRoute({ children, role, allowedRoles }) {
  if (!role) {
    return <Navigate to="/login" replace />; // 비로그인 → 로그인
  }
  if (!allowedRoles.includes(role)) {
    return <Navigate to="/review" replace />; // 권한 없는 경우 → 메인으로
  }
  return children;
}

export default PrivateRoute;
