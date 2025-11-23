/* 회원 삭제시 DB에서 관련 히스토리 삭제해야지 가능함. */

import React, { useState, useEffect, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';

const Admin = () => {
  const [users, setUsers] = useState([]);
  const [inquiries, setInquiries] = useState([]);
  const [answerText, setAnswerText] = useState('');
  const [selectedInquiryId, setSelectedInquiryId] = useState(null);

  const navigate = useNavigate();
  const token = localStorage.getItem('token');

  const api = React.useMemo(() => axios.create({
    baseURL: 'http://localhost:8080/admin',
    headers: { 'Authorization': `Bearer ${token}` }
  }), [token]);

  const fetchUsers = useCallback(async () => {
    try {
      const res = await api.get('/users');
      setUsers(res.data);
    } catch (err) {
      console.error('사용자 목록 불러오기 실패:', err);
    }
  }, [api]);

  const fetchInquiries = useCallback(async () => {
    try {
      const res = await api.get('/inquiries');
      setInquiries(res.data);
    } catch (err) {
      console.error('문의 목록 불러오기 실패:', err);
    }
  }, [api]);

  useEffect(() => {
    const role = localStorage.getItem('role');
    if (role !== 'ROLE_ADMIN') {
      alert('관리자 권한이 필요합니다.');
      navigate('/');
    } else {
      fetchUsers();
      fetchInquiries();
    }
  }, [fetchUsers, fetchInquiries, navigate]);

  const deleteUser = async (id) => {
    try {
      await api.delete(`/users/${id}`);
      fetchUsers();
    } catch (err) {
      console.error('삭제 실패:', err);
    }
  };

  const submitAnswer = async () => {
    try {
      await api.put(`/inquiries/${selectedInquiryId}/answer`, { answer: answerText });
      setAnswerText('');
      setSelectedInquiryId(null);
      fetchInquiries();
    } catch (err) {
      console.error('답변 실패:', err);
    }
  };

  return (
    <div className="admin-dashboard-content" >
      {/* 사용자 목록 */}
      <section id="users-section">
        <h2>Users</h2>
        <table className="admin-table">
          <thead>
            <tr>
              <th>Username</th>
              <th>Email</th>
              <th>Role</th>
              <th>Action</th>
            </tr>
          </thead>
          <tbody>
            {users.map(u => (
              <tr key={u.id}>
              <td>{u.username}</td>
              <td>{u.email}</td>
              <td>{u.role}</td>
              <td>
                <button className="btn-delete" onClick={() => deleteUser(u.id)}>Delete</button>
              </td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>

      {/* 문의 목록 */}
      <section id="inquiries-section" style={{ marginTop: '40px' }}>
        <h2>Inquiries</h2>
        <table className="admin-table">
          <thead>
            <tr>
              <th>Username</th>
              <th>Title</th>
              <th>Content</th>
              <th>Answer</th>
              <th>Action</th>
            </tr>
          </thead>
          <tbody>
            {inquiries.map(q => (
              <tr key={q.id}>
                <td>{q.username}</td>
                <td>{q.title}</td>
                <td>{q.content}</td>
                <td>{q.answer || '대기 중'}</td>
                <td>
                  {selectedInquiryId === q.id ? (
                    <div>
                      <textarea
                        value={answerText}
                        onChange={(e) => setAnswerText(e.target.value)}
                        rows="3"
                        cols="30"
                      />
                      <button className="btn-reply" onClick={submitAnswer}>등록</button>
                    </div>
                  ) : (
                   !q.answer && (
                      <button className="btn-reply" onClick={() => setSelectedInquiryId(q.id)}>
                        Reply
                      </button>
                    )
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>
    </div>
  );
};

export default Admin;
