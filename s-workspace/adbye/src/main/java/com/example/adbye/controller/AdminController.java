package com.example.adbye.controller;

import com.example.adbye.dto.AnswerRequest;
import com.example.adbye.entity.*;
import com.example.adbye.service.*;
import java.util.List;

import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/admin")
public class AdminController {
	
	private final UserService userService;
	private final InquiryService inquiryService;
	
    public AdminController(UserService userService, InquiryService inquiryService) {
        this.userService = userService;
        this.inquiryService = inquiryService;
    }

    // 모든 사용자 조회 (관리자 전용)
    @GetMapping("/users")
    public ResponseEntity<List<User>> getAllUsers() {
        List<User> users = userService.findAllUsers();
        return ResponseEntity.ok(users);
    }

    // 특정 사용자 삭제 (관리자 전용)
    @DeleteMapping("/users/{id}")
    public ResponseEntity<String> deleteUser(@PathVariable Long id) {
        userService.deleteUserById(id);
        return ResponseEntity.ok("User with ID " + id + " has been deleted.");
    }

    // 모든 문의 조회 (관리자 전용)
    @GetMapping("/inquiries")
    public ResponseEntity<List<Inquiry>> getAllInquiries() {
        List<Inquiry> inquiries = inquiryService.findAll();
        return ResponseEntity.ok(inquiries);
    }

    // 특정 문의에 답변 (관리자 전용)
    @PutMapping("/inquiries/{id}/answer")
    public ResponseEntity<Inquiry> answerInquiry(
            @PathVariable Long id,
            @RequestBody AnswerRequest request
    ) {
        Inquiry updatedInquiry = inquiryService.answerInquiry(id, request.getAnswer());
        return ResponseEntity.ok(updatedInquiry);
    }
    
    // 이거 사용 x
    @GetMapping("/dashboard")
    public ResponseEntity<String> getAdminDashboard() {
        // 이 API는 관리자 대시보드 데이터 제공
        return ResponseEntity.ok("Admin access: Dashboard data.");
    }
}
