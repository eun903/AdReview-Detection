package com.example.adbye.controller;

import com.example.adbye.entity.Inquiry;
import com.example.adbye.service.InquiryService;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/api/inquiries")
@CrossOrigin(origins = "*")
public class InquiryController {

    private final InquiryService inquiryService;

    public InquiryController(InquiryService inquiryService) {
        this.inquiryService = inquiryService;
    }

    // 회원 - 문의 작성 (POST /api/inquiries)
    @PostMapping
    public ResponseEntity<Inquiry> createInquiry(@RequestBody Inquiry inquiry) {
        return ResponseEntity.ok(inquiryService.saveInquiry(inquiry));
    }

    // 현서 코드에는 관리자 답변 API 존재
    // 회원 - 나의 문의 조회 (GET /api/inquiries/user/{username})
    @GetMapping("/user/{username}")
    public ResponseEntity<List<Inquiry>> getUserInquiries(@PathVariable String username) {
        return ResponseEntity.ok(inquiryService.findByUsername(username));
    }
    
    // 회원 - 나의 문의 삭제 (DELETE /api/inquiries/{id})
    @DeleteMapping("/{id}")
    public ResponseEntity<String> deleteInquiry(@PathVariable Long id, @RequestParam String username) {
        boolean deleted = inquiryService.deleteInquiry(id, username);
        if (deleted) {
            return ResponseEntity.ok("문의가 삭제되었습니다.");
        } else {
            return ResponseEntity.status(403).body("삭제 권한이 없습니다.");
        }
    }
}