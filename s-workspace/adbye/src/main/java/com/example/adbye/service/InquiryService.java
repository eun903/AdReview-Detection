package com.example.adbye.service;

import com.example.adbye.entity.Inquiry;
import com.example.adbye.repository.InquiryRepository;
import org.springframework.stereotype.Service;

import java.time.LocalDateTime;
import java.util.*;

@Service
public class InquiryService {
    private final InquiryRepository inquiryRepository;

    public InquiryService(InquiryRepository inquiryRepository) {
        this.inquiryRepository = inquiryRepository;
    }

    public Inquiry saveInquiry(Inquiry inquiry) {
        return inquiryRepository.save(inquiry);
    }
    
    // 현서랑 코드 조금 다름

    public List<Inquiry> findAll() {
        return inquiryRepository.findAll();
    }

    public List<Inquiry> findByUsername(String username) {
        return inquiryRepository.findByUsername(username);
    }

    public Inquiry answerInquiry(Long id, String answer) {
        Inquiry inquiry = inquiryRepository.findById(id)
                .orElseThrow(() -> new RuntimeException("문의가 존재하지 않습니다."));
        inquiry.setAnswer(answer);
        inquiry.setAnsweredAt(LocalDateTime.now());
        return inquiryRepository.save(inquiry);
    }
    
    public boolean deleteInquiry(Long id, String username) {
        Optional<Inquiry> optionalInquiry = inquiryRepository.findById(id);
        if (optionalInquiry.isPresent()) {
            Inquiry inquiry = optionalInquiry.get();

            // 본인 확인
            if (inquiry.getUsername().equals(username)) {
                inquiryRepository.delete(inquiry);
                return true;
            } else {
                return false; // 다른 사람이면 삭제 불가
            }
        }
        return false;
    }

}
