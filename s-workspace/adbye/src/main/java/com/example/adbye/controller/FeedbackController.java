package com.example.adbye.controller;

import com.example.adbye.dto.FeedbackRequest;
import com.example.adbye.entity.FeedbackEntity;
import com.example.adbye.repository.FeedbackRepository;

import jakarta.transaction.Transactional;

import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

@RestController
@CrossOrigin(origins = "http://localhost:3000")
@RequestMapping("/review")
public class FeedbackController {

    private final FeedbackRepository feedbackRepository;

    public FeedbackController(FeedbackRepository feedbackRepository) {
        this.feedbackRepository = feedbackRepository;
    }

    @PostMapping("/feedback")
    @Transactional
    public ResponseEntity<String> saveFeedback(@RequestBody FeedbackRequest feedback) {

        FeedbackEntity entity = new FeedbackEntity(
            (feedback.getReview() != null) ? feedback.getReview() : "",
            (feedback.getScore() != null) ? feedback.getScore() : 0.0,
            feedback.getSimilarAdReview(),
            (feedback.getDecision() != null) ? feedback.getDecision() : "미정",
            (feedback.getFeedback() != null) ? feedback.getFeedback() : "없음"
        );

        FeedbackEntity saved = feedbackRepository.save(entity); 

        return ResponseEntity.ok("저장 완료");
    }
}
//아예 파일을 새로 추가함