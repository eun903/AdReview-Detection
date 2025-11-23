package com.example.adbye.entity;

import jakarta.persistence.*;

@Entity
@Table(name = "feedback")
public class FeedbackEntity {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(nullable = false)
    private String review;

    @Column(nullable = false)
    private Double score = 0.0;

    @Column(name = "similar_ad_review")
    private String similarAdReview;

    @Column(nullable = false)
    private String decision;

    @Column(nullable = false)
    private String feedback;

    public FeedbackEntity() {}

    public FeedbackEntity(String review, Double score, String similarAdReview,
                          String decision, String feedback) {
        this.review = (review != null) ? review : "";
        this.score = (score != null) ? score : 0.0;
        this.similarAdReview = (similarAdReview != null) ? similarAdReview : "";
        this.decision = (decision != null) ? decision : "미정";
        this.feedback = (feedback != null) ? feedback : "없음";
    }

    // Getter / Setter
    public Long getId() { return id; }
    public String getReview() { return review; }
    public void setReview(String review) { this.review = review; }
    public Double getScore() { return score; }
    public void setScore(Double score) { this.score = score; }
    public String getSimilarAdReview() { return similarAdReview; }
    public void setSimilarAdReview(String similarAdReview) { this.similarAdReview = similarAdReview; }
    public String getDecision() { return decision; }
    public void setDecision(String decision) { this.decision = decision; }
    public String getFeedback() { return feedback; }
    public void setFeedback(String feedback) { this.feedback = feedback; }
}
