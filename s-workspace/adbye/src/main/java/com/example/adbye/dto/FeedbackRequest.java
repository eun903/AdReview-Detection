package com.example.adbye.dto;
public class FeedbackRequest {

    private String review;
    private Double score;
    private String similarAdReview;
    private String decision;
    private String feedback;

    public FeedbackRequest() {}

    // Getter / Setter
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
