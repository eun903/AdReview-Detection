package com.example.adbye.dto;

import com.fasterxml.jackson.annotation.JsonProperty;
import java.util.List;
import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

@Getter
@Setter
@AllArgsConstructor
@NoArgsConstructor
public class AnalyzeResponse {

    @JsonProperty("입력_리뷰")
    private String inputReview;
    
    @JsonProperty("요약문")
    private String summary; //추가

    @JsonProperty("유사도_점수")
    private double similarityScore;

    @JsonProperty("가장_유사한_광고_리뷰")
    private String mostSimilarAdReview;

    @JsonProperty("광고_키워드")
    private List<String> adKeywords;

    @JsonProperty("비광고_키워드")
    private List<String> nonAdKeywords;

    @JsonProperty("판단")
    private String decision;
    
    @JsonProperty("label")
    private int label;
}