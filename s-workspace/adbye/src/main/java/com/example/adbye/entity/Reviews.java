package com.example.adbye.entity;

import jakarta.persistence.*;
import lombok.*;

@Entity
@Table(name = "reviews")
@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class Reviews {

    // 🔹 실제 기본키 (DB AUTO_INCREMENT)
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Column(name = "review_id")
    private Long reviewId;

    // 🔸 기존 id (카테고리 내 고유 id)
    @Column(name = "id")
    private Long id;

    @Column(length = 512)
    private String category;

    @Lob
    @Column(columnDefinition = "LONGTEXT")
    private String content;

    @Column(name = "product_name", length = 2048)
    private String productName;

    @Lob
    @Column(name = "cleaned_review", columnDefinition = "LONGTEXT")
    private String cleanedReview;

    private Integer label;
}
