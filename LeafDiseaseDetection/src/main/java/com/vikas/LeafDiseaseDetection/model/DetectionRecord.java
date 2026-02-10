package com.vikas.LeafDiseaseDetection.model;

import jakarta.persistence.*;

@Entity
@Table(name = "detection_record")
public class DetectionRecord {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    private String disease;
    private double confidence;

    public DetectionRecord() {
    }

    public DetectionRecord(Long id, String disease, double confidence) {
        this.id = id;
        this.disease = disease;
        this.confidence = confidence;
    }

    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public String getDisease() {
        return disease;
    }

    public void setDisease(String disease) {
        this.disease = disease;
    }

    public double getConfidence() {
        return confidence;
    }

    public void setConfidence(double confidence) {
        this.confidence = confidence;
    }
}
