package com.vikas.LeafDiseaseDetection.dto;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;
import java.util.Map;

@JsonIgnoreProperties(ignoreUnknown = true)
public class PredictionResponse {

    private boolean success;

    @JsonProperty("predicted_class")  // Flask sends "predicted_class"
    private String predictedClass;

    private double confidence;
    private String message;

    @JsonProperty("all_predictions")  // Flask sends "all_predictions"
    private Map<String, Double> allPredictions;

    // No-args constructor
    public PredictionResponse() {
    }

    // Constructor for manual error responses
    public PredictionResponse(String predictedClass, double confidence) {
        this.predictedClass = predictedClass;
        this.confidence = confidence;
        this.success = true;
    }

    // Getters and Setters
    public boolean isSuccess() {
        return success;
    }

    public void setSuccess(boolean success) {
        this.success = success;
    }

    public String getPredictedClass() {
        return predictedClass;
    }

    public void setPredictedClass(String predictedClass) {
        this.predictedClass = predictedClass;
    }

    public double getConfidence() {
        return confidence;
    }

    public void setConfidence(double confidence) {
        this.confidence = confidence;
    }

    public String getMessage() {
        return message;
    }

    public void setMessage(String message) {
        this.message = message;
    }

    public Map<String, Double> getAllPredictions() {
        return allPredictions;
    }

    public void setAllPredictions(Map<String, Double> allPredictions) {
        this.allPredictions = allPredictions;
    }

    @Override
    public String toString() {
        return "PredictionResponse{" +
                "success=" + success +
                ", predictedClass='" + predictedClass + '\'' +
                ", confidence=" + confidence +
                ", message='" + message + '\'' +
                '}';
    }
}
