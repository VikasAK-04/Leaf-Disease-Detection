package com.vikas.LeafDiseaseDetection.service;

import com.vikas.LeafDiseaseDetection.dto.PredictionResponse;
import com.vikas.LeafDiseaseDetection.model.DetectionRecord;
import com.vikas.LeafDiseaseDetection.repository.DetectionRecordRepository;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.FileSystemResource;
import org.springframework.http.*;
import org.springframework.stereotype.Service;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.MultiValueMap;
import org.springframework.web.client.HttpClientErrorException;
import org.springframework.web.client.HttpServerErrorException;
import org.springframework.web.client.ResourceAccessException;
import org.springframework.web.client.RestTemplate;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

@Service
public class PredictionService {

    private final RestTemplate restTemplate;
    private final DetectionRecordRepository repository;

    @Value("${ml.service.url:http://localhost:5000}")
    private String mlServiceUrl;

    // Confidence threshold for detecting unknown images
    private static final double CONFIDENCE_THRESHOLD = 0.70; // 70%

    public PredictionService(RestTemplate restTemplate, DetectionRecordRepository repository) {
        this.restTemplate = restTemplate;
        this.repository = repository;
    }

    public PredictionResponse predict(MultipartFile file) {
        Path temp = null;
        try {
            // Save file temporarily
            String originalFilename = file.getOriginalFilename();
            String suffix = originalFilename != null ? originalFilename : "upload.jpg";
            temp = Files.createTempFile("leaf-upload-", "-" + suffix);
            file.transferTo(temp.toFile());

            // Prepare multipart request
            MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
            body.add("file", new FileSystemResource(temp.toFile()));

            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.MULTIPART_FORM_DATA);
            HttpEntity<MultiValueMap<String, Object>> requestEntity = new HttpEntity<>(body, headers);

            // Send request to Flask ML service
            System.out.println("📤 Sending request to: " + mlServiceUrl + "/predict");

            ResponseEntity<PredictionResponse> response = restTemplate.postForEntity(
                    mlServiceUrl + "/predict",
                    requestEntity,
                    PredictionResponse.class
            );

            PredictionResponse prediction = response.getBody();

            if (prediction == null) {
                throw new RuntimeException("Empty response from ML service");
            }

            // Check confidence threshold
            if (prediction.getConfidence() / 100.0 < CONFIDENCE_THRESHOLD) {
                prediction.setPredictedClass("Unknown or Unrelated Image");
                prediction.setMessage("The uploaded image doesn't match any known leaf disease.");
                System.out.println("⚠️ Low confidence → Unknown image.");
            }

            System.out.println("✅ Received: " + prediction);

            // Save to database
            DetectionRecord record = new DetectionRecord();
            record.setDisease(prediction.getPredictedClass());
            record.setConfidence(prediction.getConfidence());
            repository.save(record);

            System.out.println("💾 Saved to database with ID: " + record.getId());

            return prediction;

        } catch (HttpClientErrorException | HttpServerErrorException ex) {
            System.err.println("❌ ML Service Error: " + ex.getStatusCode());
            System.err.println("Response: " + ex.getResponseBodyAsString());
jh
            PredictionResponse errorResponse = new PredictionResponse();
            errorResponse.setSuccess(false);
            errorResponse.setPredictedClass("ML Service Error");
            errorResponse.setConfidence(0.0);
            errorResponse.setMessage("ML service error: " + ex.getMessage());
            return errorResponse;

        } catch (ResourceAccessException ex) {
            System.err.println("❌ Cannot connect to ML service at " + mlServiceUrl);
            System.err.println("Error: " + ex.getMessage());

            PredictionResponse errorResponse = new PredictionResponse();
            errorResponse.setSuccess(false);
            errorResponse.setPredictedClass("ML Service Unavailable");
            errorResponse.setConfidence(0.0);
            errorResponse.setMessage("Cannot connect to ML service. Is Flask running?");
            return errorResponse;

        } catch (IOException ex) {
            System.err.println("❌ File error: " + ex.getMessage());
            throw new RuntimeException("Failed to process file: " + ex.getMessage(), ex);

        } finally {
            // Cleanup temp file
            if (temp != null) {
                try {
                    Files.deleteIfExists(temp);
                } catch (IOException ignored) {}
            }
        }
    }
}
