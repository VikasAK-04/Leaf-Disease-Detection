package com.vikas.LeafDiseaseDetection.controller;

import com.vikas.LeafDiseaseDetection.dto.PredictionResponse;
import com.vikas.LeafDiseaseDetection.service.PredictionService;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

@RestController
@RequestMapping("/api/v1")
@CrossOrigin(origins = "*")
public class PredictionController {

    private final PredictionService predictionService;

    public PredictionController(PredictionService predictionService) {
        this.predictionService = predictionService;
    }

    @GetMapping("/test")
    public ResponseEntity<String> test() {
        return ResponseEntity.ok("✅ Backend is running!");
    }

    @PostMapping(value = "/predict", consumes = "multipart/form-data")
    public ResponseEntity<PredictionResponse> predict(
            @RequestParam(value = "file", required = false) MultipartFile file) {

        // Validate file
        if (file == null || file.isEmpty()) {
            PredictionResponse error = new PredictionResponse("No file uploaded", 0.0);
            error.setSuccess(false);
            error.setMessage("Please upload a valid image file.");
            return ResponseEntity.badRequest().body(error);
        }

        // Call service to get prediction
        PredictionResponse response = predictionService.predict(file);

        // Optional: If confidence is low, backend already sets predicted_class to "Unknown or Unrelated Image"
        // The message field will also explain why

        return ResponseEntity.ok(response);
    }
}
