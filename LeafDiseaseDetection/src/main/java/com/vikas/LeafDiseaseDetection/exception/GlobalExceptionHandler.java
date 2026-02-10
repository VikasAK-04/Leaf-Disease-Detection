package com.vikas.LeafDiseaseDetection.exception;

import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.ControllerAdvice;
import org.springframework.web.bind.annotation.ExceptionHandler;

import java.util.Map;

@ControllerAdvice
public class GlobalExceptionHandler {

    @ExceptionHandler(Exception.class)
    public ResponseEntity<Map<String, String>> handleAllExceptions(Exception ex) {
        ex.printStackTrace();

        Map<String, String> error = Map.of(
                "error", ex.getMessage() != null ? ex.getMessage() : "Internal Server Error"
        );

        return ResponseEntity.status(500).body(error);
    }
}