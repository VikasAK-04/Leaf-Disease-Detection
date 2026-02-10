package com.vikas.LeafDiseaseDetection.repository;

import com.vikas.LeafDiseaseDetection.model.DetectionRecord;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

@Repository
public interface DetectionRecordRepository extends JpaRepository<DetectionRecord, Long> {
}