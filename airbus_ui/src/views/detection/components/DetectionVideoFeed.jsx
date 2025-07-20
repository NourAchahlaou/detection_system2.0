
import React, { useState, useEffect, useRef } from "react";
import { VideoCard } from "./styledComponents";
import CameraPlaceholder from "../CameraPlaceholder";
import LiveDetectionView from "../LiveDetectionView";
import { detectionService } from "../detectionService";

const DetectionVideoFeed = ({
  isDetectionActive,
  onStartDetection,
  onStopDetection,
  cameraId,
  targetLabel,
  isModelLoaded
}) => {
  const [videoUrl, setVideoUrl] = useState("");
  const [showControls, setShowControls] = useState(false);
  const [detectionStats, setDetectionStats] = useState({
    objectDetected: false,
    detectionCount: 0,
    nonTargetCount: 0
  });
  const videoRef = useRef(null);

  // Update video URL when detection state changes
  useEffect(() => {
    if (isDetectionActive && targetLabel && cameraId) {
      const url = `/api/detection/detection/video_feed?camera_id=${cameraId}&target_label=${encodeURIComponent(targetLabel)}`;
      setVideoUrl(url);
    } else {
      setVideoUrl("");
    }
  }, [isDetectionActive, targetLabel, cameraId]);

  // Reset detection stats when target changes
  useEffect(() => {
    setDetectionStats({
      objectDetected: false,
      detectionCount: 0,
      nonTargetCount: 0
    });
  }, [targetLabel]);

  const handleStartDetection = async () => {
    if (!targetLabel || !cameraId) {
      alert("Please select a camera and enter a target label first.");
      return;
    }

    if (!isModelLoaded) {
      alert("Please wait for the model to load before starting detection.");
      return;
    }

    try {
      const feedUrl = await detectionService.startDetectionFeed(cameraId, targetLabel);
      setVideoUrl(feedUrl);
      onStartDetection();
    } catch (error) {
      console.error("Error starting detection:", error);
      alert("Failed to start detection. Please try again.");
    }
  };

  const handleStopDetection = async () => {
    try {
      await detectionService.stopDetectionFeed();
      setVideoUrl("");
      setDetectionStats({
        objectDetected: false,
        detectionCount: 0,
        nonTargetCount: 0
      });
      onStopDetection();
    } catch (error) {
      console.error("Error stopping detection:", error);
      alert("Failed to stop detection. Please try again.");
    }
  };

  // Placeholder for detection stats update (you can implement WebSocket or polling)
  useEffect(() => {
    if (isDetectionActive) {
      // You could implement WebSocket connection here to get real-time detection stats
      // For now, this is a placeholder
      const interval = setInterval(() => {
        // Simulate detection stats updates
        setDetectionStats(prev => ({
          ...prev,
          detectionCount: prev.detectionCount + Math.floor(Math.random() * 2),
          objectDetected: Math.random() > 0.5
        }));
      }, 2000);

      return () => clearInterval(interval);
    }
  }, [isDetectionActive]);

  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center" }}>
      <VideoCard
        cameraActive={isDetectionActive}
        onMouseEnter={() => setShowControls(true)}
        onMouseLeave={() => setShowControls(false)}
      >
        {isDetectionActive ? (
          <LiveDetectionView
            videoUrl={videoUrl}
            videoRef={videoRef}
            showControls={showControls}
            onStopDetection={handleStopDetection}
            detectionStats={detectionStats}
            targetLabel={targetLabel}
          />
        ) : (
          <CameraPlaceholder 
            onStartCamera={handleStartDetection}
            cameraId={cameraId}
            buttonText="Start Detection"
            icon="detection"
            disabled={!isModelLoaded || !targetLabel || !cameraId}
          />
        )}
      </VideoCard>
    </div>
  );
};

export default DetectionVideoFeed;