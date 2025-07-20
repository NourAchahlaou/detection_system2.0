// pages/AppDetection.jsx
import React, { useState, useEffect } from "react";
import { Box, Grid, Stack, Alert, CircularProgress } from '@mui/material';

import DetectionControls from "./components/DetectionControls";
import DetectionVideoFeed from "./components/DetectionVideoFeed";
import { cameraService } from "../captureImage/CameraService";
import { detectionService } from "./detectionService";

export default function AppDetection() {
  const [targetLabel, setTargetLabel] = useState("");
  const [cameraId, setCameraId] = useState("");
  const [cameras, setCameras] = useState([]);
  const [isDetectionActive, setIsDetectionActive] = useState(false);
  const [selectedCameraId, setSelectedCameraId] = useState('');
  const [isDetecting, setIsDetecting] = useState(false);
  const [isModelLoaded, setIsModelLoaded] = useState(false);
  const [modelLoadError, setModelLoadError] = useState(null);

  // Load detection model on component mount
  useEffect(() => {
    const loadModel = async () => {
      try {
        const success = await detectionService.loadModel();
        setIsModelLoaded(success);
        if (!success) {
          setModelLoadError("Failed to load detection model. Please refresh the page.");
        }
      } catch (error) {
        console.error("Error loading model:", error);
        setModelLoadError("Error loading detection model. Please refresh the page.");
        setIsModelLoaded(false);
      }
    };

    loadModel();
  }, []);

  // Cleanup on component unmount or page unload
  useEffect(() => {
    const handleBeforeUnload = async () => {
      try {
        if (isDetectionActive) {
          await detectionService.stopDetectionFeed();
        }
        await cameraService.stopCamera();
        await cameraService.cleanupTempPhotos();
      } catch (error) {
        console.error("Error during cleanup:", error);
      }
    };
    
    window.addEventListener("beforeunload", handleBeforeUnload);
    
    return () => {
      window.removeEventListener("beforeunload", handleBeforeUnload);
      handleBeforeUnload();
    };
  }, [isDetectionActive]);
  
  // Fetch available cameras on component mount
  useEffect(() => {
    const fetchCameras = async () => {
      const cameraData = await cameraService.getAllCameras();
      setCameras(cameraData);
    };
    
    fetchCameras();
  }, []);

  // Handle camera selection change
  const handleCameraChange = (event) => {
    const selectedCameraId = event.target.value;
    console.log("Selected camera ID:", selectedCameraId, "Type:", typeof selectedCameraId);
    setSelectedCameraId(selectedCameraId);
    setCameraId(selectedCameraId);
  };
  
  // Handle starting detection
  const handleStartDetection = () => {
    if (!cameraId || cameraId === '') {
      alert("Please select a camera first.");
      return;
    }
    
    if (!targetLabel || targetLabel.trim() === '') {
      alert("Please enter a target label first.");
      return;
    }
    
    if (!isModelLoaded) {
      alert("Please wait for the detection model to load.");
      return;
    }
    
    setIsDetectionActive(true);
  };
  
  // Handle stopping detection
  const handleStopDetection = async () => {
    setIsDetectionActive(false);
  };

  // Handle target label input change
  const handleTargetLabelChange = (event) => {
    setTargetLabel(event.target.value);
  };

  // Handle camera detection
  const handleDetectCameras = async () => {
    setIsDetecting(true);
    try {
      const detectedCameras = await cameraService.detectCameras();
      setCameras(detectedCameras);
      
      // If the currently selected camera is no longer available, reset selection
      if (selectedCameraId && !detectedCameras.some(cam => cam.id === selectedCameraId)) {
        setSelectedCameraId('');
        setCameraId('');
      }
      
      console.log(`Detected ${detectedCameras.length} cameras`);
    } catch (error) {
      console.error("Error detecting cameras:", error);
    } finally {
      setIsDetecting(false);
    }
  };

  return (
    <Box sx={{ width: '100%', maxWidth: { sm: '100%', md: '1700px' } }}>
      {/* Model Loading Error Alert */}
      {modelLoadError && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {modelLoadError}
        </Alert>
      )}
      
      {/* Model Loading Indicator */}
      {!isModelLoaded && !modelLoadError && (
        <Alert 
          severity="info" 
          sx={{ mb: 2, display: 'flex', alignItems: 'center' }}
          icon={<CircularProgress size={20} />}
        >
          Loading detection model, please wait...
        </Alert>
      )}
      
      <Grid
        container
        spacing={2}
        columns={12}
        sx={{ mb: (theme) => theme.spacing(2) }}
      >
        <Grid size={{ xs: 12, md: 9 }}>
          <Stack spacing={3}>
            <DetectionControls
              targetLabel={targetLabel}
              onTargetLabelChange={handleTargetLabelChange}
              selectedCameraId={selectedCameraId}
              onCameraChange={handleCameraChange}
              cameras={cameras}
              onDetectCameras={handleDetectCameras}
              isDetecting={isDetecting}
              onStartDetection={handleStartDetection}
              onStopDetection={handleStopDetection}
              isDetectionActive={isDetectionActive}
              isModelLoaded={isModelLoaded}
            />
            
            <DetectionVideoFeed
              isDetectionActive={isDetectionActive}
              onStartDetection={handleStartDetection}
              onStopDetection={handleStopDetection}
              cameraId={cameraId}
              targetLabel={targetLabel}
              isModelLoaded={isModelLoaded}
            />
          </Stack>
        </Grid>
      </Grid>
    </Box>
  );
}