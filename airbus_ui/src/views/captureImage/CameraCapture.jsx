// pages/AppPartLibrary.jsx
import React, { useState, useEffect } from "react";
import { Stack } from "@mui/material";
import { Container } from "./components/styledComponents";
import CameraControls from "./components/CameraControls";
import VideoFeed from "./components/VideoFeed";
import { cameraService } from "./CameraService";

export default function AppPartLibrary() {
  const [targetLabel, setTargetLabel] = useState("");
  const [cameraId, setCameraId] = useState("");
  const [cameras, setCameras] = useState([]);
  const [isCameraStarted, setCameraStarted] = useState(false);
  const [selectedCameraId, setSelectedCameraId] = useState('');
  
  // Cleanup on component unmount or page unload
  useEffect(() => {
    const handleBeforeUnload = async () => {
      try {
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
  }, []);
  
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
  
  // Handle starting camera
  const handleStartCamera = async (cameraId) => {
    console.log("Starting camera with ID:", cameraId);
    if (cameraId && cameraId !== '') {
      const success = await cameraService.startCamera(cameraId);
      setCameraStarted(success);
    } else {
      alert("Please select a camera first.");
    }
  };
  
  // Handle stopping camera
  const handleStopCamera = async () => {
    await cameraService.cleanupTempPhotos();
    const success = await cameraService.stopCamera();
    if (success) {
      setCameraStarted(false);
      window.location.reload();
    }
  };

  // Handle target label input change
  const handleTargetLabelChange = (event) => {
    setTargetLabel(event.target.value);
  };

  return (
    <Container>
      <Stack spacing={3}>
        <CameraControls
          targetLabel={targetLabel}
          onTargetLabelChange={handleTargetLabelChange}
          selectedCameraId={selectedCameraId}
          onCameraChange={handleCameraChange}
          cameras={cameras}
        />
        <VideoFeed
          isCameraStarted={isCameraStarted}
          onStartCamera={handleStartCamera}
          onStopCamera={handleStopCamera}
          cameraId={cameraId}
          targetLabel={targetLabel}
        />
      </Stack>
    </Container>
  );
}