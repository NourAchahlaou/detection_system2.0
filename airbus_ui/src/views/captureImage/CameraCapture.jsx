// pages/AppPartLibrary.jsx
import React, { useState, useEffect, useRef } from "react";
import { Box, Grid, Stack } from '@mui/material';

import CameraControls from "./components/CameraControls";
import VideoFeed from "./components/VideoFeed";
import ImageSlider from "./ImageSlider";
import { cameraService } from "./CameraService";
import HorizontalImageSlider from "./HorizontalImageSlider";

export default function AppPartLibrary() {
  const [targetLabel, setTargetLabel] = useState("");
  const [cameraId, setCameraId] = useState("");
  const [cameras, setCameras] = useState([]);
  const [isCameraStarted, setCameraStarted] = useState(false);
  const [selectedCameraId, setSelectedCameraId] = useState('');
  const [imageRefreshTrigger, setImageRefreshTrigger] = useState(0);
  const [horizontalSliderOpen, setHorizontalSliderOpen] = useState(false);
  const [horizontalSliderInitialIndex, setHorizontalSliderInitialIndex] = useState(0);
  
  // Add ref to store VideoFeed's image count callback
  const imageCountCallbackRef = useRef(null);

  const handleImageDoubleClick = (imageIndex) => {
    setHorizontalSliderInitialIndex(imageIndex);
    setHorizontalSliderOpen(true);
  };
  const handleHorizontalSliderClose = () => {
    setHorizontalSliderOpen(false);
  };  
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

  // Handle image captured - trigger ImageSlider refresh
  const handleImageCaptured = () => {
    // Increment trigger to cause ImageSlider to refresh
    setImageRefreshTrigger(prev => prev + 1);
  };

  // Handle image count change from ImageSlider (to update VideoFeed)
  const handleImageCountChange = (newCount) => {
    if (imageCountCallbackRef.current) {
      imageCountCallbackRef.current(newCount);
    }
  };

  // Set the callback ref when VideoFeed provides it
  const setImageCountCallback = (callback) => {
    imageCountCallbackRef.current = callback;
  };

  return (
    <Box sx={{ width: '100%', maxWidth: { sm: '100%', md: '1700px' } }}>
      <Grid
        container
        spacing={2}
        columns={12}
        sx={{ mb: (theme) => theme.spacing(2) }}
      >
        <Grid size={{ xs: 12, md: 9 }}>
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
              onImageCaptured={handleImageCaptured}
              onSetImageCountCallback={setImageCountCallback} // New prop
            />
          </Stack>
        </Grid>
        
        <Grid size={{ xs: 12, md: 3 }}>
          {/* Image Slider Component - now event-driven */}
          <ImageSlider 
            targetLabel={targetLabel}
            refreshTrigger={imageRefreshTrigger}
            onImageCountChange={handleImageCountChange} 
            onImageDoubleClick={handleImageDoubleClick} 

          />         
        </Grid>
      </Grid>
      <HorizontalImageSlider
        open={horizontalSliderOpen}
        onClose={handleHorizontalSliderClose}
        targetLabel={targetLabel}
        initialIndex={horizontalSliderInitialIndex}
        onImageCountChange={handleImageCountChange}
      />      
    </Box>
  );
}