// components/camera/VideoFeed.jsx
import React, { useState, useEffect, useRef } from "react";
import { VideoCard } from "./styledComponents";
import CameraPlaceholder from "../CameraPlaceholder";
import LiveCameraView from "../LiveCamera";
import CaptureDialog from "../CaptureDialog";
import { cameraService } from "../CameraService";

const VideoFeed = ({
  isCameraStarted,
  onStartCamera,
  onStopCamera,
  cameraId,
  targetLabel,
  onImagesCaptured  // New prop to communicate with parent
}) => {
  const [videoUrl, setVideoUrl] = useState("");
  const [dialogOpen, setDialogOpen] = useState(false);
  const [capturedImages, setCapturedImages] = useState([]);
  const [capturedImagesCount, setCapturedImagesCount] = useState(0);
  const [snapshotEffect, setSnapshotEffect] = useState(false);
  const [isCapturing, setIsCapturing] = useState(false);
  const [showControls, setShowControls] = useState(false);
  const requiredCaptures = 10;
  const videoRef = useRef(null);

  useEffect(() => {
    if (isCameraStarted) {
      setVideoUrl("/api/artifact_keeper/camera/video_feed");
    } else {
      setVideoUrl("");
    }
  }, [isCameraStarted]);

  // Reset captured images when target label changes
  useEffect(() => {
    if (targetLabel !== '') {
      // Clean up old blob URLs
      capturedImages.forEach(imageUrl => {
        if (imageUrl.startsWith('blob:')) {
          URL.revokeObjectURL(imageUrl);
        }
      });
      
      setCapturedImages([]);
      setCapturedImagesCount(0);
      
      // Notify parent component about the reset
      if (onImagesCaptured) {
        onImagesCaptured([]);
      }
    }
  }, [targetLabel, capturedImages, onImagesCaptured]);

  // Notify parent component when images change
  useEffect(() => {
    if (onImagesCaptured) {
      const imageObjects = capturedImages.map((url, index) => ({
        url,
        src: url,
        image_name: `capture_${Date.now()}_${index}.jpg`,
        timestamp: new Date().toISOString(),
        isTemporary: true,
        isLocal: true  // Flag to distinguish from server temp images
      }));
      onImagesCaptured(imageObjects);
    }
  }, [capturedImages, onImagesCaptured]);

  const handleCaptureImages = async () => {
    if (isCapturing) return;
    
    if (!targetLabel || targetLabel.trim() === "") {
      alert("Please enter a piece label before capturing images.");
      return;
    }

    setIsCapturing(true);
    setSnapshotEffect(true);
    
    try {
      const imageUrl = await cameraService.captureImages(targetLabel);
      if (imageUrl) {
        setCapturedImages((prevImages) => [...prevImages, imageUrl]);
        setCapturedImagesCount((prevCount) => prevCount + 1);
        
        if (capturedImagesCount + 1 >= requiredCaptures) {
          setDialogOpen(true);
        }
      } else {
        alert("Failed to capture image. Please try again.");
      }
    } catch (error) {
      console.error("Error during image capture:", error);
      alert("Error capturing image. Please try again.");
    } finally {
      setIsCapturing(false);
      setTimeout(() => setSnapshotEffect(false), 1000);
    }
  };

  const handleSaveImages = async () => {
    if (!targetLabel) {
      alert("Piece label is required.");
      return;
    }
    
    const success = await cameraService.saveImagesToDatabase(targetLabel);
    if (success) {
      // Clean up captured images
      capturedImages.forEach(imageUrl => {
        if (imageUrl.startsWith('blob:')) {
          URL.revokeObjectURL(imageUrl);
        }
      });
      setCapturedImages([]);
      setCapturedImagesCount(0);
      
      // Notify parent component
      if (onImagesCaptured) {
        onImagesCaptured([]);
      }
      
      await onStopCamera();
      window.location.reload();
    }
    setDialogOpen(false);
  };

  // Cleanup blob URLs when component unmounts
  useEffect(() => {
    return () => {
      capturedImages.forEach(imageUrl => {
        if (imageUrl.startsWith('blob:')) {
          URL.revokeObjectURL(imageUrl);
        }
      });
    };
  }, [capturedImages]);

  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center" }}>
      <VideoCard
        cameraActive={isCameraStarted}
        onMouseEnter={() => setShowControls(true)}
        onMouseLeave={() => setShowControls(false)}
      >
        {isCameraStarted ? (
          <LiveCameraView
            videoUrl={videoUrl}
            videoRef={videoRef}
            showControls={showControls}
            onCaptureImages={handleCaptureImages}
            onStopCamera={onStopCamera}
            isCapturing={isCapturing}
            capturedImagesCount={capturedImagesCount}
            requiredCaptures={requiredCaptures}
            snapshotEffect={snapshotEffect}
            capturedImages={capturedImages}
          />
        ) : (
          <CameraPlaceholder 
            onStartCamera={onStartCamera}
            cameraId={cameraId}
          />
        )}
      </VideoCard>

      <CaptureDialog
        open={dialogOpen}
        onClose={() => setDialogOpen(false)}
        onSaveImages={handleSaveImages}
        capturedImagesCount={capturedImagesCount}
        targetLabel={targetLabel}
      />
    </div>
  );
};

export default VideoFeed;