// components/LiveCamera.jsx
import React from "react";
import { Box, IconButton, Typography, Tooltip } from "@mui/material";
import { CameraAlt, Stop, FiberManualRecord } from "@mui/icons-material";
import { 
  VideoImage, 
  FloatingControls, 
  StatusIndicator, 
  CaptureCounter, 
  SnapshotEffect,
  CapturedImagesStack 
} from "./components/styledComponents";

const LiveCameraView = ({
  videoUrl,
  videoRef,
  showControls,
  onCaptureImages,
  onStopCamera,
  isCapturing,
  capturedImagesCount,
  requiredCaptures,
  snapshotEffect,
  capturedImages
}) => {
  return (
    <>
      <VideoImage
        ref={videoRef}
        src={videoUrl}
        alt="Video Feed"
      />
      
      {/* Status Indicator - Fixed: Use data attributes instead of invalid props */}
      <StatusIndicator data-active={true}>
        <FiberManualRecord sx={{ fontSize: 8 }} />
        LIVE
      </StatusIndicator>

      {/* Capture Counter */}
      {capturedImagesCount > 0 && (
        <CaptureCounter>
          {capturedImagesCount}/{requiredCaptures} Captured
        </CaptureCounter>
      )}

      {/* Floating Controls */}
      <FloatingControls sx={{ opacity: showControls ? 1 : 0.7 }}>
        <Tooltip title="Capture Image">
          <IconButton
            onClick={onCaptureImages}
            disabled={isCapturing || capturedImagesCount >= requiredCaptures}
            sx={{ 
              color: "white",
              backgroundColor: "rgba(76, 175, 80, 0.8)",
              "&:hover": { backgroundColor: "rgba(76, 175, 80, 1)" },
              "&:disabled": { backgroundColor: "rgba(255, 255, 255, 0.2)" }
            }}
          >
            <CameraAlt />
          </IconButton>
        </Tooltip>
        
        <Typography variant="caption" sx={{ color: "white", minWidth: "80px", textAlign: "center" }}>
          {isCapturing ? "Capturing..." : `${capturedImagesCount}/${requiredCaptures}`}
        </Typography>
        
        <Tooltip title="Stop Camera">
          <IconButton
            onClick={onStopCamera}
            sx={{ 
              color: "white",
              backgroundColor: "rgba(244, 67, 54, 0.8)",
              "&:hover": { backgroundColor: "rgba(244, 67, 54, 1)" }
            }}
          >
            <Stop />
          </IconButton>
        </Tooltip>
      </FloatingControls>

      {/* Snapshot Effect */}
      {snapshotEffect && (
        <SnapshotEffect>
          📸
        </SnapshotEffect>
      )}

      {/* Captured Images Stack */}
      <CapturedImagesStack>
        {capturedImages.slice(-3).map((imageUrl, index) => (
          <Box
            key={capturedImages.length - 3 + index}
            sx={{
              position: "absolute",
              bottom: index * 8 + "px",
              right: index * 8 + "px",
              transform: `rotate(${(index - 1) * 5}deg)`,
              zIndex: 3 - index,
            }}
          >
            <img
              src={imageUrl}
              alt={`Captured ${capturedImages.length - 3 + index + 1}`}
              style={{
                width: "80px",
                height: "60px",
                objectFit: "cover",
                border: "3px solid white",
                borderRadius: "8px",
                boxShadow: "0 4px 12px rgba(0, 0, 0, 0.14)",
              }}
            />
          </Box>
        ))}
      </CapturedImagesStack>
    </>
  );
};

export default LiveCameraView;