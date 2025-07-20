// components/camera/LiveDetectionView.jsx
import React from "react";
import {
  IconButton,
  Typography,
  Tooltip,
  Chip,
  Box,
  Stack
} from "@mui/material";
import { 
  Stop, 
  FiberManualRecord, 
  Visibility,
  VisibilityOff,
  Warning,
  CheckCircle
} from "@mui/icons-material";
import {
  VideoImage,
  FloatingControls,
  StatusIndicator,
  CaptureCounter,
} from "./components/styledComponents";

const LiveDetectionView = ({
  videoUrl,
  videoRef,
  showControls,
  onStopDetection,
  detectionStats,
  targetLabel
}) => {
  const { objectDetected, detectionCount, nonTargetCount } = detectionStats;

  return (
    <>
      <VideoImage
        ref={videoRef}
        src={videoUrl}
        alt="Detection Feed"
      />
      
      {/* Status Indicator */}
      <StatusIndicator data-active={true}>
        <FiberManualRecord sx={{ fontSize: 8 }} />
        DETECTING
      </StatusIndicator>
      
      {/* Detection Stats */}
      <Box
        sx={{
          position: "absolute",
          top: 16,
          left: 16,
          display: "flex",
          flexDirection: "column",
          gap: 1,
          zIndex: 10
        }}
      >
        <Chip
          icon={objectDetected ? <CheckCircle /> : <VisibilityOff />}
          label={objectDetected ? "Target Detected" : "Searching..."}
          color={objectDetected ? "success" : "default"}
          size="small"
          sx={{
            backgroundColor: objectDetected ? "rgba(76, 175, 80, 0.9)" : "rgba(0, 0, 0, 0.7)",
            color: "white",
            "& .MuiChip-icon": { color: "white" }
          }}
        />
        
        <Chip
          icon={<Visibility />}
          label={`Detections: ${detectionCount}`}
          size="small"
          sx={{
            backgroundColor: "rgba(33, 150, 243, 0.9)",
            color: "white",
            "& .MuiChip-icon": { color: "white" }
          }}
        />
        
        {nonTargetCount > 0 && (
          <Chip
            icon={<Warning />}
            label={`Non-target: ${nonTargetCount}`}
            size="small"
            sx={{
              backgroundColor: "rgba(255, 152, 0, 0.9)",
              color: "white",
              "& .MuiChip-icon": { color: "white" }
            }}
          />
        )}
      </Box>
      
      {/* Target Label Display */}
      <Box
        sx={{
          position: "absolute",
          top: 16,
          right: 16,
          backgroundColor: "rgba(0, 0, 0, 0.7)",
          color: "white",
          padding: "8px 12px",
          borderRadius: "8px",
          zIndex: 10
        }}
      >
        <Typography variant="caption" sx={{ fontWeight: "bold" }}>
          Target: {targetLabel}
        </Typography>
      </Box>
      
      {/* Floating Controls */}
      <FloatingControls sx={{ opacity: showControls ? 1 : 0.7 }}>
        <Stack direction="row" spacing={2} alignItems="center">
          <Typography variant="caption" sx={{ color: "white", minWidth: "100px", textAlign: "center" }}>
            {objectDetected ? "Target Found!" : "Detecting..."}
          </Typography>
          
          <Tooltip title="Stop Detection">
            <IconButton
              onClick={onStopDetection}
              sx={{
                color: "white",
                backgroundColor: "rgba(244, 63, 94, 0.9)",
                "&:hover": { 
                  backgroundColor: "rgba(244, 63, 94, 1)",
                  boxShadow: '0 6px 16px rgba(244, 63, 94, 0.5)',
                }
              }}
            >
              <Stop />
            </IconButton>
          </Tooltip>
        </Stack>
      </FloatingControls>
      
      {/* Detection Pulse Effect */}
      {objectDetected && (
        <Box
          sx={{
            position: "absolute",
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            border: "3px solid #4caf50",
            borderRadius: "12px",
            animation: "pulse 2s infinite",
            zIndex: 5,
            "@keyframes pulse": {
              "0%": { opacity: 0.7, transform: "scale(1)" },
              "50%": { opacity: 1, transform: "scale(1.02)" },
              "100%": { opacity: 0.7, transform: "scale(1)" }
            }
          }}
        />
      )}
    </>
  );
};

export default LiveDetectionView;