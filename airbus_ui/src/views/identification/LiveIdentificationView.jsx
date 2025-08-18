// components/camera/LiveIdentificationView.jsx - Fixed with proper null checks
import React, { useState, useEffect, useCallback } from "react";
import {
  IconButton,
  Typography,
  Tooltip,
  Chip,
  Box,
  Stack,
  LinearProgress,
  Badge
} from "@mui/material";
import { 
  Stop, 
  Visibility,
  VisibilityOff,
  Warning,
  CheckCircle,
  Speed,
  Memory,
  NetworkCheck,
  SignalCellularAlt
} from "@mui/icons-material";
import {
  VideoImage,
  FloatingControls,
} from "./components/styledComponents.js";

const LiveIdentificationView = ({
  videoUrl,
  videoRef,
  showControls,
  onStopIdentification,
  identificationStats = {}, // Default empty object
  streamStatus = {},
  performanceMetrics = {}
}) => {
  // Destructure with defaults to prevent undefined errors
  const { 
    objectDetected = false, 
    detectionCount = 0, 
    nonTargetCount = 0,
    avgProcessingTime = 0,
    detectionFps = 0, // Default to 0 instead of undefined
    queueDepth = 0,
    streamQuality = 85,
    lastDetectionTime = null
  } = identificationStats || {};

  const [showAdvancedStats, setShowAdvancedStats] = useState(false);
  const [connectionQuality, setConnectionQuality] = useState('good');

  // Determine connection quality based on performance metrics
  useEffect(() => {
    if (avgProcessingTime > 200 || queueDepth > 5) {
      setConnectionQuality('poor');
    } else if (avgProcessingTime > 100 || queueDepth > 2) {
      setConnectionQuality('fair');
    } else {
      setConnectionQuality('good');
    }
  }, [avgProcessingTime, queueDepth]);

  // Format last detection time
  const formatLastDetection = useCallback(() => {
    if (!lastDetectionTime) return 'No detections yet';
    
    try {
      const detectionTime = new Date(lastDetectionTime);
      const now = new Date();
      const diffSeconds = Math.floor((now - detectionTime) / 1000);
      
      if (diffSeconds < 5) return 'Just now';
      if (diffSeconds < 60) return `${diffSeconds}s ago`;
      if (diffSeconds < 3600) return `${Math.floor(diffSeconds / 60)}m ago`;
      return detectionTime.toLocaleTimeString();
    } catch (error) {
      return 'Invalid time';
    }
  }, [lastDetectionTime]);

  // Get quality color for performance indicators
  const getQualityColor = (value, thresholds) => {
    if (value <= thresholds.good) return 'success';
    if (value <= thresholds.fair) return 'warning';
    return 'error';
  };

  // Performance thresholds
  const performanceThresholds = {
    processingTime: { good: 50, fair: 100 },
    queueDepth: { good: 1, fair: 3 },
    detectionFps: { good: 4, fair: 2 }
  };

  // Safe number formatting functions
  const safeToFixed = (value, digits = 1) => {
    if (value === null || value === undefined || isNaN(value)) {
      return '0';
    }
    return Number(value).toFixed(digits);
  };

  return (
    <>
      <VideoImage
        ref={videoRef}
        src={videoUrl}
        alt="Optimized Detection Feed"
        style={{
          filter: !streamStatus?.isConnected ? 'grayscale(100%)' : 'none',
          opacity: streamStatus?.isConnected ? 1 : 0.7
        }}
      />
      
      {/* Primary Detection Stats */}
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
        {/* Main Detection Status */}
        <Chip
          icon={objectDetected ? <CheckCircle /> : <VisibilityOff />}
          label={objectDetected ? "Target Detected" : "Searching..."}
          color={objectDetected ? "success" : "default"}
          size="small"
          sx={{
            backgroundColor: objectDetected ? "rgba(76, 175, 80, 0.9)" : "rgba(0, 0, 0, 0.7)",
            color: "white",
            "& .MuiChip-icon": { color: "white" },
            animation: objectDetected ? "pulse 2s infinite" : "none",
            "@keyframes pulse": {
              "0%": { opacity: 0.8 },
              "50%": { opacity: 1 },
              "100%": { opacity: 0.8 }
            }
          }}
        />
        
        {/* Detection Counter */}
        <Badge badgeContent={detectionCount || 0} color="primary" max={999}>
          <Chip
            icon={<Visibility />}
            label={`Detections: ${detectionCount || 0}`}
            size="small"
            sx={{
              backgroundColor: "rgba(33, 150, 243, 0.9)",
              color: "white",
              "& .MuiChip-icon": { color: "white" }
            }}
          />
        </Badge>
        
        {/* Non-target Detections */}
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

        {/* Performance Indicator */}
        <Chip
          icon={<Speed />}
          label={`${safeToFixed(avgProcessingTime, 0)}ms`}
          size="small"
          color={getQualityColor(avgProcessingTime, performanceThresholds.processingTime)}
          sx={{
            backgroundColor: avgProcessingTime < 50 
              ? "rgba(76, 175, 80, 0.9)" 
              : avgProcessingTime < 100 
                ? "rgba(255, 152, 0, 0.9)" 
                : "rgba(244, 67, 54, 0.9)",
            color: "white",
            "& .MuiChip-icon": { color: "white" }
          }}
        />

        {/* Queue Depth Warning */}
        {queueDepth > 0 && (
          <Chip
            icon={<Memory />}
            label={`Queue: ${queueDepth}`}
            size="small"
            color={getQualityColor(queueDepth, performanceThresholds.queueDepth)}
            sx={{
              backgroundColor: queueDepth <= 1 
                ? "rgba(76, 175, 80, 0.9)" 
                : queueDepth <= 3 
                  ? "rgba(255, 152, 0, 0.9)" 
                  : "rgba(244, 67, 54, 0.9)",
              color: "white",
              "& .MuiChip-icon": { color: "white" }
            }}
          />
        )}
      </Box>
      
      {/* Target Label and Stream Info */}
      <Box
        sx={{
          position: "absolute",
          top: 16,
          right: 16,
          display: "flex",
          flexDirection: "column",
          gap: 1,
          alignItems: "flex-end",
          zIndex: 10
        }}
      >
        {/* Target Label */}
        <Box
          sx={{
            backgroundColor: "rgba(0, 0, 0, 0.8)",
            color: "white",
            padding: "8px 12px",
            borderRadius: "8px",
            display: "flex",
            alignItems: "center",
            gap: 1
          }}
        >
        </Box>

        {/* Stream Quality - FIXED: Using safe number formatting */}
        <Box
          sx={{
            backgroundColor: "rgba(0, 0, 0, 0.7)",
            color: "white",
            padding: "4px 8px",
            borderRadius: "6px",
            display: "flex",
            alignItems: "center",
            gap: 0.5
          }}
        >
          <NetworkCheck sx={{ fontSize: 14 }} />
          <Typography variant="caption">
            {streamQuality || 85}% • {safeToFixed(detectionFps, 1)}fps
          </Typography>
        </Box>

        {/* Connection Quality Indicator */}
        <Chip
          icon={<SignalCellularAlt />}
          label={connectionQuality.toUpperCase()}
          size="small"
          sx={{
            backgroundColor: connectionQuality === 'good' 
              ? "rgba(76, 175, 80, 0.9)" 
              : connectionQuality === 'fair' 
                ? "rgba(255, 152, 0, 0.9)" 
                : "rgba(244, 67, 54, 0.9)",
            color: "white",
            "& .MuiChip-icon": { color: "white" }
          }}
        />
      </Box>

      {/* Advanced Stats Panel (Toggleable) */}
      {showAdvancedStats && (
        <Box
          sx={{
            position: "absolute",
            bottom: 80,
            left: 16,
            right: 16,
            backgroundColor: "rgba(0, 0, 0, 0.8)",
            color: "white",
            padding: 2,
            borderRadius: 2,
            zIndex: 10
          }}
        >
          <Typography variant="subtitle2" sx={{ mb: 1 }}>
            Performance Metrics
          </Typography>
          
          <Stack spacing={1}>
            {/* Processing Time */}
            <Box>
              <Typography variant="caption">
                Processing Time: {safeToFixed(avgProcessingTime, 1)}ms
              </Typography>
              <LinearProgress
                variant="determinate"
                value={Math.min(((avgProcessingTime || 0) / 200) * 100, 100)}
                color={avgProcessingTime < 50 ? "success" : avgProcessingTime < 100 ? "warning" : "error"}
                sx={{ height: 4, borderRadius: 2 }}
              />
            </Box>

            {/* Queue Depth */}
            <Box>
              <Typography variant="caption">
                Queue Depth: {queueDepth || 0}/10
              </Typography>
              <LinearProgress
                variant="determinate"
                value={((queueDepth || 0) / 10) * 100}
                color={queueDepth < 2 ? "success" : queueDepth < 5 ? "warning" : "error"}
                sx={{ height: 4, borderRadius: 2 }}
              />
            </Box>

            {/* Detection Rate - FIXED: Using safe number formatting */}
            <Box>
              <Typography variant="caption">
                Detection Rate: {safeToFixed(detectionFps, 1)}/5.0 fps
              </Typography>
              <LinearProgress
                variant="determinate"
                value={((detectionFps || 0) / 5) * 100}
                color={detectionFps > 4 ? "success" : detectionFps > 2 ? "warning" : "error"}
                sx={{ height: 4, borderRadius: 2 }}
              />
            </Box>

            {/* Last Detection */}
            <Typography variant="caption" sx={{ opacity: 0.8 }}>
              Last Detection: {formatLastDetection()}
            </Typography>
          </Stack>
        </Box>
      )}
      
      {/* Floating Controls */}
      <FloatingControls sx={{ opacity: showControls ? 1 : 0.7 }}>
        <Stack direction="row" spacing={2} alignItems="center">
          {/* Status Text */}
          <Typography 
            variant="caption" 
            sx={{ 
              color: "white", 
              minWidth: "120px", 
              textAlign: "center",
              fontWeight: "bold"
            }}
          >
            {!streamStatus?.isConnected ? "Reconnecting..." :
             objectDetected ? "Target Found!" : 
             `Detecting... (${detectionCount || 0})`}
          </Typography>

          {/* Toggle Advanced Stats */}
          <Tooltip title="Toggle Performance Stats">
            <IconButton
              onClick={() => setShowAdvancedStats(!showAdvancedStats)}
              sx={{
                color: "white",
                backgroundColor: showAdvancedStats 
                  ? "rgba(33, 150, 243, 0.9)" 
                  : "rgba(0, 0, 0, 0.6)",
                "&:hover": { 
                  backgroundColor: showAdvancedStats 
                    ? "rgba(33, 150, 243, 1)" 
                    : "rgba(0, 0, 0, 0.8)"
                }
              }}
            >
              <Speed />
            </IconButton>
          </Tooltip>
          
          {/* Stop Detection */}
          <Tooltip title="Stop Optimized Detection">
            <IconButton
              onClick={onStopIdentification}
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
      
      {/* Detection Success Pulse Effect */}
      {objectDetected && streamStatus?.isConnected && (
        <Box
          sx={{
            position: "absolute",
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            border: "4px solid #4caf50",
            borderRadius: "12px",
            animation: "detectionPulse 1.5s infinite",
            zIndex: 5,
            "@keyframes detectionPulse": {
              "0%": { opacity: 0.6, transform: "scale(1)", borderColor: "#4caf50" },
              "50%": { opacity: 1, transform: "scale(1.01)", borderColor: "#81c784" },
              "100%": { opacity: 0.6, transform: "scale(1)", borderColor: "#4caf50" }
            }
          }}
        />
      )}

      {/* Connection Lost Overlay */}
      {!streamStatus?.isConnected && (
        <Box
          sx={{
            position: "absolute",
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            backgroundColor: "rgba(244, 67, 54, 0.1)",
            border: "2px dashed #f44336",
            borderRadius: "12px",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            zIndex: 8
          }}
        >
          <Box sx={{ textAlign: "center", color: "white" }}>
            <Warning sx={{ fontSize: 48, mb: 1 }} />
            <Typography variant="h6">Connection Lost</Typography>
            <Typography variant="caption">Attempting to reconnect...</Typography>
          </Box>
        </Box>
      )}

      {/* Performance Warning Overlay */}
      {queueDepth > 5 && (
        <Box
          sx={{
            position: "absolute",
            bottom: 120,
            left: "50%",
            transform: "translateX(-50%)",
            backgroundColor: "rgba(255, 152, 0, 0.9)",
            color: "white",
            padding: "8px 16px",
            borderRadius: 2,
            zIndex: 10,
            animation: "warning-blink 2s infinite",
            "@keyframes warning-blink": {
              "0%": { opacity: 0.8 },
              "50%": { opacity: 1 },
              "100%": { opacity: 0.8 }
            }
          }}
        >
          <Typography variant="caption" sx={{ fontWeight: "bold" }}>
            ⚠️ High Queue Depth - Performance Impact
          </Typography>
        </Box>
      )}
    </>
  );
};

export default LiveIdentificationView;