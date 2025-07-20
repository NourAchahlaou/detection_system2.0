// components/CameraPlaceholder.jsx
import React from "react";
import { 
  Box, 
  Button, 
  Typography, 
  Stack,
  alpha
} from "@mui/material";
import { 
  CameraAlt, 
  PlayArrow, 
  Visibility,
  VideocamOff 
} from "@mui/icons-material";

const CameraPlaceholder = ({ 
  onStartCamera, 
  cameraId, 
  buttonText = "Start Camera",
  icon = "camera",
  disabled = false,
  subtitle = "Select a camera to begin"
}) => {
  const getIcon = () => {
    switch (icon) {
      case "detection":
        return <Visibility sx={{ fontSize: 48 }} />;
      case "play":
        return <PlayArrow sx={{ fontSize: 48 }} />;
      case "camera":
      default:
        return <CameraAlt sx={{ fontSize: 48 }} />;
    }
  };

  const getTitle = () => {
    switch (icon) {
      case "detection":
        return "Detection Ready";
      case "play":
        return "Ready to Stream";
      case "camera":
      default:
        return "Camera Ready";
    }
  };

  return (
    <Box
      sx={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        height: "100%",
        width: "100%",
        minHeight: "400px",
        background: `linear-gradient(135deg, ${alpha("#667eea", 0.1)} 0%, ${alpha("#764ba2", 0.1)} 100%)`,
        borderRadius: "12px",
        border: `2px dashed ${alpha("#667eea", 0.3)}`,
        transition: "all 0.3s ease",
        "&:hover": {
          borderColor: alpha("#667eea", 0.5),
          background: `linear-gradient(135deg, ${alpha("#667eea", 0.15)} 0%, ${alpha("#764ba2", 0.15)} 100%)`,
        }
      }}
    >
      <Stack spacing={3} alignItems="center">
        {/* Icon */}
        <Box
          sx={{
            color: disabled ? "text.disabled" : "primary.main",
            opacity: disabled ? 0.5 : 1,
            transition: "all 0.3s ease"
          }}
        >
          {getIcon()}
        </Box>
        
        {/* Title */}
        <Typography 
          variant="h6" 
          sx={{ 
            fontWeight: 600,
            color: disabled ? "text.disabled" : "text.primary",
            textAlign: "center"
          }}
        >
          {disabled ? "Camera Not Available" : getTitle()}
        </Typography>
        
        {/* Subtitle */}
        <Typography 
          variant="body2" 
          sx={{ 
            color: "text.secondary",
            textAlign: "center",
            maxWidth: "300px"
          }}
        >
          {disabled ? "Please select a camera and ensure all requirements are met" : subtitle}
        </Typography>
        
        {/* Action Button */}
        <Button
          variant="contained"
          onClick={() => onStartCamera(cameraId)}
          disabled={disabled || !cameraId}
          startIcon={disabled ? <VideocamOff /> : getIcon()}
          sx={{
            minWidth: "160px",
            height: "48px",
            borderRadius: "24px",
            backgroundColor: disabled ? "action.disabledBackground" : "#667eea",
            "&:hover": {
              backgroundColor: disabled ? "action.disabledBackground" : "rgb(91, 76, 175)",
              boxShadow: disabled ? "none" : "0 8px 24px rgba(103, 126, 234, 0.4)"
            },
            "&:disabled": {
              color: "action.disabled"
            }
          }}
        >
          {disabled ? "Unavailable" : buttonText}
        </Button>
        
        {/* Camera ID Display */}
        {cameraId && (
          <Typography 
            variant="caption" 
            sx={{ 
              color: "text.secondary",
              fontFamily: "monospace",
              backgroundColor: alpha("#000", 0.05),
              padding: "4px 8px",
              borderRadius: "4px"
            }}
          >
            Camera ID: {cameraId}
          </Typography>
        )}
      </Stack>
    </Box>
  );
};

export default CameraPlaceholder;