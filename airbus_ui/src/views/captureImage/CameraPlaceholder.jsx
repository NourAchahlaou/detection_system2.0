// components/camera/CameraPlaceholder.jsx
import React from "react";
import { Typography, Fab } from "@mui/material";
import { VideocamOff, PlayArrow } from "@mui/icons-material";
import { PlaceholderContent } from "./components/styledComponents";

const CameraPlaceholder = ({ onStartCamera, cameraId }) => {
  return (
    <PlaceholderContent>
      <VideocamOff 
        sx={{ 
          fontSize: 80, 
          color: "#bbb", 
          marginBottom: 2 
        }} 
      />
      <Typography 
        variant="h5" 
        sx={{ 
          color: "#888", 
          marginBottom: 1,
          fontWeight: 500 
        }}
      >
        Camera Not Started
      </Typography>
      <Typography 
        variant="body1" 
        sx={{ 
          color: "#aaa",
          maxWidth: "300px",
          lineHeight: 1.5
        }}
      >
        Click the play button to start the camera feed
      </Typography>
      <Fab
        color="primary"
        size="large"
        onClick={() => onStartCamera(cameraId)}
        disabled={!cameraId}
        sx={{
          width: 50,
          height: 50,
          mt: 5,
          "&:hover": {
            transform: "scale(1.1)",
          },
          "&:disabled": {
            opacity: 0.5,
          },
          transition: "all 0.3s ease",
        }}
      >
        <PlayArrow sx={{ fontSize: 40 }} />
      </Fab>
    </PlaceholderContent>
  );
};

export default CameraPlaceholder;