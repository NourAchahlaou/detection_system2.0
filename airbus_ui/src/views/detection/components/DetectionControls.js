// components/camera/DetectionControls.jsx
import React from "react";
import {
  Stack,
  TextField,
  Select,
  MenuItem,
  Button,
  CircularProgress,
  Typography,
  Box
} from "@mui/material";
import { Refresh, PlayArrow, Stop, Visibility } from "@mui/icons-material";

const DetectionControls = ({
  targetLabel,
  onTargetLabelChange,
  selectedCameraId,
  onCameraChange,
  cameras,
  onDetectCameras,
  isDetecting = false,
  onStartDetection,
  onStopDetection,
  isDetectionActive = false,
  isModelLoaded = false
}) => {
  const canStartDetection = targetLabel && selectedCameraId && isModelLoaded && !isDetectionActive;
  
  return (
    <Stack
      direction="row"
      sx={{
        display: { xs: 'none', md: 'flex' },
        width: '100%',
        alignItems: { xs: 'flex-start', md: 'center' },
        justifyContent: 'center',
        maxWidth: { sm: '100%', md: '1700px' },
        pt: 1.5,
      }}
      spacing={2}
    >
      <TextField
        label="Target Label"
        value={targetLabel}
        onChange={onTargetLabelChange}
        placeholder="e.g., G123.12345.123.12"
        required
        sx={{ minWidth: 250 }}
        disabled={isDetectionActive}
      />
      
      <Select
        labelId="camera-select-label"
        value={selectedCameraId}
        onChange={onCameraChange}
        displayEmpty
        sx={{ minWidth: 200 }}
        disabled={isDetectionActive}
      >
        <MenuItem value="" disabled>Select a Camera</MenuItem>
        {cameras.map((camera) => (
          <MenuItem key={camera.id} value={camera.id}>
            {camera.model}
          </MenuItem>
        ))}
      </Select>
      
      <Button
        variant="outlined"
        onClick={onDetectCameras}
        disabled={isDetecting || isDetectionActive}
        startIcon={isDetecting ? <CircularProgress size={16} /> : <Refresh />}
        sx={{ minWidth: 120 }}
      >
        {isDetecting ? 'Detecting...' : 'Detect'}
      </Button>
      
      {!isDetectionActive ? (
        <Button
          variant="contained"
          onClick={onStartDetection}
          disabled={!canStartDetection}
          startIcon={<PlayArrow />}
          sx={{ 
            minWidth: 140,
            backgroundColor: "#667eea",
            "&:hover": { backgroundColor: "rgb(91, 76, 175)" }
          }}
        >
          Start Detection
        </Button>
      ) : (
        <Button
          variant="contained"
          onClick={onStopDetection}
          startIcon={<Stop />}
          sx={{ 
            minWidth: 140,
            backgroundColor: "rgba(244, 63, 94, 0.9)",
            "&:hover": { backgroundColor: "rgba(244, 63, 94, 1)" }
          }}
        >
          Stop Detection
        </Button>
      )}
      
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
        <Visibility sx={{ fontSize: 16, color: isModelLoaded ? 'success.main' : 'text.secondary' }} />
        <Typography variant="caption" color={isModelLoaded ? 'success.main' : 'text.secondary'}>
          {isModelLoaded ? 'Model Ready' : 'Loading Model...'}
        </Typography>
      </Box>
    </Stack>
  );
};

export default DetectionControls;