// components/camera/CameraControls.jsx
import React from "react";
import { 
  Stack, 
  TextField, 
  Select, 
  MenuItem, 
  Button,
  CircularProgress
} from "@mui/material";
import { Refresh } from "@mui/icons-material";

const CameraControls = ({ 
  targetLabel, 
  onTargetLabelChange, 
  selectedCameraId, 
  onCameraChange, 
  cameras,
  onDetectCameras,
  isDetecting = false
}) => {
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
      />
      <Select
        labelId="camera-select-label"
        value={selectedCameraId}
        onChange={onCameraChange}
        displayEmpty
        sx={{ minWidth: 200 }}
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
        disabled={isDetecting}
        startIcon={isDetecting ? <CircularProgress size={16} /> : <Refresh />}
        sx={{ minWidth: 120 }}
      >
        {isDetecting ? 'Detecting...' : 'Detect'}
      </Button>
    </Stack>
  );
};

export default CameraControls;