// components/camera/IdentificationControls.jsx
import React from "react";
import {
  Stack,
  Select,
  MenuItem,
  Button,
  CircularProgress,

} from "@mui/material";
import { Refresh} from "@mui/icons-material";

const IdentificationControls = ({
  selectedCameraId,
  onCameraChange,
  cameras,
  onDetectCameras,
  isDetecting = false,
  isIdentificationActive = false,
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

      
      <Select
        labelId="camera-select-label"
        value={selectedCameraId}
        onChange={onCameraChange}
        displayEmpty
        sx={{ minWidth: 200 }}
        disabled={isIdentificationActive}
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
        disabled={isDetecting || isIdentificationActive}
        startIcon={isDetecting ? <CircularProgress size={16} /> : <Refresh />}
        sx={{ minWidth: 120 }}
      >
        {isDetecting ? 'Detecting...' : 'Detect'}
      </Button>

    </Stack>
  );
};

export default IdentificationControls;