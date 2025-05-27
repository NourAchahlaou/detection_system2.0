// components/camera/CameraControls.jsx
import React from "react";
import { 
  Box, 
  TextField, 
  Select, 
  MenuItem 
} from "@mui/material";

const CameraControls = ({ 
  targetLabel, 
  onTargetLabelChange, 
  selectedCameraId, 
  onCameraChange, 
  cameras 
}) => {
  return (
    <Box 
      className="controls"
      sx={{
        display: "flex",
        gap: 2,
        flexWrap: "wrap",
        alignItems: "center",
        justifyContent: "center"
      }}
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
    </Box>
  );
};

export default CameraControls;