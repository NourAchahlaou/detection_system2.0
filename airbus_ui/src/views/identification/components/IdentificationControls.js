// components/camera/IdentificationControls.jsx - Fixed to prevent constant group loading
import React, { useState, useEffect } from "react";
import {
  Stack,
  Select,
  MenuItem,
  Button,
  CircularProgress,
  TextField,
  FormControl,
  InputLabel,
  Alert,
  Box,
  Chip
} from "@mui/material";
import { Refresh, CheckCircle } from "@mui/icons-material";

const IdentificationControls = ({
  selectedCameraId,
  onCameraChange,
  cameras,
  onDetectCameras,
  isDetecting = false,
  isSystemReady = false,
  systemHealth,
  identificationOptions,
  onIdentificationOptionsChange,
  identificationState,
  confidenceThreshold,
  onConfidenceThresholdChange,
  // Group-related props
  selectedGroupName = "",
  onGroupNameChange,
  availableGroups = [],
  onLoadAvailableGroups,
  isGroupLoaded = false,
  currentGroup = null,
  onSelectGroup,
  isGroupLoading = false
}) => {
  const [localGroupInput, setLocalGroupInput] = useState(selectedGroupName);
  const [groupInputMode, setGroupInputMode] = useState('select'); // 'select' or 'input'

  // Update local input when prop changes
  useEffect(() => {
    setLocalGroupInput(selectedGroupName);
  }, [selectedGroupName]);

  // REMOVED: Automatic group loading - only load when explicitly requested
  // The parent component now handles initial loading during system initialization

  const handleGroupInputChange = (event) => {
    const value = event.target.value;
    setLocalGroupInput(value);
    if (onGroupNameChange) {
      onGroupNameChange(value);
    }
  };

  const handleGroupSelectChange = (event) => {
    const value = event.target.value;
    setLocalGroupInput(value);
    if (onGroupNameChange) {
      onGroupNameChange(value);
    }
    // Auto-select the group when chosen from dropdown
    if (onSelectGroup && value) {
      onSelectGroup(value);
    }
  };

  const handleSelectGroup = () => {
    if (onSelectGroup && localGroupInput.trim()) {
      onSelectGroup(localGroupInput.trim());
    }
  };

  // Manual refresh groups button handler
  const handleRefreshGroups = () => {
    if (onLoadAvailableGroups) {
      console.log('Manual group refresh requested');
      onLoadAvailableGroups();
    }
  };

  const handleToggleInputMode = () => {
    setGroupInputMode(prev => prev === 'select' ? 'input' : 'select');
    setLocalGroupInput('');
    if (onGroupNameChange) {
      onGroupNameChange('');
    }
  };

  const isIdentificationActive = identificationState === 'RUNNING';
  const canSelectGroup = isSystemReady && !isIdentificationActive && !isGroupLoading;

  return (
    <Box>
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
        {/* Group Selection/Input */}
        <Box sx={{ display: 'flex', flexDirection: 'column', minWidth: 280 }}>
          <Stack direction="row" spacing={1} alignItems="center">
            {groupInputMode === 'select' && availableGroups.length > 0 ? (
              <FormControl sx={{ minWidth: 200 }}>
                <InputLabel id="group-select-label">Select Group</InputLabel>
                <Select
                  labelId="group-select-label"
                  value={localGroupInput}
                  onChange={handleGroupSelectChange}
                  label="Select Group"
                  disabled={!canSelectGroup}
                >
                  <MenuItem value="">
                    <em>Choose a group...</em>
                  </MenuItem>
                  {availableGroups.map((group) => (
                    <MenuItem key={group} value={group}>
                      {group}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            ) : (
              <TextField
                label="Group Name"
                value={localGroupInput}
                onChange={handleGroupInputChange}
                placeholder="e.g., E539, G123"
                sx={{ minWidth: 200 }}
                disabled={!canSelectGroup}
                helperText={availableGroups.length === 0 ? "Enter group name manually" : ""}
              />
            )}
            
            {/* Toggle between select and input mode */}
            {availableGroups.length > 0 && (
              <Button
                variant="text"
                size="small"
                onClick={handleToggleInputMode}
                disabled={!canSelectGroup}
                sx={{ minWidth: 'auto', px: 1 }}
              >
                {groupInputMode === 'select' ? 'Type' : 'Select'}
              </Button>
            )}
            
            {/* Select/Load Group Button */}
            {(!isGroupLoaded || currentGroup !== localGroupInput.trim()) && localGroupInput.trim() && (
              <Button
                variant="outlined"
                size="small"
                onClick={handleSelectGroup}
                disabled={!canSelectGroup || !localGroupInput.trim()}
                startIcon={isGroupLoading ? <CircularProgress size={16} /> : <CheckCircle />}
                sx={{ minWidth: 80 }}
              >
                {isGroupLoading ? 'Loading...' : 'Load'}
              </Button>
            )}
          </Stack>
          
          {/* Current Group Status */}
          {currentGroup && (
            <Box sx={{ mt: 0.5 }}>
              <Chip 
                label={`Current: ${currentGroup}`}
                color="success"
                size="small"
                variant="outlined"
              />
            </Box>
          )}
        </Box>

        {/* Camera Selection */}
        <FormControl sx={{ minWidth: 200 }}>
          <InputLabel id="camera-select-label">Camera</InputLabel>
          <Select
            labelId="camera-select-label"
            value={selectedCameraId}
            onChange={onCameraChange}
            label="Camera"
            disabled={isIdentificationActive || !isSystemReady}
          >
            <MenuItem value="">
              <em>Select a Camera</em>
            </MenuItem>
            {cameras.map((camera) => (
              <MenuItem key={camera.id} value={camera.id}>
                {camera.model}
              </MenuItem>
            ))}
          </Select>
        </FormControl>

        {/* Detect Cameras Button */}
        <Button
          variant="outlined"
          onClick={onDetectCameras}
          disabled={isDetecting || isIdentificationActive}
          startIcon={isDetecting ? <CircularProgress size={16} /> : <Refresh />}
          sx={{ minWidth: 120 }}
        >
          {isDetecting ? 'Detecting...' : 'Detect'}
        </Button>

        {/* Confidence Threshold */}
        <TextField
          label="Confidence"
          type="number"
          value={confidenceThreshold}
          onChange={(e) => onConfidenceThresholdChange(parseFloat(e.target.value))}
          inputProps={{
            min: 0.1,
            max: 1.0,
            step: 0.1
          }}
          sx={{ minWidth: 120 }}
          disabled={isIdentificationActive}
        />
      </Stack>

      {/* Status alerts for group selection */}
      {!isGroupLoaded && isSystemReady && (
        <Alert severity="info" sx={{ mt: 1, display: { xs: 'none', md: 'block' } }}>
          Please select a group to enable piece identification. Available groups: {availableGroups.length > 0 ? availableGroups.join(', ') : 'Loading...'}
        </Alert>
      )}
      
      {isGroupLoaded && currentGroup && !selectedCameraId && (
        <Alert severity="info" sx={{ mt: 1, display: { xs: 'none', md: 'block' } }}>
          Group "{currentGroup}" loaded successfully. Now select a camera to start identification.
        </Alert>
      )}
    </Box>
  );
};

export default IdentificationControls;