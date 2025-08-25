// components/camera/IdentificationControls.jsx - Updated with notification window and reorganized layout
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
  Chip,
  Paper,
  IconButton,
  Typography,
  Snackbar
} from "@mui/material";
import { Refresh, CheckCircle, Close, Info } from "@mui/icons-material";

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
  const [showGroupNotification, setShowGroupNotification] = useState(false);
  const [showSuccessNotification, setShowSuccessNotification] = useState(false);

  // Update local input when prop changes
  useEffect(() => {
    setLocalGroupInput(selectedGroupName);
  }, [selectedGroupName]);

  // Show notifications based on group state
  useEffect(() => {
    if (!isGroupLoaded && isSystemReady) {
      setShowGroupNotification(true);
      setShowSuccessNotification(false);
    } else if (isGroupLoaded && currentGroup && !selectedCameraId) {
      setShowSuccessNotification(true);
      setShowGroupNotification(false);
    } else {
      setShowGroupNotification(false);
      setShowSuccessNotification(false);
    }
  }, [isGroupLoaded, isSystemReady, currentGroup, selectedCameraId]);

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
    <Box sx={{ position: 'relative' }}>
      <Stack
        direction="row"
        sx={{
          display: { xs: 'none', md: 'flex' },
          width: '100%',
          alignItems: { xs: 'flex-start', md: 'center' },
          justifyContent: 'flex-start',
          maxWidth: { sm: '100%', md: '1700px' },
          pt: 1.5,
        }}
        spacing={2}
      >


        {/* Controllers grouped together - Center */}
        <Stack direction="row" spacing={2} sx={{ flex: 1, justifyContent: 'center' }}>
                  {/* Load Group Button - Left side */}
          <Box sx={{ display: 'flex', flexDirection: 'column' }}>
            {(!isGroupLoaded || currentGroup !== localGroupInput.trim()) && localGroupInput.trim() && (
              <Button
                variant="outlined"
                size="medium"
                onClick={handleSelectGroup}
                disabled={!canSelectGroup || !localGroupInput.trim()}
                startIcon={isGroupLoading ? <CircularProgress size={16} /> : <CheckCircle />}
                sx={{ minWidth: 100, mb: 1 }}
              >
                {isGroupLoading ? 'Loading...' : 'Load Group'}
              </Button>
            )}
          </Box>
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
         </Box> 
        </Stack>
      </Stack>

      {/* Group Selection Notification - Right side floating */}
      <Snackbar
        open={showGroupNotification}
        anchorOrigin={{ vertical: 'top', horizontal: 'right' }}
        sx={{ 
          position: 'fixed',
          top: 80,
          right: 20,
          zIndex: 1400,
          '& .MuiSnackbarContent-root': {
            minWidth: 350
          }
        }}
      >
        <Paper
          elevation={6}
          sx={{
            p: 2,
            backgroundColor: '#e3f2fd',
            border: '1px solid #2196f3',
            borderRadius: 2,
            position: 'relative'
          }}
        >
          <IconButton
            size="small"
            onClick={() => setShowGroupNotification(false)}
            sx={{ 
              position: 'absolute', 
              top: 4, 
              right: 4,
              color: '#1976d2'
            }}
          >
            <Close fontSize="small" />
          </IconButton>
          <Stack direction="row" spacing={1} alignItems="flex-start">
            <Info sx={{ color: '#1976d2', mt: 0.5 }} />
            <Box>
              <Typography variant="body2" sx={{ fontWeight: 600, color: '#1976d2' }}>
                Select a Group to Continue
              </Typography>
              <Typography variant="body2" sx={{ mt: 0.5, color: '#1565c0' }}>
                Please select a group to enable piece identification.
              </Typography>
              <Typography variant="caption" sx={{ color: '#1565c0' }}>
                Available groups: {availableGroups.length > 0 ? availableGroups.join(', ') : 'Loading...'}
              </Typography>
            </Box>
          </Stack>
        </Paper>
      </Snackbar>

      {/* Success Notification - Right side floating */}
      <Snackbar
        open={showSuccessNotification}
        anchorOrigin={{ vertical: 'top', horizontal: 'right' }}
        sx={{ 
          position: 'fixed',
          top: 80,
          right: 20,
          zIndex: 1400,
          '& .MuiSnackbarContent-root': {
            minWidth: 350
          }
        }}
      >
        <Paper
          elevation={6}
          sx={{
            p: 2,
            backgroundColor: '#e8f5e8',
            border: '1px solid #4caf50',
            borderRadius: 2,
            position: 'relative'
          }}
        >
          <IconButton
            size="small"
            onClick={() => setShowSuccessNotification(false)}
            sx={{ 
              position: 'absolute', 
              top: 4, 
              right: 4,
              color: '#388e3c'
            }}
          >
            <Close fontSize="small" />
          </IconButton>
          <Stack direction="row" spacing={1} alignItems="flex-start">
            <CheckCircle sx={{ color: '#4caf50', mt: 0.5 }} />
            <Box>
              <Typography variant="body2" sx={{ fontWeight: 600, color: '#388e3c' }}>
                Group Loaded Successfully
              </Typography>
              <Typography variant="body2" sx={{ mt: 0.5, color: '#2e7d32' }}>
                Group "{currentGroup}" is ready. Now select a camera to start identification.
              </Typography>
            </Box>
          </Stack>
        </Paper>
      </Snackbar>
    </Box>
  );
};

export default IdentificationControls;