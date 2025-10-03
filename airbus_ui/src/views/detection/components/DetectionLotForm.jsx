// components/DetectionLotForm.jsx - Enhanced lot information form with sidebar support
import React, { useState, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  CardHeader,
  Typography,
  TextField,
  Button,
  Stack,
  Divider,
  Chip,
  Alert,
  IconButton,
  Tooltip,
  InputAdornment,
  FormControl,
  FormLabel,
  RadioGroup,
  FormControlLabel,
  Radio,
  Collapse,
  CircularProgress,
  Autocomplete,
  Avatar
} from '@mui/material';
import {
  Close,
  Save,
  Add,
  Inventory,
  Numbers,
  Label,
  Info,
  CheckCircle,
  Warning,
  Refresh,
  Search,
  Category,
  PlayArrow
} from '@mui/icons-material';

const DetectionLotForm = ({
  isOpen,
  isInSidebar = false, // Prop for sidebar mode
  onClose,
  onSubmit,
  onCreateLot, // Only prop for creating lot
  cameraId,
  targetLabel,
  detectionOptions,
  isSubmitting = false,
  existingLots = [],
  onRefreshLots,
}) => {
  const apiBaseUrl = '/api/artifact_keeper' // Add API base URL prop
  
  // Form state
  const [formData, setFormData] = useState({
    lotName: '',
    expectedPieceLabel: '', // Changed from expectedPieceId to expectedPieceLabel
    expectedPieceId: '', // This will be auto-populated from selected piece
    expectedPieceNumber: '',
    useExistingLot: false,
    selectedLotId: ''
  });

  // Pieces state
  const [pieces, setPieces] = useState([]);
  const [selectedPiece, setSelectedPiece] = useState(null);
  const [loadingPieces, setLoadingPieces] = useState(false);
  const [piecesError, setPiecesError] = useState(null);

  // Validation state
  const [errors, setErrors] = useState({});
  const [isValid, setIsValid] = useState(false);

  // UI state
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [isRefreshingLots, setIsRefreshingLots] = useState(false);

  // Fetch pieces from API
  const fetchPieces = async () => {
    setLoadingPieces(true);
    setPiecesError(null);
    try {
      const response = await fetch(`${apiBaseUrl}/captureImage/pieces`);
      if (!response.ok) {
        throw new Error(`Failed to fetch pieces: ${response.statusText}`);
      }
      const piecesData = await response.json();
      setPieces(piecesData);
    } catch (error) {
      console.error('Error fetching pieces:', error);
      setPiecesError(error.message);
    } finally {
      setLoadingPieces(false);
    }
  };

  // Fetch piece details by label
  const fetchPieceByLabel = async (pieceLabel) => {
    try {
      const response = await fetch(`${apiBaseUrl}/captureImage/pieces/${encodeURIComponent(pieceLabel)}`);
      if (!response.ok) {
        throw new Error(`Failed to fetch piece details: ${response.statusText}`);
      }
      const pieceData = await response.json();
      return pieceData;
    } catch (error) {
      console.error('Error fetching piece by label:', error);
      throw error;
    }
  };

  // Load pieces when form opens
  useEffect(() => {
    if (isOpen) {
      fetchPieces();
    }
  }, [isOpen]);

  // Reset form when opened/closed
  useEffect(() => {
    if (isOpen) {
      setFormData({
        lotName: '',
        expectedPieceLabel: '',
        expectedPieceId: '',
        expectedPieceNumber: '',
        useExistingLot: false,
        selectedLotId: ''
      });
      setSelectedPiece(null);
      setErrors({});
      setShowAdvanced(false);
    }
  }, [isOpen]);

  // Handle piece selection - FIXED
  const handlePieceSelection = async (event, value) => {
    if (value) {
      try {
        // Fetch detailed piece information
        const pieceDetails = await fetchPieceByLabel(value.piece_label);
        setSelectedPiece(pieceDetails);
        setFormData(prev => ({
          ...prev,
          expectedPieceLabel: value.piece_label,
          expectedPieceId: pieceDetails.id.toString() // Ensure it's a string for form handling
        }));
        
        // Clear any previous errors
        setErrors(prev => {
          const newErrors = { ...prev };
          delete newErrors.expectedPieceLabel;
          delete newErrors.expectedPieceId;
          return newErrors;
        });
        
      } catch (error) {
        console.error('Error selecting piece:', error);
        setErrors(prev => ({
          ...prev,
          expectedPieceLabel: 'Failed to load piece details'
        }));
        setSelectedPiece(null);
        setFormData(prev => ({
          ...prev,
          expectedPieceLabel: '',
          expectedPieceId: ''
        }));
      }
    } else {
      setSelectedPiece(null);
      setFormData(prev => ({
        ...prev,
        expectedPieceLabel: '',
        expectedPieceId: ''
      }));
    }
  };

  // Auto-generate lot name based on current timestamp
  const generateLotName = () => {
    const now = new Date();
    const timestamp = now.toISOString().replace(/[:.]/g, '-').slice(0, 19);
    const pieceLabel = formData.expectedPieceLabel ? `_${formData.expectedPieceLabel}` : '';
    const lotName = `Lot_${timestamp}${pieceLabel}`;
    setFormData(prev => ({ ...prev, lotName }));
  };

  // Validation logic - FIXED
  useEffect(() => {
    const newErrors = {};
    let valid = true;

    if (formData.useExistingLot) {
      // Validate existing lot selection
      if (!formData.selectedLotId) {
        newErrors.selectedLotId = 'Please select a lot';
        valid = false;
      }
    } else {
      // Validate new lot creation
      if (!formData.lotName.trim()) {
        newErrors.lotName = 'OF(fabrication order) is required';
        valid = false;
      } else if (formData.lotName.length > 100) {
        newErrors.lotName = 'Lot name must be 100 characters or less';
        valid = false;
      }

      if (!formData.expectedPieceLabel) {
        newErrors.expectedPieceLabel = 'Please select a piece';
        valid = false;
      }

      // Check if expectedPieceId is properly set
      if (!formData.expectedPieceId) {
        newErrors.expectedPieceId = 'Piece ID is required (auto-filled from piece selection)';
        valid = false;
      }

      if (!formData.expectedPieceNumber || formData.expectedPieceNumber <= 0) {
        newErrors.expectedPieceNumber = 'Expected piece number must be a positive number';
        valid = false;
      }
    }

    setErrors(newErrors);
    setIsValid(valid);
  }, [formData]);

  // Handle form field changes
  const handleFieldChange = (field, value) => {
    setFormData(prev => ({
      ...prev,
      [field]: value
    }));
  };

  // Handle radio button change for lot selection
  const handleLotModeChange = (event) => {
    const useExisting = event.target.value === 'existing';
    setFormData(prev => ({
      ...prev,
      useExistingLot: useExisting,
      // Clear the opposite mode's data
      ...(useExisting ? 
        { lotName: '', expectedPieceLabel: '', expectedPieceId: '', expectedPieceNumber: '' } :
        { selectedLotId: '' }
      )
    }));
    if (useExisting) {
      setSelectedPiece(null);
    }
  };

  // Handle refresh lots
  const handleRefreshLots = async () => {
    setIsRefreshingLots(true);
    try {
      if (onRefreshLots) {
        await onRefreshLots();
      }
    } catch (error) {
      console.error('Error refreshing lots:', error);
    } finally {
      setIsRefreshingLots(false);
    }
  };

  // Handle form submission - UPDATED TO HANDLE BOTH CREATE AND EXISTING LOT
// REPLACE the entire handleSubmit function with:
const handleSubmit = () => {
  if (!isValid || isSubmitting) return;

  const submissionData = {
    ...formData,
    cameraId: parseInt(cameraId),
    targetLabel,
    detectionOptions
  };

  if (formData.useExistingLot) {
    // Use existing lot
    onSubmit({
      type: 'existing_lot',
      lotId: parseInt(formData.selectedLotId),
      ...submissionData
    });
  } else {
    // Create new lot only
    onCreateLot({
      type: 'create_only',
      lotName: formData.lotName.trim(),
      expectedPieceId: parseInt(formData.expectedPieceId),
      expectedPieceNumber: parseInt(formData.expectedPieceNumber),
      expectedPieceLabel: formData.expectedPieceLabel,
      ...submissionData
    });
  }
};

  // Get selected lot info for display
  const selectedLot = existingLots.find(lot => lot.lot_id === parseInt(formData.selectedLotId));

  if (!isOpen) return null;

  // Content JSX
  const formContent = (
    <Stack spacing={2}>
      {/* Detection Context Info */}
      <Alert severity="info" icon={<Info />} sx={{ fontSize: '0.875rem' }}>
        <Typography variant="body2" sx={{ fontSize: '0.8rem' }}>
          <strong>Camera:</strong> {cameraId} | <strong>Target:</strong> {targetLabel}
        </Typography>
        <Typography variant="caption" color="textSecondary" sx={{ fontSize: '0.75rem' }}>
          Quality: {detectionOptions.quality || 85}% | 
          Mode: {detectionOptions.detectionFps ? `${detectionOptions.detectionFps} FPS` : 'On-demand'}
        </Typography>
      </Alert>

      {/* Lot Mode Selection */}
      <FormControl component="fieldset">
        <FormLabel component="legend">
          <Typography variant="subtitle2" sx={{ fontWeight: 'bold', fontSize: '0.875rem' }}>
            Lot Management
          </Typography>
        </FormLabel>
        <RadioGroup
          value={formData.useExistingLot ? 'existing' : 'new'}
          onChange={handleLotModeChange}
          row={!isInSidebar} // Stack vertically in sidebar for better space usage
        >
          <FormControlLabel 
            value="new" 
            control={<Radio size="small" />} 
            label={<Typography variant="body2">Create New</Typography>}
          />
          <FormControlLabel 
            value="existing" 
            control={<Radio size="small" />} 
            label={<Typography variant="body2">Use Existing</Typography>}
          />
        </RadioGroup>
      </FormControl>

      <Divider />

      {/* New Lot Creation Form */}
      {!formData.useExistingLot && (
        <Stack spacing={2}>
          <Typography variant="subtitle2" color="primary" sx={{ fontWeight: 'bold', fontSize: '0.875rem' }}>
            📦 New Lot Information
          </Typography>

          {/* Piece Selection */}
          <FormControl fullWidth error={!!errors.expectedPieceLabel}>
            <Autocomplete
              options={pieces}
              size="small"
              getOptionLabel={(option) => option.piece_label || ''}
              value={pieces.find(p => p.piece_label === formData.expectedPieceLabel) || null}
              onChange={handlePieceSelection}
              loading={loadingPieces}
              disabled={isSubmitting}
              renderInput={(params) => (
                <TextField
                  {...params}
                  label="Select Piece"
                  size="small"
                  error={!!errors.expectedPieceLabel}
                  helperText={errors.expectedPieceLabel || 'Choose expected piece'}
                  required
                  InputProps={{
                    ...params.InputProps,
                    startAdornment: (
                      <InputAdornment position="start">
                        <Category fontSize="small" />
                      </InputAdornment>
                    ),
                    endAdornment: (
                      <>
                        {loadingPieces ? <CircularProgress size={16} /> : null}
                        {params.InputProps.endAdornment}
                      </>
                    ),
                  }}
                />
              )}
              renderOption={(props, option) => (
                <Box component="li" {...props} sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  {option.piece_img && (
                    <Avatar
                      src={option.piece_img.image_url}
                      sx={{ width: 24, height: 24 }}
                      variant="rounded"
                    />
                  )}
                  <Box>
                    <Typography variant="body2" sx={{ fontWeight: 'medium', fontSize: '0.8rem' }}>
                      {option.piece_label}
                    </Typography>
                    <Typography variant="caption" color="textSecondary" sx={{ fontSize: '0.7rem' }}>
                      ID: {option.piece_id}
                    </Typography>
                  </Box>
                </Box>
              )}
              noOptionsText={piecesError ? "Error loading pieces" : "No pieces found"}
            />
          </FormControl>

          {/* Debug Info - Show the selected piece ID */}
          {formData.expectedPieceId && (
            <Alert severity="success" sx={{ fontSize: '0.75rem' }}>
              <Typography variant="caption">
                ✅ Piece ID {formData.expectedPieceId} selected for "{formData.expectedPieceLabel}"
              </Typography>
            </Alert>
          )}

          {/* Selected Piece Info */}
          {selectedPiece && (
            <Card variant="outlined" sx={{ backgroundColor: 'rgba(76, 175, 80, 0.04)' }}>
              <CardContent sx={{ py: 1.5 }}>
                <Typography variant="subtitle2" sx={{ fontWeight: 'bold', mb: 1, fontSize: '0.8rem' }}>
                  Selected Piece
                </Typography>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5 }}>
                  {selectedPiece.piece_img && (
                    <Avatar
                      src={selectedPiece.piece_img.image_url}
                      sx={{ width: 40, height: 40 }}
                      variant="rounded"
                    />
                  )}
                  <Stack spacing={0.5}>
                    <Typography variant="body2" sx={{ fontSize: '0.8rem' }}>
                      <strong>Label:</strong> {selectedPiece.piece_label}
                    </Typography>
                    <Typography variant="body2" sx={{ fontSize: '0.8rem' }}>
                      <strong>ID:</strong> {selectedPiece.piece_id}
                    </Typography>
                    {selectedPiece.description && (
                      <Typography variant="caption" color="textSecondary" sx={{ fontSize: '0.7rem' }}>
                        {selectedPiece.description}
                      </Typography>
                    )}
                  </Stack>
                </Box>
              </CardContent>
            </Card>
          )}

          {/* Lot Name */}
          <TextField
            label="Fabrication Order (OF)"
            size="small"
            value={formData.lotName}
            onChange={(e) => handleFieldChange('lotName', e.target.value)}
            error={!!errors.lotName}
            helperText={errors.lotName || 'Fabrication Order(OF) for this lot'}
            fullWidth
            required
            InputProps={{
              startAdornment: (
                <InputAdornment position="start">
                  <Label fontSize="small" />
                </InputAdornment>
              ),
              endAdornment: (
                <InputAdornment position="end">
                  <Tooltip title="Auto-generate lot name">
                    <IconButton 
                      size="small" 
                      onClick={generateLotName}
                      disabled={isSubmitting}
                    >
                      <Add fontSize="small" />
                    </IconButton>
                  </Tooltip>
                </InputAdornment>
              )
            }}
          />

          {/* Expected Piece Number */}
          <TextField
            label="Expected Piece Number"
            type="number"
            size="small"
            value={formData.expectedPieceNumber}
            onChange={(e) => handleFieldChange('expectedPieceNumber', parseInt(e.target.value) || '')}
            error={!!errors.expectedPieceNumber}
            helperText={errors.expectedPieceNumber || 'Number/quantity expected'}
            fullWidth
            required
            InputProps={{
              startAdornment: (
                <InputAdornment position="start">
                  <Numbers fontSize="small" />
                </InputAdornment>
              )
            }}
          />
        </Stack>
      )}

      {/* Existing Lot Selection */}
      {formData.useExistingLot && (
        <Stack spacing={2}>
          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <Typography variant="subtitle2" color="primary" sx={{ fontWeight: 'bold', fontSize: '0.875rem' }}>
              📋 Select Existing Lot
            </Typography>
            <Tooltip title="Refresh lot list">
              <IconButton 
                size="small" 
                onClick={handleRefreshLots}
                disabled={isRefreshingLots || isSubmitting}
              >
                {isRefreshingLots ? <CircularProgress size={14} /> : <Refresh fontSize="small" />}
              </IconButton>
            </Tooltip>
          </Box>

          {existingLots.length === 0 ? (
            <Alert severity="warning" sx={{ fontSize: '0.8rem' }}>
              No existing lots available. Create a new lot instead.
            </Alert>
          ) : (
            <FormControl fullWidth error={!!errors.selectedLotId}>
              <TextField
                select
                size="small"
                label="Select Lot"
                value={formData.selectedLotId}
                onChange={(e) => handleFieldChange('selectedLotId', e.target.value)}
                SelectProps={{ native: true }}
                helperText={errors.selectedLotId || 'Choose an existing lot'}
                required
              >
                <option value="">Select a lot...</option>
                {existingLots.map((lot) => (
                  <option key={lot.lot_id} value={lot.lot_id}>
                    {lot.lot_name} - #{lot.expected_piece_number} 
                    {lot.is_target_match ? ' ✅' : ' 🔄'}
                  </option>
                ))}
              </TextField>
            </FormControl>
          )}

          {/* Selected Lot Info */}
          {selectedLot && (
            <Card variant="outlined" sx={{ backgroundColor: 'rgba(25, 118, 210, 0.04)' }}>
              <CardContent sx={{ py: 1.5 }}>
                <Typography variant="subtitle2" sx={{ fontWeight: 'bold', mb: 1, fontSize: '0.8rem' }}>
                  Selected Lot Details
                </Typography>
                <Stack spacing={0.5}>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <Typography variant="body2" color="textSecondary" sx={{ fontSize: '0.75rem' }}>Name:</Typography>
                    <Typography variant="body2" sx={{ fontWeight: 'medium', fontSize: '0.75rem' }}>
                      {selectedLot.lot_name}
                    </Typography>
                  </Box>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <Typography variant="body2" color="textSecondary" sx={{ fontSize: '0.75rem' }}>Expected:</Typography>
                    <Typography variant="body2" sx={{ fontWeight: 'medium', fontSize: '0.75rem' }}>
                      ID {selectedLot.expected_piece_id} - #{selectedLot.expected_piece_number}
                    </Typography>
                  </Box>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <Typography variant="body2" color="textSecondary" sx={{ fontSize: '0.75rem' }}>Status:</Typography>
                    <Chip
                      size="small"
                      icon={selectedLot.is_target_match ? <CheckCircle fontSize="small" /> : <Warning fontSize="small" />}
                      label={selectedLot.is_target_match ? 'Completed' : 'Pending'}
                      color={selectedLot.is_target_match ? 'success' : 'warning'}
                      variant="outlined"
                      sx={{ fontSize: '0.7rem', height: 20 }}
                    />
                  </Box>
                </Stack>
              </CardContent>
            </Card>
          )}
        </Stack>
      )}

      {/* Pieces Loading Error */}
      {piecesError && (
        <Alert severity="error" action={
          <Button size="small" onClick={fetchPieces}>
            Retry
          </Button>
        }>
          <Typography variant="body2" sx={{ fontSize: '0.8rem' }}>
            Failed to load pieces: {piecesError}
          </Typography>
        </Alert>
      )}

      {/* Advanced Options */}
      <Button
        variant="text"
        size="small"
        onClick={() => setShowAdvanced(!showAdvanced)}
        sx={{ alignSelf: 'flex-start', fontSize: '0.8rem' }}
      >
        {showAdvanced ? 'Hide' : 'Show'} Advanced Options
      </Button>

      <Collapse in={showAdvanced}>
        <Stack spacing={2}>
          <Divider />
          <Typography variant="subtitle2" color="textSecondary" sx={{ fontSize: '0.8rem' }}>
            Advanced Detection Settings
          </Typography>
          
          <Alert severity="info" icon={<Info fontSize="small" />} sx={{ fontSize: '0.8rem' }}>
            <Typography variant="body2" sx={{ fontSize: '0.75rem' }}>
              Current detection will use:
            </Typography>
            <Typography variant="caption" component="div" sx={{ fontSize: '0.7rem' }}>
              • Quality: {detectionOptions.quality || 85}%<br/>
              • FPS: {detectionOptions.detectionFps || 'On-demand'}<br/>
              • Priority: {detectionOptions.priority || 1}<br/>
              • Adaptive Quality: {detectionOptions.enableAdaptiveQuality ? 'Enabled' : 'Disabled'}
            </Typography>
          </Alert>
        </Stack>
      </Collapse>

      <Divider />

      {/* Action Buttons - SIMPLIFIED TO SINGLE SUBMIT */}
      <Stack direction={isInSidebar ? "column" : "row"} spacing={1} sx={{ justifyContent: 'flex-end' }}>
        {!isInSidebar && (
          <Button
            variant="outlined"
            size="small"
            onClick={onClose}
            disabled={isSubmitting}
          >
            Cancel
          </Button>
        )}
        
        {/* Single Submit Button */}
        <Button
          variant="contained"
          size="small"
          onClick={handleSubmit}
          disabled={!isValid || isSubmitting}
          startIcon={
            isSubmitting ? (
              <CircularProgress size={14} />
            ) : (
              formData.useExistingLot ? <PlayArrow fontSize="small" /> : <Save fontSize="small" />
            )
          }
          sx={{ fontSize: '0.8rem' }}
        >
          {isSubmitting ? 'Processing...' : (
            formData.useExistingLot ? 'Use Existing Lot' : 'Create Lot'
          )}
        </Button>
      </Stack>

      {/* Helpful Information - UPDATED */}
      <Alert severity="info" sx={{ mt: 2, fontSize: '0.8rem' }}>
        <Typography variant="body2" sx={{ fontWeight: 'bold', mb: 1, fontSize: '0.8rem' }}>
          What happens next:
        </Typography>
        <Typography variant="body2" component="div" sx={{ fontSize: '0.75rem' }}>
          {formData.useExistingLot ? (
            <>
              1. Selected lot will be used for detection<br/>
              2. Lot information ready for detection session<br/>
              3. Lot status can be updated separately<br/>
            </>
          ) : (
            <>
              1. New lot "{formData.lotName || '[Name]'}" will be created<br/>
              2. Expected piece: {formData.expectedPieceLabel || '[Select piece]'} (ID: {formData.expectedPieceId || '[Auto-filled]'})<br/>
              3. Lot saved for future detection use<br/>
              4. Detection can be started separately
            </>
          )}
        </Typography>
      </Alert>
    </Stack>
  );

  // Render based on mode
  if (isInSidebar) {
    // Sidebar mode - no overlay, no modal styling
    return (
      <Box sx={{ width: '100%', height: '100%' }}>
        <CardHeader
          title={
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <Inventory color="primary" fontSize="small" />
              <Typography variant="h6" sx={{ fontSize: '1rem' }}>Lot Setup</Typography>
            </Box>
          }
          action={onClose && (
            <IconButton onClick={onClose} disabled={isSubmitting} size="small">
              <Close fontSize="small" />
            </IconButton>
          )}
          sx={{ pb: 1, px: 2, pt: 2 }}
        />
        <CardContent sx={{ pt: 0, px: 2, pb: 2, maxHeight: 'calc(100vh - 200px)', overflowY: 'auto' }}>
          {formContent}
        </CardContent>
      </Box>
    );
  }

  // Modal mode - original overlay behavior
  return (
    <Box
      sx={{
        position: 'fixed',
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        backgroundColor: 'rgba(0, 0, 0, 0.5)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        zIndex: 1300,
        p: 2
      }}
    >
      <Card
        sx={{
          width: '100%',
          maxWidth: 500,
          maxHeight: '90vh',
          overflow: 'auto'
        }}
      >
        <CardHeader
          title={
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <Inventory color="primary" />
              <Typography variant="h6">Detection Lot Setup</Typography>
            </Box>
          }
          action={
            <IconButton onClick={onClose} disabled={isSubmitting}>
              <Close />
            </IconButton>
          }
          sx={{ pb: 1 }}
        />

        <CardContent sx={{ pt: 0 }}>
          {formContent}
        </CardContent>
      </Card>
    </Box>
  );
};

export default DetectionLotForm;