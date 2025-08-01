// components/BasicModeControls.jsx
import React from 'react';
import {
  Box,
  Stack,
  Alert,
  CircularProgress,
  Card,
  CardContent,
  Typography,
  Chip,
  Button,
  ButtonGroup
} from '@mui/material';
import {
  CameraAlt,
  AcUnit, // Freeze icon
  Whatshot, // Unfreeze icon
  PlayArrow
} from '@mui/icons-material';

const BasicModeControls = ({
  isStreamFrozen,
  onDemandDetecting,
  detectionInProgress,
  lastDetectionResult,
  targetLabel,
  onOnDemandDetection,
  onFreezeStream,
  onUnfreezeStream
}) => {
  return (
    <Card>
      <CardContent sx={{ py: 2 }}>
        <Typography variant="h6" gutterBottom color="primary">
          Basic Mode Controls
        </Typography>
        
        {/* Stream Status */}
        <Box sx={{ mb: 2 }}>
          <Stack direction="row" spacing={1} alignItems="center" sx={{ mb: 1 }}>
            <Chip 
              label={isStreamFrozen ? "FROZEN" : "LIVE"} 
              size="small" 
              color={isStreamFrozen ? "warning" : "success"}
              icon={isStreamFrozen ? <AcUnit /> : <PlayArrow />}
            />
            <Typography variant="caption" color="textSecondary">
              Stream Status
            </Typography>
          </Stack>
        </Box>

        {/* Control Buttons */}
        <Stack spacing={1}>
          {/* On-Demand Detection Button */}
          <Button
            variant="contained"
            size="small"
            fullWidth
            startIcon={<CameraAlt />}
            onClick={() => onOnDemandDetection({ autoUnfreeze: false })}
            disabled={onDemandDetecting || detectionInProgress || !targetLabel}
            color="primary"
          >
            {onDemandDetecting ? 'Detecting...' : 'Detect Now'}
          </Button>

          {/* Freeze/Unfreeze Controls */}
          <ButtonGroup size="small" variant="outlined" fullWidth>
            <Button
              onClick={onFreezeStream}
              disabled={isStreamFrozen || onDemandDetecting}
              startIcon={<AcUnit />}
              sx={{ flex: 1 }}
            >
              Freeze
            </Button>
            <Button
              onClick={onUnfreezeStream}
              disabled={!isStreamFrozen || onDemandDetecting}
              startIcon={<Whatshot />}
              sx={{ flex: 1 }}
            >
              Unfreeze
            </Button>
          </ButtonGroup>
        </Stack>

        {/* Last Detection Result */}
        {lastDetectionResult && (
          <Box sx={{ mt: 2, p: 1.5, bgcolor: 'background.paper', borderRadius: 1, border: '1px solid', borderColor: 'divider' }}>
            <Typography variant="subtitle2" gutterBottom>
              Last Detection Result
            </Typography>
            <Stack spacing={0.5}>
              <Typography variant="body2" color={lastDetectionResult.detected ? 'success.main' : 'text.secondary'}>
                {lastDetectionResult.detected ? '✅ TARGET FOUND' : '❌ NOT FOUND'}
              </Typography>
              {lastDetectionResult.confidence && (
                <Typography variant="caption" color="textSecondary">
                  Confidence: {(lastDetectionResult.confidence * 100).toFixed(1)}%
                </Typography>
              )}
              <Typography variant="caption" color="textSecondary">
                Processing Time: {lastDetectionResult.processingTime}ms
              </Typography>
              {lastDetectionResult.detected && (
                <Typography variant="caption" color="success.main">
                  Target "{targetLabel}" detected successfully!
                </Typography>
              )}
            </Stack>
          </Box>
        )}

        {/* Stream Frozen Alert */}
        {isStreamFrozen && (
          <Alert severity="info" sx={{ mt: 2 }}>
            <Typography variant="body2">
              Stream is frozen for detection analysis. Use controls above to unfreeze or perform detection.
            </Typography>
          </Alert>
        )}

        {/* On-Demand Detection in Progress */}
        {onDemandDetecting && (
          <Alert 
            severity="info" 
            sx={{ mt: 2 }}
            icon={<CircularProgress size={16} />}
          >
            <Typography variant="body2">
              Performing on-demand detection...
            </Typography>
          </Alert>
        )}
      </CardContent>
    </Card>
  );
};

export default BasicModeControls;