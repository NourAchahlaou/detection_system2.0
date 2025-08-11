// LotWorkflowPanel.jsx - UPDATED: Integrated BasicModeControls to avoid duplication
import React, { useState, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,

  Alert,
  CircularProgress,

  LinearProgress
} from '@mui/material';


// Import BasicModeControls component
import BasicModeControls from './BasicModeControls';

const LotWorkflowPanel = ({
  currentLot,
  selectedLotId,
  detectionHistory,
  detectionInProgress,
  onDemandDetecting,
  lastDetectionResult,
  isStreamFrozen,
  targetLabel,
  onDetectLot,
  onFreezeStream,
  onUnfreezeStream,
  onStopWorkflow,
  onReloadHistory,
  streamManager
}) => {

  const [localHistory, setLocalHistory] = useState(detectionHistory || []);

  // Update local history when prop changes
  useEffect(() => {
    setLocalHistory(detectionHistory || []);
  }, [detectionHistory]);





  if (!currentLot) {
    return (
      <Card>
        <CardContent>
          <Typography variant="h6" gutterBottom color="primary">
            Lot Workflow
          </Typography>
          <Alert severity="info">
            No lot selected for workflow
          </Alert>
        </CardContent>
      </Card>
    );
  }

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2, height: '100%' }}>
      {/* BasicModeControls Integration */}
      <BasicModeControls
        isStreamFrozen={isStreamFrozen}
        onDemandDetecting={onDemandDetecting}
        detectionInProgress={detectionInProgress}
        lastDetectionResult={lastDetectionResult}
        targetLabel={targetLabel}
        onOnDemandDetection={onDetectLot} // Use lot detection instead of generic detection
        onFreezeStream={onFreezeStream}
        onUnfreezeStream={onUnfreezeStream}
      />


      {/* Progress indicator for active detection */}
      {(detectionInProgress || onDemandDetecting) && (
        <Card>
          <CardContent sx={{ py: 2 }}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
              <CircularProgress size={20} />
              <Box sx={{ flex: 1 }}>
                <Typography variant="body2" sx={{ fontWeight: 'medium' }}>
                  {detectionInProgress ? 'Processing detection...' : 'Preparing detection...'}
                </Typography>
                <LinearProgress sx={{ mt: 1 }} />
              </Box>
            </Box>
          </CardContent>
        </Card>
      )}

      {/* Lot completion status */}
      {currentLot.is_target_match === true && (
        <Alert severity="success" sx={{ mt: 1 }}>
          <Typography variant="body2" sx={{ fontWeight: 'bold' }}>
            🎉 Lot Completed Successfully!
          </Typography>
          <Typography variant="caption">
            All requirements have been met for this lot.
          </Typography>
        </Alert>
      )}

      {currentLot.is_target_match === false && localHistory.length > 0 && (
        <Alert severity="warning" sx={{ mt: 1 }}>
          <Typography variant="body2" sx={{ fontWeight: 'bold' }}>
            ⚠️ Lot Needs Correction
          </Typography>
          <Typography variant="caption">
            Continue detecting until the correct piece is identified.
          </Typography>
        </Alert>
      )}
    </Box>
  );
};

export default LotWorkflowPanel;