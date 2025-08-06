// LotWorkflowPanel.jsx - UPDATED: Integrated BasicModeControls to avoid duplication
import React, { useState, useEffect, useCallback } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Button,
  Stack,
  Chip,
  Alert,
  CircularProgress,
  Divider,
  IconButton,
  Tooltip,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Collapse,
  LinearProgress
} from '@mui/material';
import {
  PlayArrow,
  Stop,
  CheckCircle,
  Warning,
  RadioButtonUnchecked,
  Refresh,
  History,
  ExpandMore,
  ExpandLess,
  Timer,
  CameraAlt,
  AcUnit,
  Whatshot,
  Info
} from '@mui/icons-material';

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
  const [showHistory, setShowHistory] = useState(false);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [localHistory, setLocalHistory] = useState(detectionHistory || []);

  // Update local history when prop changes
  useEffect(() => {
    setLocalHistory(detectionHistory || []);
  }, [detectionHistory]);

  const handleReloadHistory = useCallback(async () => {
    if (!selectedLotId || !streamManager) return;
    
    setHistoryLoading(true);
    try {
      const result = await streamManager.getLotDetectionSessions(selectedLotId);
      if (result.success) {
        setLocalHistory(result.sessions || []);
        if (onReloadHistory) {
          onReloadHistory();
        }
      }
    } catch (error) {
      console.error('Failed to reload history:', error);
    } finally {
      setHistoryLoading(false);
    }
  }, [selectedLotId, streamManager, onReloadHistory]);

  const getStatusInfo = (lot) => {
    if (!lot) return { variant: 'not-started', label: 'Not Started', icon: <RadioButtonUnchecked />, color: '#999' };
    
    if (lot.is_target_match === null) {
      return {
        variant: 'not-started',
        label: 'Not Started',
        icon: <RadioButtonUnchecked />,
        color: '#999'
      };
    } else if (lot.is_target_match === false) {
      return {
        variant: 'pending',
        label: 'Needs Correction',
        icon: <Warning />,
        color: '#ff9800'
      };
    } else {
      return {
        variant: 'completed',
        label: 'Completed',
        icon: <CheckCircle />,
        color: '#4caf50'
      };
    }
  };

  const formatDate = (dateString) => {
    if (!dateString) return 'N/A';
    return new Date(dateString).toLocaleString();
  };

  const statusInfo = getStatusInfo(currentLot);

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

      {/* Detection History */}
      <Card sx={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
        <CardContent sx={{ py: 2, flex: 1, display: 'flex', flexDirection: 'column' }}>
          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
            <Typography variant="h6" color="primary">
              Detection History
            </Typography>
            <Box sx={{ display: 'flex', gap: 0.5 }}>
              <Tooltip title="Refresh History">
                <IconButton 
                  size="small" 
                  onClick={handleReloadHistory}
                  disabled={historyLoading}
                >
                  {historyLoading ? <CircularProgress size={16} /> : <Refresh />}
                </IconButton>
              </Tooltip>
              <IconButton 
                size="small" 
                onClick={() => setShowHistory(!showHistory)}
              >
                {showHistory ? <ExpandLess /> : <ExpandMore />}
              </IconButton>
            </Box>
          </Box>

          <Box sx={{ mb: 1 }}>
            <Typography variant="body2" color="textSecondary">
              {localHistory.length} total sessions
            </Typography>
          </Box>

          <Collapse in={showHistory}>
            <Box sx={{ flex: 1, overflow: 'auto', maxHeight: '300px' }}>
              {localHistory.length === 0 ? (
                <Alert severity="info" sx={{ mt: 1 }}>
                  <Typography variant="body2">
                    No detection sessions yet. Start detecting to see history.
                  </Typography>
                </Alert>
              ) : (
                <List dense sx={{ pt: 0 }}>
                  {localHistory.slice().reverse().map((session, index) => (
                    <ListItem 
                      key={session.session_id || index}
                      sx={{ 
                        px: 0, 
                        py: 0.5,
                        borderBottom: index < localHistory.length - 1 ? '1px solid rgba(0,0,0,0.08)' : 'none'
                      }}
                    >
                      <ListItemIcon sx={{ minWidth: 32 }}>
                        {session.detected_target ? 
                          <CheckCircle sx={{ fontSize: 16, color: '#4caf50' }} /> : 
                          <Warning sx={{ fontSize: 16, color: '#ff9800' }} />
                        }
                      </ListItemIcon>
                      <ListItemText
                        primary={
                          <Typography variant="body2">
                            {session.detected_target ? 'Target Found' : 'Target Not Found'}
                            {session.confidence && (
                              <Chip 
                                size="small" 
                                label={`${Math.round(session.confidence * 100)}%`}
                                sx={{ ml: 1, height: 16, fontSize: '0.7rem' }}
                              />
                            )}
                          </Typography>
                        }
                        secondary={
                          <Typography variant="caption" color="textSecondary">
                            {formatDate(session.created_at)} • {session.processing_time}ms
                            {session.detected_piece_number && (
                              <> • Piece #{session.detected_piece_number}</>
                            )}
                          </Typography>
                        }
                      />
                    </ListItem>
                  ))}
                </List>
              )}
            </Box>
          </Collapse>

          {!showHistory && localHistory.length > 0 && (
            <Box sx={{ mt: 1 }}>
              <Typography variant="caption" color="textSecondary">
                Last session: {localHistory.length > 0 ? 
                  formatDate(localHistory[localHistory.length - 1]?.created_at) : 'None'}
              </Typography>
            </Box>
          )}
        </CardContent>
      </Card>

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