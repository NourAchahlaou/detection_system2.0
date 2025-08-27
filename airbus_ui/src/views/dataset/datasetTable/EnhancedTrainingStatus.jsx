import React, { useState, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  LinearProgress,
  Chip,
  Button,
  IconButton,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  List,
  ListItem,
  ListItemText,
  Divider,
  Grid,
  Tooltip,
  Collapse,
  Accordion,
  AccordionSummary,
  AccordionDetails,
} from '@mui/material';
import {
  PlayArrow,
  Stop,
  Pause,
  RestartAlt as Resume,
  ExpandMore,
  History,
  Analytics,
  Refresh,
  CheckCircle,
  Schedule,
  Computer,
} from '@mui/icons-material';
import { datasetService } from '../datasetService';

const TrainingStatusComponent = ({ onTrainingStateChange }) => {
  const [trainingStatus, setTrainingStatus] = useState(null);
  const [trainingSessions, setTrainingSessions] = useState([]);
  const [persistedSessionInfo, setPersistedSessionInfo] = useState(null);
  const [showSessionsDialog, setShowSessionsDialog] = useState(false);
  const [showLogsDialog, setShowLogsDialog] = useState(false);
  const [trainingLogs, setTrainingLogs] = useState([]);
  const [loading, setLoading] = useState(false);
  const [expanded, setExpanded] = useState(false);

  // Fetch training sessions (only when dialog is opened)
  const fetchTrainingSessions = async () => {
    try {
      const response = await datasetService.getTrainingSessions({ limit: 10 });
      setTrainingSessions(response.data?.sessions || []);
    } catch (error) {
      console.error('Error fetching training sessions:', error);
    }
  };

  // Manual refresh function for buttons
  const refreshTrainingData = async () => {
    try {
      const statusResponse = await datasetService.getTrainingStatus(true);
      const newStatus = statusResponse.data;

      setTrainingStatus(newStatus);

      if (newStatus?.session_info) {
        setPersistedSessionInfo(newStatus.session_info);
      }

      if (onTrainingStateChange) {
        const currentPersistedInfo = newStatus?.session_info || persistedSessionInfo;
        const hasActiveSession = newStatus?.is_training || 
          (currentPersistedInfo && !currentPersistedInfo.completed_at);
        
        onTrainingStateChange(hasActiveSession);
      }
    } catch (error) {
      console.error('Error refreshing training data:', error);
    }
  };

  // Fetch training logs (only when dialog is opened)
  const fetchTrainingLogs = async () => {
    try {
      const response = await datasetService.getTrainingLogs({ limit: 100 });
      setTrainingLogs(response.data?.data?.logs || []);
    } catch (error) {
      console.error('Error fetching training logs:', error);
    }
  };

  // Stop training
  const handleStopTraining = async () => {
    setLoading(true);
    try {
      await datasetService.stopTraining();
      setPersistedSessionInfo(null);
      await refreshTrainingData();
      
      if (onTrainingStateChange) {
        onTrainingStateChange(false);
      }
    } catch (error) {
      console.error('Error stopping training:', error);
    } finally {
      setLoading(false);
    }
  };

  // Pause training
  const handlePauseTraining = async () => {
    setLoading(true);
    try {
      await datasetService.pauseTraining();
      await refreshTrainingData();
    } catch (error) {
      console.error('Error pausing training:', error);
    } finally {
      setLoading(false);
    }
  };

  // Resume training session using session ID from paused session
  const handleResumeSession = async (sessionId) => {
    setLoading(true);
    try {
      await datasetService.resumeTrainingSession(sessionId);
      await refreshTrainingData();
    } catch (error) {
      console.error('Error resuming training session:', error);
    } finally {
      setLoading(false);
    }
  };

  // Get status info
  const getStatusInfo = () => {
    if (!trainingStatus) return { color: 'default', text: 'Unknown', icon: null };
    
    if (trainingStatus.is_training) {
      return { 
        color: 'success', 
        text: 'Training in Progress', 
        icon: <PlayArrow fontSize="small" />
      };
    } else if (persistedSessionInfo && !persistedSessionInfo.completed_at) {
      return { 
        color: 'warning', 
        text: 'Paused', 
        icon: <Pause fontSize="small" />
      };
    } else if (trainingStatus.session_info || persistedSessionInfo) {
      return { 
        color: 'info', 
        text: 'Session Available', 
        icon: <Schedule fontSize="small" />
      };
    } else {
      return { 
        color: 'default', 
        text: 'Idle', 
        icon: <Schedule fontSize="small" />
      };
    }
  };

  // Format time duration
  const formatDuration = (seconds) => {
    if (!seconds) return 'N/A';
    const hrs = Math.floor(seconds / 3600);
    const mins = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);
    return `${hrs}:${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  // Initial data fetch only - no polling
  useEffect(() => {
    refreshTrainingData();
  }, []);

  // Use persisted session info or current session info
  const currentSessionInfo = trainingStatus?.session_info || persistedSessionInfo;
  const statusInfo = getStatusInfo();

  return (
    <Card sx={{ mb: 2, border: '1px solid rgba(102, 126, 234, 0.2)' }}>
      <CardContent>
        <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
          <Box display="flex" alignItems="center" gap={1}>
            <Typography variant="h6" color="#667eea" fontWeight="600">
              Training Status
            </Typography>
            <Chip
              icon={statusInfo.icon}
              label={statusInfo.text}
              color={statusInfo.color}
              size="small"
              variant="outlined"
            />
          </Box>
          
          <Box display="flex" gap={1}>
            <Tooltip title="Refresh Status">
              <IconButton onClick={refreshTrainingData} size="small">
                <Refresh fontSize="small" />
              </IconButton>
            </Tooltip>
            
            <Tooltip title="View Sessions">
              <IconButton 
                onClick={() => {
                  fetchTrainingSessions();
                  setShowSessionsDialog(true);
                }} 
                size="small"
                color="primary"
              >
                <History fontSize="small" />
              </IconButton>
            </Tooltip>
            
            {(trainingStatus?.is_training || currentSessionInfo) && (
              <Tooltip title="View Logs">
                <IconButton 
                  onClick={() => {
                    fetchTrainingLogs();
                    setShowLogsDialog(true);
                  }} 
                  size="small"
                  color="primary"
                >
                  <Analytics fontSize="small" />
                </IconButton>
              </Tooltip>
            )}
          </Box>
        </Box>

        {currentSessionInfo && (
          <Accordion expanded={expanded} onChange={(e, isExpanded) => setExpanded(isExpanded)}>
            <AccordionSummary expandIcon={<ExpandMore />}>
              <Box display="flex" alignItems="center" width="100%" gap={2}>
                <Typography variant="body2" fontWeight="600">
                  {currentSessionInfo.session_name}
                </Typography>
                {trainingStatus?.is_training && (
                  <Box display="flex" alignItems="center" gap={1} flexGrow={1}>
                    <Typography variant="caption" color="text.secondary">
                      Epoch {currentSessionInfo.current_epoch} / {currentSessionInfo.epochs}
                    </Typography>
                    <LinearProgress 
                      variant="determinate" 
                      value={currentSessionInfo.progress_percentage || 0}
                      sx={{ flexGrow: 1, height: 6, borderRadius: 3 }}
                    />
                    <Typography variant="caption" fontWeight="600" color="primary">
                      {Math.round(currentSessionInfo.progress_percentage || 0)}%
                    </Typography>
                  </Box>
                )}
                {!trainingStatus?.is_training && currentSessionInfo && (
                  <Typography variant="caption" color="text.secondary">
                    Last Progress: {Math.round(currentSessionInfo.progress_percentage || 0)}%
                  </Typography>
                )}
              </Box>
            </AccordionSummary>
            
            <AccordionDetails>
              <Grid container spacing={2}>
                <Grid item xs={12} sm={6}>
                  <Box>
                    <Typography variant="caption" color="text.secondary" display="block">
                      Piece Labels
                    </Typography>
                    <Box display="flex" flexWrap="wrap" gap={0.5} mt={0.5}>
                      {currentSessionInfo.piece_labels?.map((label, index) => (
                        <Chip key={index} label={label} size="small" variant="outlined" />
                      ))}
                    </Box>
                  </Box>
                </Grid>
                
                <Grid item xs={12} sm={6}>
                  <Grid container spacing={1}>
                    <Grid item xs={6}>
                      <Typography variant="caption" color="text.secondary" display="block">
                        Model Type
                      </Typography>
                      <Typography variant="body2" fontWeight="600">
                        {currentSessionInfo.model_type}
                      </Typography>
                    </Grid>
                    <Grid item xs={6}>
                      <Typography variant="caption" color="text.secondary" display="block">
                        Device
                      </Typography>
                      <Box display="flex" alignItems="center" gap={0.5}>
                        <Computer fontSize="small" color="action" />
                        <Typography variant="body2">
                          {currentSessionInfo.device_used || 'N/A'}
                        </Typography>
                      </Box>
                    </Grid>
                  </Grid>
                </Grid>

                {currentSessionInfo.total_images && (
                  <Grid item xs={12}>
                    <Box display="flex" gap={3}>
                      <Box>
                        <Typography variant="caption" color="text.secondary">Total Images</Typography>
                        <Typography variant="body2" fontWeight="600">
                          {currentSessionInfo.total_images}
                        </Typography>
                      </Box>
                      <Box>
                        <Typography variant="caption" color="text.secondary">Validation</Typography>
                        <Typography variant="body2" fontWeight="600">
                          {currentSessionInfo.validation_images || 0}
                        </Typography>
                      </Box>
                      <Box>
                        <Typography variant="caption" color="text.secondary">Batch Size</Typography>
                        <Typography variant="body2" fontWeight="600">
                          {currentSessionInfo.batch_size}
                        </Typography>
                      </Box>
                      {currentSessionInfo.current_epoch && (
                        <Box>
                          <Typography variant="caption" color="text.secondary">Current Epoch</Typography>
                          <Typography variant="body2" fontWeight="600">
                            {currentSessionInfo.current_epoch} / {currentSessionInfo.epochs}
                          </Typography>
                        </Box>
                      )}
                    </Box>
                  </Grid>
                )}
              </Grid>
            </AccordionDetails>
          </Accordion>
        )}

        {/* Control Buttons - Show when we have session info or training is active */}
        {(trainingStatus?.is_training || currentSessionInfo) && (
          <Box display="flex" gap={1} mt={2}>
            {trainingStatus?.is_training ? (
              <>
                <Button
                  variant="outlined"
                  color="warning"
                  startIcon={<Pause />}
                  onClick={handlePauseTraining}
                  disabled={loading}
                  size="small"
                >
                  Pause
                </Button>
                <Button
                  variant="outlined"
                  color="error"
                  startIcon={<Stop />}
                  onClick={handleStopTraining}
                  disabled={loading}
                  size="small"
                >
                  Stop
                </Button>
              </>
            ) : (
              // Show resume button when paused session exists
              currentSessionInfo && !currentSessionInfo.completed_at && (
                <Button
                  variant="outlined"
                  color="success"
                  startIcon={<Resume />}
                  onClick={() => handleResumeSession(currentSessionInfo.id)}
                  disabled={loading}
                  size="small"
                >
                  Resume Training
                </Button>
              )
            )}
          </Box>
        )}
      </CardContent>

      {/* Training Sessions Dialog */}
      <Dialog 
        open={showSessionsDialog} 
        onClose={() => setShowSessionsDialog(false)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>Training Sessions</DialogTitle>
        <DialogContent>
          <List>
            {trainingSessions.map((session, index) => (
              <React.Fragment key={session.id}>
                <ListItem>
                  <ListItemText
                    primary={
                      <Box display="flex" alignItems="center" gap={1}>
                        <Typography variant="subtitle2">{session.session_name}</Typography>
                        <Chip
                          size="small"
                          label={session.is_training ? 'Active' : session.completed_at ? 'Completed' : 'Paused'}
                          color={session.is_training ? 'success' : session.completed_at ? 'default' : 'warning'}
                          icon={session.is_training ? <PlayArrow /> : session.completed_at ? <CheckCircle /> : <Pause />}
                        />
                      </Box>
                    }
                    secondary={
                      <Box>
                        <Typography variant="caption" display="block">
                          Started: {new Date(session.started_at).toLocaleString()}
                        </Typography>
                        {session.completed_at && (
                          <Typography variant="caption" display="block">
                            Completed: {new Date(session.completed_at).toLocaleString()}
                          </Typography>
                        )}
                        {session.progress_percentage && (
                          <Typography variant="caption" display="block">
                            Progress: {Math.round(session.progress_percentage)}%
                          </Typography>
                        )}
                      </Box>
                    }
                  />
                  {!session.is_training && !session.completed_at && (
                    <Button
                      variant="outlined"
                      size="small"
                      startIcon={<Resume />}
                      onClick={() => {
                        handleResumeSession(session.id);
                        setShowSessionsDialog(false);
                      }}
                      disabled={loading}
                    >
                      Resume
                    </Button>
                  )}
                </ListItem>
                {index < trainingSessions.length - 1 && <Divider />}
              </React.Fragment>
            ))}
            {trainingSessions.length === 0 && (
              <ListItem>
                <ListItemText primary="No training sessions found" />
              </ListItem>
            )}
          </List>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setShowSessionsDialog(false)}>Close</Button>
        </DialogActions>
      </Dialog>

      {/* Training Logs Dialog */}
      <Dialog 
        open={showLogsDialog} 
        onClose={() => setShowLogsDialog(false)}
        maxWidth="lg"
        fullWidth
      >
        <DialogTitle>Training Logs</DialogTitle>
        <DialogContent>
          <Box sx={{ maxHeight: 400, overflow: 'auto' }}>
            {trainingLogs.length > 0 ? (
              <List dense>
                {trainingLogs.map((log, index) => (
                  <ListItem key={index}>
                    <ListItemText
                      primary={
                        <Box display="flex" alignItems="center" gap={1}>
                          <Chip
                            size="small"
                            label={log.level}
                            color={log.level === 'ERROR' ? 'error' : log.level === 'WARNING' ? 'warning' : 'default'}
                          />
                          <Typography variant="caption" color="text.secondary">
                            {new Date(log.timestamp).toLocaleTimeString()}
                          </Typography>
                        </Box>
                      }
                      secondary={log.message}
                    />
                  </ListItem>
                ))}
              </List>
            ) : (
              <Typography color="text.secondary" textAlign="center" py={2}>
                No logs available
              </Typography>
            )}
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setShowLogsDialog(false)}>Close</Button>
        </DialogActions>
      </Dialog>
    </Card>
  );
};

export default TrainingStatusComponent;