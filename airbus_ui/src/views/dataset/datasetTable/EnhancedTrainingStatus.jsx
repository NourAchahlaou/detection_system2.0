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
  Alert,
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
  Error,
  Schedule,
  Computer,
} from '@mui/icons-material';
import { datasetService } from '../datasetService';

const TrainingStatusComponent = ({ onTrainingStateChange }) => {
  const [trainingStatus, setTrainingStatus] = useState(null);
  const [trainingSessions, setTrainingSessions] = useState([]);
  const [resumableSessions, setResumableSessions] = useState([]);
  // Add state to persist session info even when training is paused
  const [persistedSessionInfo, setPersistedSessionInfo] = useState(null);
  const [showSessionsDialog, setShowSessionsDialog] = useState(false);
  const [showLogsDialog, setShowLogsDialog] = useState(false);
  const [trainingLogs, setTrainingLogs] = useState([]);
  const [loading, setLoading] = useState(false);
  const [expanded, setExpanded] = useState(false);

  // Fetch training sessions
  const fetchTrainingSessions = async () => {
    try {
      const response = await datasetService.getTrainingSessions({ limit: 10 });
      setTrainingSessions(response.data?.sessions || []);
    } catch (error) {
      console.error('Error fetching training sessions:', error);
    }
  };

  // Fetch resumable sessions
  const fetchResumableSessions = async () => {
    try {
      const response = await datasetService.getResumableSessions();
      setResumableSessions(response.data?.resumable_sessions || []);
    } catch (error) {
      console.error('Error fetching resumable sessions:', error);
    }
  };

  // Manual refresh function for buttons
  const refreshTrainingData = async () => {
    try {
      const [statusResponse, sessionsResponse, resumableResponse] = await Promise.all([
        datasetService.getTrainingStatus(true),
        datasetService.getTrainingSessions({ limit: 10 }),
        datasetService.getResumableSessions()
      ]);

      const newStatus = statusResponse.data;
      const sessions = sessionsResponse.data?.sessions || [];
      const resumable = resumableResponse.data?.resumable_sessions || [];

      setTrainingStatus(newStatus);
      setTrainingSessions(sessions);
      setResumableSessions(resumable);

      if (newStatus?.session_info) {
        setPersistedSessionInfo(newStatus.session_info);
      }

      if (onTrainingStateChange) {
        const currentPersistedInfo = newStatus?.session_info || persistedSessionInfo;
        const hasActiveSession = newStatus?.is_training || 
          (currentPersistedInfo && !currentPersistedInfo.completed_at && resumable.length > 0);
        
        onTrainingStateChange(hasActiveSession);
      }
    } catch (error) {
      console.error('Error refreshing training data:', error);
    }
  };

  // Fetch training logs
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
      
      // Clear persisted session info when explicitly stopping
      setPersistedSessionInfo(null);
      
      // Refresh all data
      await refreshTrainingData();
      
      // Notify parent that training is no longer active
      if (onTrainingStateChange) {
        console.log('Notifying parent after stop - no active sessions');
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

  // Resume training session
  const handleResumeSession = async (sessionId) => {
    setLoading(true);
    try {
      // If no specific sessionId provided, try to resume the most recent resumable session
      if (!sessionId && resumableSessions.length > 0) {
        sessionId = resumableSessions[0].id;
      }
      
      await datasetService.resumeTrainingSession(sessionId);
      await refreshTrainingData();
    } catch (error) {
      console.error('Error resuming training session:', error);
    } finally {
      setLoading(false);
    }
  };

  // Get status info with improved logic
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
    } else if (resumableSessions.length > 0) {
      return { 
        color: 'info', 
        text: 'Resumable Session Available', 
        icon: <Resume fontSize="small" />
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

  // Auto-refresh training status
  useEffect(() => {
    let isMounted = true;

    const fetchAllData = async () => {
      try {
        // Fetch all data in sequence to avoid race conditions
        const [statusResponse, sessionsResponse, resumableResponse] = await Promise.all([
          datasetService.getTrainingStatus(true),
          datasetService.getTrainingSessions({ limit: 10 }),
          datasetService.getResumableSessions()
        ]);

        if (!isMounted) return;

        const newStatus = statusResponse.data;
        const sessions = sessionsResponse.data?.sessions || [];
        const resumable = resumableResponse.data?.resumable_sessions || [];

        // Update all states
        setTrainingStatus(newStatus);
        setTrainingSessions(sessions);
        setResumableSessions(resumable);

        // Handle session persistence
        if (newStatus?.session_info) {
          setPersistedSessionInfo(newStatus.session_info);
        }

        // Notify parent component with current data
        if (onTrainingStateChange) {
          const currentPersistedInfo = newStatus?.session_info || persistedSessionInfo;
          
          const hasActiveSession = newStatus?.is_training || 
            (currentPersistedInfo && !currentPersistedInfo.completed_at && resumable.length > 0);
          
          console.log('Notifying parent - hasActiveSession:', hasActiveSession, {
            isTraining: newStatus?.is_training,
            hasPersistedInfo: !!currentPersistedInfo,
            resumableSessions: resumable.length
          });
          
          onTrainingStateChange(hasActiveSession);
        }
      } catch (error) {
        console.error('Error fetching training data:', error);
      }
    };

    // Initial fetch
    fetchAllData();

    const interval = setInterval(fetchAllData, 5000);

    return () => {
      isMounted = false;
      clearInterval(interval);
    };
  }, [persistedSessionInfo, onTrainingStateChange]);

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
                onClick={() => setShowSessionsDialog(true)} 
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

        {/* Use currentSessionInfo instead of trainingStatus.session_info */}
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
        {(trainingStatus?.is_training || currentSessionInfo || resumableSessions.length > 0) && (
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
              // Show resume button when paused or when resumable sessions exist
              (currentSessionInfo || resumableSessions.length > 0) && (
                <Button
                  variant="outlined"
                  color="success"
                  startIcon={<Resume />}
                  onClick={() => handleResumeSession()}
                  disabled={loading}
                  size="small"
                >
                  Resume Training
                </Button>
              )
            )}
          </Box>
        )}

        {/* Resumable Sessions Alert */}
        {resumableSessions.length > 0 && !trainingStatus?.is_training && (
          <Alert 
            severity="info" 
            sx={{ mt: 2 }}
            action={
              <Button 
                color="inherit" 
                size="small" 
                onClick={() => setShowSessionsDialog(true)}
              >
                View All
              </Button>
            }
          >
            {resumableSessions.length} training session(s) can be resumed
          </Alert>
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
          {/* Resumable Sessions Section */}
          {resumableSessions.length > 0 && (
            <>
              <Typography variant="h6" gutterBottom color="primary">
                Resumable Sessions
              </Typography>
              <List>
                {resumableSessions.map((session, index) => (
                  <React.Fragment key={session.id}>
                    <ListItem>
                      <ListItemText
                        primary={
                          <Box display="flex" alignItems="center" gap={1}>
                            <Typography variant="subtitle2">{session.session_name}</Typography>
                            <Chip
                              size="small"
                              label="Resumable"
                              color="warning"
                              icon={<Pause />}
                            />
                          </Box>
                        }
                        secondary={
                          <Box>
                            <Typography variant="caption" display="block">
                              Started: {new Date(session.started_at).toLocaleString()}
                            </Typography>
                            <Typography variant="caption" display="block">
                              Progress: {Math.round(session.progress_percentage || 0)}%
                            </Typography>
                          </Box>
                        }
                      />
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
                    </ListItem>
                    {index < resumableSessions.length - 1 && <Divider />}
                  </React.Fragment>
                ))}
              </List>
              <Divider sx={{ my: 2 }} />
            </>
          )}

          {/* All Sessions Section */}
          <Typography variant="h6" gutterBottom>
            Recent Sessions
          </Typography>
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