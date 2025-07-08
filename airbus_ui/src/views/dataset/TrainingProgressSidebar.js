import React, { useState, useEffect } from 'react';
import {
  Box,
  Card,
  Typography,
  LinearProgress,
  IconButton,
  Collapse,
  Chip,
  Divider,
  Avatar,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  CircularProgress,
  Button,
  Alert,
  Stack,
  Tooltip,
  Badge
} from '@mui/material';
import {
  Close,
  PlayArrow,
  Stop,
  Memory,
  Timer,
  Dataset,
  ModelTraining,
  CheckCircle,
  Error,
  Warning,
  Info,
  Speed,
  TrendingUp,
  Refresh,
  ExpandMore,
  ExpandLess,
  Computer,
  Storage,
  Analytics
} from '@mui/icons-material';
import { styled, keyframes } from '@mui/material/styles';

// Animations
const pulse = keyframes`
  0% {
    transform: scale(1);
    opacity: 1;
  }
  50% {
    transform: scale(1.05);
    opacity: 0.8;
  }
  100% {
    transform: scale(1);
    opacity: 1;
  }
`;

const slideIn = keyframes`
  from {
    transform: translateX(100%);
    opacity: 0;
  }
  to {
    transform: translateX(0);
    opacity: 1;
  }
`;

// Styled Components
const SidebarContainer = styled(Card)(({ theme }) => ({
  position: 'fixed',
  right: 0,
  top: 0,
  height: '100vh',
  width: '400px',
  zIndex: 1300,
  borderRadius: '16px 0 0 16px',
  boxShadow: '-4px 0 32px rgba(0, 0, 0, 0.15)',
  backgroundColor: '#fff',
  animation: `${slideIn} 0.3s ease-out`,
  display: 'flex',
  flexDirection: 'column',
  overflow: 'hidden',
  border: '2px solid rgba(102, 126, 234, 0.1)',
  [theme.breakpoints.down('md')]: {
    width: '350px',
  },
  [theme.breakpoints.down('sm')]: {
    width: '100%',
    borderRadius: 0,
  },
}));

const Header = styled(Box)(({ theme }) => ({
  backgroundColor: 'rgba(102, 126, 234, 0.05)',
  padding: '20px',
  borderBottom: '2px solid rgba(102, 126, 234, 0.1)',
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'space-between',
  position: 'sticky',
  top: 0,
  zIndex: 1,
}));

const Content = styled(Box)({
  flex: 1,
  overflow: 'auto',
  padding: '0',
});

const StatusIndicator = styled(Box)(({ status }) => ({
  display: 'flex',
  alignItems: 'center',
  gap: '8px',
  padding: '8px 12px',
  borderRadius: '20px',
  backgroundColor: status === 'training' 
    ? 'rgba(102, 126, 234, 0.1)' 
    : status === 'completed'
    ? 'rgba(76, 175, 80, 0.1)'
    : 'rgba(244, 67, 54, 0.1)',
  color: status === 'training' 
    ? '#667eea' 
    : status === 'completed'
    ? '#4caf50'
    : '#f44336',
  fontWeight: '600',
  fontSize: '0.875rem',
  animation: status === 'training' ? `${pulse} 2s infinite` : 'none',
}));

const ProgressCard = styled(Card)(({ theme }) => ({
  margin: '16px',
  padding: '16px',
  borderRadius: '12px',
  backgroundColor: 'rgba(102, 126, 234, 0.02)',
  border: '1px solid rgba(102, 126, 234, 0.1)',
}));

const MetricCard = styled(Card)(({ theme }) => ({
  margin: '8px 16px',
  padding: '12px',
  borderRadius: '8px',
  backgroundColor: '#fff',
  border: '1px solid rgba(102, 126, 234, 0.08)',
  display: 'flex',
  alignItems: 'center',
  gap: '12px',
}));

const LogContainer = styled(Box)(({ theme }) => ({
  margin: '16px',
  backgroundColor: '#f8f9fa',
  borderRadius: '8px',
  border: '1px solid rgba(102, 126, 234, 0.1)',
  maxHeight: '300px',
  overflow: 'auto',
  fontFamily: 'monospace',
}));

const LogItem = styled(Box)(({ level }) => ({
  padding: '6px 12px',
  borderBottom: '1px solid rgba(102, 126, 234, 0.05)',
  fontSize: '0.75rem',
  color: level === 'ERROR' ? '#f44336' : 
        level === 'WARNING' ? '#ff9800' : 
        level === 'INFO' ? '#2196f3' : '#666',
  display: 'flex',
  alignItems: 'center',
  gap: '8px',
  '&:last-child': {
    borderBottom: 'none',
  },
}));

const TrainingProgressSidebar = ({ 
  isOpen, 
  onClose, 
  trainingData,
  onStopTraining,
  onRefresh 
}) => {
  const [expandedSections, setExpandedSections] = useState({
    progress: true,
    metrics: true,
    logs: false,
    system: false,
  });

  const [startTime, setStartTime] = useState(null);
  const [duration, setDuration] = useState(0);

  useEffect(() => {
    if (trainingData?.status === 'training' && !startTime) {
      setStartTime(Date.now());
    }
  }, [trainingData?.status, startTime]);

  useEffect(() => {
    let interval;
    if (trainingData?.status === 'training' && startTime) {
      interval = setInterval(() => {
        setDuration(Math.floor((Date.now() - startTime) / 1000));
      }, 1000);
    }
    return () => clearInterval(interval);
  }, [trainingData?.status, startTime]);

  const formatDuration = (seconds) => {
    const hrs = Math.floor(seconds / 3600);
    const mins = Math.floor((seconds % 3600) / 60);
    const secs = seconds % 60;
    return `${hrs.toString().padStart(2, '0')}:${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  const getStatusIcon = (status) => {
    switch (status) {
      case 'training':
        return <ModelTraining sx={{ animation: `${pulse} 2s infinite` }} />;
      case 'completed':
        return <CheckCircle />;
      case 'error':
        return <Error />;
      case 'stopped':
        return <Stop />;
      default:
        return <PlayArrow />;
    }
  };

  const getLogIcon = (level) => {
    switch (level) {
      case 'ERROR':
        return <Error fontSize="small" />;
      case 'WARNING':
        return <Warning fontSize="small" />;
      case 'INFO':
        return <Info fontSize="small" />;
      default:
        return <Info fontSize="small" />;
    }
  };

  const toggleSection = (section) => {
    setExpandedSections(prev => ({
      ...prev,
      [section]: !prev[section]
    }));
  };

  // Mock data structure based on your logs
  const mockTrainingData = trainingData || {
    status: 'training',
    piece_labels: ['G123.12345.123.12'],
    current_epoch: 1,
    total_epochs: 25,
    progress: 4, // 1/25 * 100
    batch_size: 4,
    image_size: 640,
    device: 'cpu',
    model_path: '/app/shared/models/yolov8m.pt',
    dataset_path: '/app/shared/dataset/dataset_custom/data.yaml',
    total_images: 10,
    augmented_images: 500,
    validation_images: 10,
    losses: {
      box_loss: 0.0,
      cls_loss: 0.0,
      dfl_loss: 0.0,
    },
    metrics: {
      instances: 0,
      lr: 0.002,
      momentum: 0.9,
    },
    logs: [
      { level: 'INFO', message: 'Starting training process for piece labels: [\'G123.12345.123.12\']', timestamp: '2025-07-08 19:36:39' },
      { level: 'INFO', message: 'Found 10 images for piece: G123.12345.123.12', timestamp: '2025-07-08 19:36:39' },
      { level: 'INFO', message: 'Created dataset directories for piece: G123.12345.123.12', timestamp: '2025-07-08 19:36:39' },
      { level: 'WARNING', message: 'Train dataset split ratio is outside the recommended range (75-85%)', timestamp: '2025-07-08 19:36:39' },
      { level: 'INFO', message: 'No GPU detected. Using CPU.', timestamp: '2025-07-08 19:37:37' },
      { level: 'INFO', message: 'Starting fine-tuning for piece: G123.12345.123.12 with 25 epochs', timestamp: '2025-07-08 19:37:38' },
      { level: 'INFO', message: 'Starting epoch 1/25 for piece G123.12345.123.12', timestamp: '2025-07-08 19:37:38' },
    ]
  };

  const currentData = mockTrainingData;

  if (!isOpen) return null;

  return (
    <SidebarContainer>
      <Header>
        <Box display="flex" alignItems="center" gap={2}>
          <Avatar sx={{ bgcolor: '#667eea', width: 32, height: 32 }}>
            <ModelTraining fontSize="small" />
          </Avatar>
          <Box>
            <Typography variant="h6" fontWeight="600" color="#333">
              Training Progress
            </Typography>
            <Typography variant="caption" color="text.secondary">
              {currentData.piece_labels?.join(', ')}
            </Typography>
          </Box>
        </Box>
        <IconButton onClick={onClose} size="small">
          <Close />
        </IconButton>
      </Header>

      <Content>
        {/* Status Section */}
        <Box p={2}>
          <StatusIndicator status={currentData.status}>
            {getStatusIcon(currentData.status)}
            <Typography variant="body2" fontWeight="600">
              {currentData.status === 'training' ? 'Training Active' : 
               currentData.status === 'completed' ? 'Training Completed' : 
               'Training Stopped'}
            </Typography>
          </StatusIndicator>

          {currentData.status === 'training' && (
            <Stack spacing={1} mt={2}>
              <Box display="flex" justifyContent="space-between" alignItems="center">
                <Typography variant="body2" color="text.secondary">
                  Duration: {formatDuration(duration)}
                </Typography>
                <Button
                  size="small"
                  variant="outlined"
                  color="error"
                  onClick={onStopTraining}
                  startIcon={<Stop />}
                >
                  Stop
                </Button>
              </Box>
            </Stack>
          )}
        </Box>

        {/* Progress Section */}
        <ProgressCard>
          <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
            <Typography variant="subtitle2" fontWeight="600" color="#667eea">
              Training Progress
            </Typography>
            <IconButton size="small" onClick={() => toggleSection('progress')}>
              {expandedSections.progress ? <ExpandLess /> : <ExpandMore />}
            </IconButton>
          </Box>
          
          <Collapse in={expandedSections.progress}>
            <Box mb={2}>
              <Box display="flex" justifyContent="space-between" mb={1}>
                <Typography variant="body2">
                  Epoch {currentData.current_epoch}/{currentData.total_epochs}
                </Typography>
                <Typography variant="body2" color="#667eea" fontWeight="600">
                  {currentData.progress}%
                </Typography>
              </Box>
              <LinearProgress
                variant="determinate"
                value={currentData.progress}
                sx={{
                  height: 8,
                  borderRadius: 4,
                  backgroundColor: 'rgba(102, 126, 234, 0.1)',
                  '& .MuiLinearProgress-bar': {
                    backgroundColor: '#667eea',
                    borderRadius: 4,
                  },
                }}
              />
            </Box>

            <Stack spacing={1}>
              <Box display="flex" justify="space-between">
                <Typography variant="caption" color="text.secondary">Images:</Typography>
                <Typography variant="caption">{currentData.total_images}</Typography>
              </Box>
              <Box display="flex" justify="space-between">
                <Typography variant="caption" color="text.secondary">Augmented:</Typography>
                <Typography variant="caption">{currentData.augmented_images}</Typography>
              </Box>
              <Box display="flex" justify="space-between">
                <Typography variant="caption" color="text.secondary">Validation:</Typography>
                <Typography variant="caption">{currentData.validation_images}</Typography>
              </Box>
            </Stack>
          </Collapse>
        </ProgressCard>

        {/* Metrics Section */}
        <ProgressCard>
          <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
            <Typography variant="subtitle2" fontWeight="600" color="#667eea">
              Training Metrics
            </Typography>
            <IconButton size="small" onClick={() => toggleSection('metrics')}>
              {expandedSections.metrics ? <ExpandLess /> : <ExpandMore />}
            </IconButton>
          </Box>
          
          <Collapse in={expandedSections.metrics}>
            <Stack spacing={1}>
              <MetricCard>
                <TrendingUp color="primary" />
                <Box>
                  <Typography variant="body2" fontWeight="600">Box Loss</Typography>
                  <Typography variant="caption" color="text.secondary">
                    {currentData.losses.box_loss.toFixed(4)}
                  </Typography>
                </Box>
              </MetricCard>
              <MetricCard>
                <Analytics color="primary" />
                <Box>
                  <Typography variant="body2" fontWeight="600">Classification Loss</Typography>
                  <Typography variant="caption" color="text.secondary">
                    {currentData.losses.cls_loss.toFixed(4)}
                  </Typography>
                </Box>
              </MetricCard>
              <MetricCard>
                <Speed color="primary" />
                <Box>
                  <Typography variant="body2" fontWeight="600">DFL Loss</Typography>
                  <Typography variant="caption" color="text.secondary">
                    {currentData.losses.dfl_loss.toFixed(4)}
                  </Typography>
                </Box>
              </MetricCard>
            </Stack>
          </Collapse>
        </ProgressCard>

        {/* System Info */}
        <ProgressCard>
          <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
            <Typography variant="subtitle2" fontWeight="600" color="#667eea">
              System Information
            </Typography>
            <IconButton size="small" onClick={() => toggleSection('system')}>
              {expandedSections.system ? <ExpandLess /> : <ExpandMore />}
            </IconButton>
          </Box>
          
          <Collapse in={expandedSections.system}>
            <Stack spacing={1}>
              <Box display="flex" alignItems="center" gap={1}>
                <Computer fontSize="small" color="action" />
                <Typography variant="caption">Device: {currentData.device.toUpperCase()}</Typography>
              </Box>
              <Box display="flex" alignItems="center" gap={1}>
                <Storage fontSize="small" color="action" />
                <Typography variant="caption">Batch Size: {currentData.batch_size}</Typography>
              </Box>
              <Box display="flex" alignItems="center" gap={1}>
                <Dataset fontSize="small" color="action" />
                <Typography variant="caption">Image Size: {currentData.image_size}px</Typography>
              </Box>
              <Box display="flex" alignItems="center" gap={1}>
                <Speed fontSize="small" color="action" />
                <Typography variant="caption">Learning Rate: {currentData.metrics.lr}</Typography>
              </Box>
            </Stack>
          </Collapse>
        </ProgressCard>

        {/* Training Logs */}
        <ProgressCard>
          <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
            <Typography variant="subtitle2" fontWeight="600" color="#667eea">
              Training Logs
            </Typography>
            <Box display="flex" gap={1}>
              <IconButton size="small" onClick={onRefresh}>
                <Refresh />
              </IconButton>
              <IconButton size="small" onClick={() => toggleSection('logs')}>
                {expandedSections.logs ? <ExpandLess /> : <ExpandMore />}
              </IconButton>
            </Box>
          </Box>
          
          <Collapse in={expandedSections.logs}>
            <LogContainer>
              {currentData.logs.slice(-10).map((log, index) => (
                <LogItem key={index} level={log.level}>
                  {getLogIcon(log.level)}
                  <Box flex={1}>
                    <Typography variant="caption" display="block">
                      [{log.timestamp}] {log.message}
                    </Typography>
                  </Box>
                </LogItem>
              ))}
            </LogContainer>
          </Collapse>
        </ProgressCard>
      </Content>
    </SidebarContainer>
  );
};

export default TrainingProgressSidebar;