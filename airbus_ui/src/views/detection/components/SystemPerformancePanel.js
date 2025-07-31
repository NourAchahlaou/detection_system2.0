// System Performance Panel with Collapsible Sections
import React, { useState } from 'react';
import {
  Card,
  CardContent,
  Typography,
  Stack,
  Box,
  Chip,
  Button,
  Divider,
  Collapse,
  IconButton,
  CircularProgress
} from '@mui/material';
import {
  ExpandMore,
  ExpandLess,
  Speed,
  Assessment,
  Settings,
  HealthAndSafety,
  Tune
} from '@mui/icons-material';

// Collapsible Section Component
const CollapsibleSection = ({ 
  title, 
  children, 
  defaultExpanded = false, 
  icon,
  badge,
  badgeColor = "primary"
}) => {
  const [expanded, setExpanded] = useState(defaultExpanded);

  return (
    <Box>
      <Box
        sx={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          cursor: 'pointer',
          py: 1,
          '&:hover': {
            bgcolor: 'action.hover',
            borderRadius: 1
          }
        }}
        onClick={() => setExpanded(!expanded)}
      >
        <Stack direction="row" spacing={1} alignItems="center">
          {icon}
          <Typography variant="subtitle2" color="textSecondary">
            {title}
          </Typography>
          {badge && (
            <Chip
              label={badge}
              size="small"
              color={badgeColor}
              sx={{ height: 18, fontSize: '0.7rem' }}
            />
          )}
        </Stack>
        <IconButton size="small" sx={{ p: 0.5 }}>
          {expanded ? <ExpandLess /> : <ExpandMore />}
        </IconButton>
      </Box>
      <Collapse in={expanded}>
        <Box sx={{ pl: 3, pb: 1 }}>
          {children}
        </Box>
      </Collapse>
    </Box>
  );
};

// Updated System Performance Panel Component
const SystemPerformancePanel = ({
  detectionState,
  systemHealth,
  globalStats,
  detectionOptions,
  healthCheckPerformed,
  autoModeEnabled,
  isBasicMode,
  getHealthCheckAge,
  handleManualHealthCheck,
  handleSwitchToBasicMode,
  handleSwitchToOptimizedMode,
  handleEnableAutoMode,
  DetectionStates
}) => {
  // Helper function to get state info
  const getStateInfo = () => {
    switch (detectionState) {
      case DetectionStates.INITIALIZING:
        return {
          color: 'info',
          message: 'Initializing adaptive detection system...',
          canOperate: false
        };
      case DetectionStates.READY:
        return {
          color: 'success',
          message: 'System ready for detection',
          canOperate: true
        };
      case DetectionStates.RUNNING:
        return {
          color: 'warning',
          message: 'Detection active',
          canOperate: true
        };
      case DetectionStates.SHUTTING_DOWN:
        return {
          color: 'error',
          message: 'System shutting down...',
          canOperate: false
        };
      default:
        return {
          color: 'default',
          message: 'Unknown state',
          canOperate: false
        };
    }
  };

  const stateInfo = getStateInfo();

  return (
    <Card sx={{ height: 'fit-content' }}>
      <CardContent sx={{ py: 2 }}>
        <Typography variant="h6" gutterBottom>
          System Performance
        </Typography>
        
        <Stack spacing={1}>
          {/* Detection State - Always Visible */}
          <CollapsibleSection
            title="Detection State"
            defaultExpanded={true}
            icon={<Speed sx={{ fontSize: 16 }} />}
            badge={detectionState}
            badgeColor={stateInfo.color}
          >
            <Stack spacing={0.5}>
              <Typography variant="caption" color="textSecondary">
                {stateInfo.message}
              </Typography>
              <Typography variant="caption">
                Status: {stateInfo.canOperate ? 'Operational' : 'Not Ready'}
              </Typography>
            </Stack>
          </CollapsibleSection>

          <Divider />

          {/* System Health */}
          <CollapsibleSection
            title="System Health"
            defaultExpanded={false}
            icon={<HealthAndSafety sx={{ fontSize: 16 }} />}
            badge={
              detectionState === DetectionStates.INITIALIZING ? "Initializing" : 
              detectionState === DetectionStates.SHUTTING_DOWN ? "Shutting Down" :
              systemHealth.overall ? "Healthy" : "Issues"
            }
            badgeColor={
              detectionState === DetectionStates.INITIALIZING ? "warning" :
              detectionState === DetectionStates.SHUTTING_DOWN ? "info" :
              systemHealth.overall ? "success" : "error"
            }
          >
            <Stack spacing={1}>
              <Stack direction="row" spacing={1} alignItems="center">
                <Button
                  size="small"
                  variant="outlined"
                  onClick={handleManualHealthCheck}
                  disabled={detectionState !== DetectionStates.READY}
                  sx={{ fontSize: '0.7rem', py: 0.5 }}
                >
                  Check Now
                </Button>
              </Stack>
              <Typography variant="caption" color="textSecondary">
                Last checked: {getHealthCheckAge()}
              </Typography>
              <Stack spacing={0.5}>
                <Typography variant="caption">
                  Streaming: {systemHealth.streaming?.status || 'unknown'}
                </Typography>
                <Typography variant="caption">
                  Detection: {systemHealth.detection?.status || 'unknown'}
                </Typography>
              </Stack>
            </Stack>
          </CollapsibleSection>

          <Divider />

          {/* Health Check Status */}
          <CollapsibleSection
            title="Health Check Status"
            defaultExpanded={false}
            icon={<HealthAndSafety sx={{ fontSize: 16 }} />}
          >
            <Stack spacing={0.5}>
              <Typography variant="caption">
                Initial: {healthCheckPerformed.current.initial ? '✅ Done' : '⏳ Pending'}
              </Typography>
              <Typography variant="caption">
                Post-Shutdown: {healthCheckPerformed.current.postShutdown ? '✅ Done' : '⏳ Pending'}
              </Typography>
            </Stack>
          </CollapsibleSection>

          {/* Manual Mode Controls - Only show when applicable */}
          {detectionState === DetectionStates.RUNNING && !autoModeEnabled && (
            <>
              <Divider />
              <CollapsibleSection
                title="Manual Mode Controls"
                defaultExpanded={false}
                icon={<Tune sx={{ fontSize: 16 }} />}
                badge="Manual"
                badgeColor="warning"
              >
                <Stack spacing={1}>
                  <Button
                    size="small"
                    variant="outlined"
                    onClick={handleSwitchToBasicMode}
                    disabled={isBasicMode}
                    color="warning"
                    fullWidth
                    sx={{ fontSize: '0.75rem', py: 0.5 }}
                  >
                    Switch to Basic
                  </Button>
                  <Button
                    size="small"
                    variant="outlined"
                    onClick={handleSwitchToOptimizedMode}
                    disabled={!isBasicMode}
                    color="success"
                    fullWidth
                    sx={{ fontSize: '0.75rem', py: 0.5 }}
                  >
                    Switch to Optimized
                  </Button>
                  <Button
                    size="small"
                    variant="outlined"
                    onClick={handleEnableAutoMode}
                    color="primary"
                    fullWidth
                    sx={{ fontSize: '0.75rem', py: 0.5 }}
                  >
                    Enable Auto Mode
                  </Button>
                </Stack>
              </CollapsibleSection>
            </>
          )}
        </Stack>
      </CardContent>
    </Card>
  );
};
export default SystemPerformancePanel;