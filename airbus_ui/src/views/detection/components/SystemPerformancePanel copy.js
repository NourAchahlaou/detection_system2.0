// SystemPerformancePanel.jsx - Enhanced with Collapsible Panel Feature
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
  CircularProgress,
  Slide,
  Fade
} from '@mui/material';
import {
  ExpandMore,
  ExpandLess,
  Speed,
  Assessment,
  Settings,
  HealthAndSafety,
  Tune,
  ChevronLeft,
  ChevronRight
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

// Toggle Button Component
const PanelToggleButton = ({ isOpen, onClick, isBasicMode, isDetectionRunning }) => {
  return (
    <Box
      sx={{
        position: 'fixed',
        right: isOpen ? 320 : 0, // Adjust based on panel width
        top: '50%',
        transform: 'translateY(-50%)',
        zIndex: 1300,
        transition: 'right 0.3s ease-in-out'
      }}
    >
      <IconButton
        onClick={onClick}
        sx={{
          bgcolor: 'primary.main',
          color: 'white',
          width: 32,
          height: 48,
          borderRadius: '8px 0 0 8px',
          '&:hover': {
            bgcolor: 'primary.dark',
          },
          position: 'relative',
          boxShadow: 2,
          // Add notification dot for basic mode detection
          '&::after': isBasicMode && isDetectionRunning ? {
            content: '""',
            position: 'absolute',
            top: 4,
            right: 4,
            width: 8,
            height: 8,
            bgcolor: 'warning.main',
            borderRadius: '50%',
            border: '1px solid white'
          } : {}
        }}
      >
        {isOpen ? <ChevronRight fontSize="small" /> : <ChevronLeft fontSize="small" />}
      </IconButton>
    </Box>
  );
};

// Updated System Performance Panel Component with Collapsible Feature
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
  DetectionStates,
  // New props for panel state
  isPanelOpen = true,
  onPanelToggle,
  isDetectionRunning = false
}) => {
  const [internalPanelOpen, setInternalPanelOpen] = useState(true);
  
  // Use external state if provided, otherwise use internal state
  const panelOpen = onPanelToggle ? isPanelOpen : internalPanelOpen;
  const togglePanel = onPanelToggle || (() => setInternalPanelOpen(!internalPanelOpen));

  // Helper function to get state 
  // 
  
    console.log('🔍 Manual Mode Controls Debug Info:', {
    detectionState,
    isReady: detectionState === DetectionStates.READY,
    autoModeEnabled,
    shouldShow: detectionState === DetectionStates.READY && autoModeEnabled,
    DetectionStates
  });
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
    <>
      {/* Toggle Button */}
      <PanelToggleButton 
        isOpen={panelOpen} 
        onClick={togglePanel}
        isBasicMode={isBasicMode}
        isDetectionRunning={isDetectionRunning}
      />

      {/* Sliding Panel */}
      <Box
        sx={{
          position: 'fixed',
          right: panelOpen ? 0 : -320,
          top: 0,
          width: 320,
          height: '100vh',
          zIndex: 1200,
          transition: 'right 0.3s ease-in-out',
          bgcolor: 'background.default',
          borderLeft: panelOpen ? '1px solid' : 'none',
          borderColor: 'divider',
          boxShadow: panelOpen ? 3 : 0,
          overflowY: 'auto'
        }}
      >
        <Card sx={{ height: '100%', borderRadius: 0, boxShadow: 'none' }}>
          <CardContent sx={{ py: 2 }}>
            {/* Panel Header with Close Button */}
            <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
              <Typography variant="h6">
                System Performance
              </Typography>
              <IconButton size="small" onClick={togglePanel}>
                <ChevronRight />
              </IconButton>
            </Box>
            
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
              {detectionState === DetectionStates.READY && autoModeEnabled && (
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
      </Box>

      {/* Backdrop for mobile */}
      {panelOpen && (
        <Box
          sx={{
            position: 'fixed',
            top: 0,
            left: 0,
            width: '100vw',
            height: '100vh',
            bgcolor: 'rgba(0, 0, 0, 0.3)',
            zIndex: 1100,
            display: { xs: 'block', md: 'none' }
          }}
          onClick={togglePanel}
        />
      )}
    </>
  );
};

export default SystemPerformancePanel;