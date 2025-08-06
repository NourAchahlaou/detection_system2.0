// InfoPanel.jsx - Professional, clean information display system
import React, { useState, useEffect } from 'react';
import {
  Box,
  Paper,
  IconButton,
  Typography,
  Chip,
  Collapse,
  Tooltip,
  Stack,
  Divider,
  alpha
} from '@mui/material';
import {
  Inventory,
  Computer,
  Smartphone,
  ExpandMore,
  ExpandLess,
  CheckCircle,
  Warning,
  Schedule,
  Memory,
  Speed,
  Close
} from '@mui/icons-material';

const InfoPanel = ({
  // Lot workflow props
  showLotWorkflowPanel,
  currentLot,
  detectionHistory,
  getPieceLabel,
  onStopLotWorkflow,
  
  // Basic mode lot props
  isBasicMode,
  
  // System mode props
  detectionState,
  systemProfile,
  currentStreamingType,
  isProfileRefreshing,
  onRefreshSystemProfile,
  onRunPerformanceTest,
  
  // Utils
  DetectionStates
}) => {
  // Separate state for each panel
  const [lotWorkflowExpanded, setLotWorkflowExpanded] = useState(false);
  const [lotWorkflowIconOnly, setLotWorkflowIconOnly] = useState(true);
  
  const [basicLotExpanded, setBasicLotExpanded] = useState(false);
  const [basicLotIconOnly, setBasicLotIconOnly] = useState(true);
  
  const [systemModeExpanded, setSystemModeExpanded] = useState(false);
  const [systemModeIconOnly, setSystemModeIconOnly] = useState(true);

// Fixed toggle functions - replace your existing toggle functions with these:

  const toggleLotWorkflowPanel = () => {
    if (lotWorkflowIconOnly) {
      setLotWorkflowIconOnly(false);
      setLotWorkflowExpanded(true); 
      return;
    }
    setLotWorkflowExpanded(!lotWorkflowExpanded);
    setBasicLotExpanded(false)
    setSystemModeExpanded(false)
  };

  const toggleBasicLotPanel = () => {
    if (basicLotIconOnly) {
      setBasicLotIconOnly(false);
      setBasicLotExpanded(true); // Ensure it expands when coming out of icon mode
      return;
    }
    setBasicLotExpanded(!basicLotExpanded);
    setLotWorkflowExpanded(false)
    setSystemModeExpanded(false)
  };

  const toggleSystemModePanel = () => {
    if (systemModeIconOnly) {
      setSystemModeIconOnly(false);
      setSystemModeExpanded(true); // Ensure it expands when coming out of icon mode
      return;
    }
    setSystemModeExpanded(!systemModeExpanded);
    setBasicLotExpanded(false)
    setLotWorkflowExpanded(false)
  };

  // Independent collapse to icon functions
  const collapseLotWorkflowToIcon = () => {
    setLotWorkflowIconOnly(!lotWorkflowIconOnly);
    setLotWorkflowExpanded(false);
  };

  const collapseBasicLotToIcon = () => {
    setBasicLotIconOnly(!basicLotIconOnly);
    setBasicLotExpanded(false);
  };

  const collapseSystemModeToIcon = () => {
    setSystemModeIconOnly(!systemModeIconOnly);
    setSystemModeExpanded(false);
  };

  const getModeInfo = () => {
    if (currentStreamingType === 'basic') {
      return {
        icon: <Smartphone />,
        title: 'BASIC MODE',
        description: 'Lot-Based Detection - Tracked detection with database integration'
      };
    }
    return {
      icon: <Computer />,
      title: 'OPTIMIZED MODE',
      description: 'High-Performance Detection - Advanced streaming with optimizations'
    };
  };

  // Professional color scheme
  const getStatusColor = (isTargetMatch, isComplete = false) => {
    if (isTargetMatch === true || isComplete) return '#1976d2'; // Professional blue
    if (isTargetMatch === false) return '#1976d2'; // Same blue for consistency
    return '#1976d2'; // Default blue
  };

  // Don't render if no relevant information to show
  const hasLotWorkflow = showLotWorkflowPanel && currentLot;
  const hasBasicLot = isBasicMode && currentLot && !showLotWorkflowPanel;
  const hasSystemMode = detectionState === DetectionStates.READY && systemProfile;


  if (!hasLotWorkflow && !hasBasicLot && !hasSystemMode) {
    return null;
  }

  const modeInfo = getModeInfo();

  return (
    <Box sx={{ 
      position: 'fixed',
      top: '30%',
      left: 0,
      zIndex: 1300,
      display: 'flex',
      gap: 1,
      flexDirection: 'column'
      
    }}>
      
      {/* Lot Workflow Panel */}
      {hasLotWorkflow && (
        <Paper 
          elevation={1}
          variant="outlined"
          sx={{ 
            borderRadius: '0 8px 8px 0',
            overflow: 'hidden',
            backgroundColor: 'background.paper',
            border: '1px solid',
            borderColor: 'divider',
            width: lotWorkflowIconOnly ? '40px' : '300px',
            height: lotWorkflowIconOnly ? '53px' : lotWorkflowExpanded ? 'auto' : '53px',
            transition: 'width 0.3s ease, height 0.3s ease'
          }}
        >
          {/* Icon-only view */}
          {lotWorkflowIconOnly ? (
            <Box sx={{ 
              p: 1,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              height: '48px',
              cursor: 'pointer',
              '&:hover': {
                backgroundColor: alpha('#1976d2', 0.04)
              }
            }}
            onClick={collapseLotWorkflowToIcon}
            >
              <Tooltip title={`Working on Lot: ${currentLot.lot_name}`} placement="right" arrow>
                <Inventory sx={{ 
                  color: '#1976d2', 
                  fontSize: '1.25rem'
                }} />
              </Tooltip>
            </Box>
          ) : (
            <>
              {/* Header - Always visible when not icon-only */}
              <Box sx={{ 
                p: 1.5,
                backgroundColor: alpha('#1976d2', 0.04),
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                height: '53px',
                borderBottom: '1px solid',
                borderBottomColor: 'divider'
              }}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <Tooltip title="Collapse to icon" arrow>
                    <Inventory 
                      sx={{ 
                        color: '#1976d2', 
                        cursor: 'pointer',
                        fontSize: '1.25rem',
                        '&:hover': {
                          color: '#1565c0'
                        }
                      }} 
                      onClick={collapseLotWorkflowToIcon}
                    />
                  </Tooltip>
                  <Typography variant="subtitle2" sx={{ 
                    fontWeight: 600, 
                    color: '#1976d2',
                    fontSize: '0.8125rem',
                    letterSpacing: '0.5px'
                  }}>
                    WORKING ON LOT
                  </Typography>
                  <Chip
                    size="small"
                    icon={currentLot.is_target_match === true ? 
                      <CheckCircle sx={{ fontSize: '0.875rem' }} /> : 
                      <Warning sx={{ fontSize: '0.875rem' }} />
                    }
                    label={currentLot.is_target_match === true ? 'Complete' : 
                           currentLot.is_target_match === false ? 'Needs Work' : 'Not Started'}
                    variant="outlined"
                    sx={{
                      borderColor: '#1976d2',
                      color: '#1976d2',
                      '& .MuiChip-icon': {
                        color: '#1976d2'
                      },
                      height: '24px',
                      fontSize: '0.75rem'
                    }}
                  />
                </Box>
                <Box sx={{ display: 'flex', alignItems: 'center' }}>

                  <IconButton 
                    size="small" 
                    onClick={toggleLotWorkflowPanel}
                    sx={{ 
                      color: '#1976d2',
                      '&:hover': {
                        backgroundColor: alpha('#1976d2', 0.08)
                      }
                    }}
                  >
                    {lotWorkflowExpanded ? <ExpandLess fontSize="small" /> : <ExpandMore fontSize="small" />}
                  </IconButton>
                </Box>
              </Box>

              {/* Expandable Content */}
              <Collapse in={lotWorkflowExpanded}>
                <Box sx={{ p: 2, backgroundColor: 'background.paper' }}>
                  <Typography variant="h6" sx={{ 
                    fontWeight: 600, 
                    mb: 2,
                    color: 'text.primary',
                    fontSize: '1rem'
                  }}>
                    {currentLot.lot_name}
                  </Typography>
                  
                  <Stack spacing={1.5}>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                      <Typography variant="caption" color="text.secondary" sx={{ fontWeight: 500 }}>
                        Expected Piece:
                      </Typography>
                      <Typography variant="body2" sx={{ fontWeight: 500, color: 'text.primary' }}>
                        {currentLot.expected_piece_label || getPieceLabel(currentLot.expected_piece_id)}
                      </Typography>
                    </Box>
                    
                    <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                      <Typography variant="caption" color="text.secondary" sx={{ fontWeight: 500 }}>
                        Expected Number:
                      </Typography>
                      <Typography variant="body2" sx={{ fontWeight: 500, color: 'text.primary' }}>
                        {currentLot.expected_piece_number}
                      </Typography>
                    </Box>
                    
                    <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                      <Typography variant="caption" color="text.secondary" sx={{ fontWeight: 500 }}>
                        Status:
                      </Typography>
                      <Typography variant="body2" sx={{ fontWeight: 500, color: '#1976d2' }}>
                        {currentLot.is_target_match === true ? 'Complete' : 
                         currentLot.is_target_match === false ? 'Needs Work' : 'Not Started'}
                      </Typography>
                    </Box>
                    
                    <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                      <Typography variant="caption" color="text.secondary" sx={{ fontWeight: 500 }}>
                        Sessions:
                      </Typography>
                      <Typography variant="body2" sx={{ fontWeight: 500, color: 'text.primary' }}>
                        {detectionHistory.length}
                      </Typography>
                    </Box>
                    
                    {currentLot.completed_at && (
                      <>
                        <Divider sx={{ my: 1 }} />
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                          <Schedule sx={{ fontSize: '0.875rem', color: 'text.secondary' }} />
                          <Typography variant="caption" color="text.secondary">
                            Completed: {new Date(currentLot.completed_at).toLocaleString()}
                          </Typography>
                        </Box>
                      </>
                    )}
                  </Stack>
                </Box>
              </Collapse>
            </>
          )}
        </Paper>
      )}

      {/* Basic Mode Lot Panel */}
      {hasBasicLot && (
        <Paper 
          elevation={1}
          variant="outlined"
          sx={{ 
            borderRadius: '0 8px 8px 0',
            overflow: 'hidden',
            backgroundColor: 'background.paper',
            border: '1px solid',
            borderColor: 'divider',
            width: basicLotIconOnly ? '40px' : '300px',
            height: basicLotIconOnly ? '53px' : basicLotExpanded ? 'auto' : '53px',
            transition: 'width 0.3s ease, height 0.3s ease'
          }}
        >
          {/* Icon-only view */}
          {basicLotIconOnly ? (
            <Box sx={{ 
              p: 1,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              height: '48px',
              cursor: 'pointer',
              '&:hover': {
                backgroundColor: alpha('#1976d2', 0.04)
              }
            }}
            onClick={collapseBasicLotToIcon}
            >
              <Tooltip title={`Active Lot: ${currentLot.lot_name}`} placement="right" arrow>
                <Inventory sx={{ 
                  color: '#1976d2', 
                  fontSize: '1.25rem'
                }} />
              </Tooltip>
            </Box>
          ) : (
            <>
              {/* Header - Always visible when not icon-only */}
              <Box sx={{ 
                p: 1.5,
                backgroundColor: alpha('#1976d2', 0.04),
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                height: '53px',
                borderBottom: '1px solid',
                borderBottomColor: 'divider'
              }}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <Tooltip title="Collapse to icon" arrow>
                    <Inventory 
                      sx={{ 
                        color: '#1976d2', 
                        cursor: 'pointer',
                        fontSize: '1.25rem',
                        '&:hover': {
                          color: '#1565c0'
                        }
                      }} 
                      onClick={collapseBasicLotToIcon}
                    />
                  </Tooltip>
                  <Typography variant="subtitle2" sx={{ 
                    fontWeight: 600, 
                    color: '#1976d2',
                    fontSize: '0.8125rem',
                    letterSpacing: '0.5px'
                  }}>
                    ACTIVE LOT
                  </Typography>
                  <Chip
                    size="small"
                    icon={currentLot.is_target_match ? 
                      <CheckCircle sx={{ fontSize: '0.875rem' }} /> : 
                      <Warning sx={{ fontSize: '0.875rem' }} />
                    }
                    label={currentLot.is_target_match ? 'Complete' : 'Pending'}
                    variant="outlined"
                    sx={{
                      borderColor: '#1976d2',
                      color: '#1976d2',
                      '& .MuiChip-icon': {
                        color: '#1976d2'
                      },
                      height: '24px',
                      fontSize: '0.75rem'
                    }}
                  />
                </Box>
                <Box sx={{ display: 'flex', alignItems: 'center' }}>
                  <IconButton 
                    size="small" 
                    onClick={toggleBasicLotPanel}
                    sx={{ 
                      color: '#1976d2',
                      '&:hover': {
                        backgroundColor: alpha('#1976d2', 0.08)
                      }
                    }}
                  >
                    {basicLotExpanded ? <ExpandLess fontSize="small" /> : <ExpandMore fontSize="small" />}
                  </IconButton>
                </Box>
              </Box>

              {/* Expandable Content */}
              <Collapse in={basicLotExpanded}>
                <Box sx={{ p: 2, backgroundColor: 'background.paper' }}>
                  <Typography variant="h6" sx={{ 
                    fontWeight: 600, 
                    mb: 2,
                    color: 'text.primary',
                    fontSize: '1rem'
                  }}>
                    {currentLot.lot_name}
                  </Typography>
                  
                  <Stack spacing={1.5}>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                      <Typography variant="caption" color="text.secondary" sx={{ fontWeight: 500 }}>
                        Expected:
                      </Typography>
                      <Typography variant="body2" sx={{ fontWeight: 500, color: 'text.primary' }}>
                        {currentLot.expected_piece_label || getPieceLabel(currentLot.expected_piece_id)}
                      </Typography>
                    </Box>
                    
                    <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                      <Typography variant="caption" color="text.secondary" sx={{ fontWeight: 500 }}>
                        Number:
                      </Typography>
                      <Typography variant="body2" sx={{ fontWeight: 500, color: 'text.primary' }}>
                        {currentLot.expected_piece_number}
                      </Typography>
                    </Box>
                    
                    <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                      <Typography variant="caption" color="text.secondary" sx={{ fontWeight: 500 }}>
                        Sessions:
                      </Typography>
                      <Typography variant="body2" sx={{ fontWeight: 500, color: 'text.primary' }}>
                        {currentLot.total_sessions || 0}
                      </Typography>
                    </Box>
                    
                    {currentLot.completed_at && (
                      <>
                        <Divider sx={{ my: 1 }} />
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                          <Schedule sx={{ fontSize: '0.875rem', color: 'text.secondary' }} />
                          <Typography variant="caption" color="text.secondary">
                            Completed: {new Date(currentLot.completed_at).toLocaleString()}
                          </Typography>
                        </Box>
                      </>
                    )}
                  </Stack>
                </Box>
              </Collapse>
            </>
          )}
        </Paper>
      )}

      {/* System Mode Panel */}
      {hasSystemMode && (
        <Paper 
          elevation={1}
          variant="outlined"
          sx={{ 
            borderRadius: '0 8px 8px 0',
            overflow: 'hidden',
            backgroundColor: 'background.paper',
            border: '1px solid',
            borderColor: 'divider',
            width: systemModeIconOnly ? '40px' : '300px',
            height: systemModeIconOnly ? '53px' : systemModeExpanded ? 'auto' : '53px',
            transition: 'width 0.3s ease, height 0.3s ease'
          }}
        >
          {/* Icon-only view */}
          {systemModeIconOnly ? (
            <Box sx={{ 
              p: 1,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              height: '48px',
              cursor: 'pointer',
              '&:hover': {
                backgroundColor: alpha('#1976d2', 0.04)
              }
            }}
            onClick={collapseSystemModeToIcon}
            >
              <Tooltip title={modeInfo.title} placement="right" arrow>
                {React.cloneElement(modeInfo.icon, { 
                  sx: { color: '#1976d2', fontSize: '1.25rem' }
                })}
              </Tooltip>
            </Box>
          ) : (
            <>
              {/* Header - Always visible when not icon-only */}
              <Box sx={{ 
                p: 1.5,
                backgroundColor: alpha('#1976d2', 0.04),
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                height: '53px',
                borderBottom: '1px solid',
                borderBottomColor: 'divider'
              }}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <Tooltip title="Collapse to icon" arrow>
                    {React.cloneElement(modeInfo.icon, { 
                      sx: { 
                        color: '#1976d2', 
                        cursor: 'pointer',
                        fontSize: '1.25rem',
                        '&:hover': {
                          color: '#1565c0'
                        }
                      },
                      onClick: collapseSystemModeToIcon
                    })}
                  </Tooltip>
                  <Typography variant="subtitle2" sx={{ 
                    fontWeight: 600, 
                    color: '#1976d2',
                    fontSize: '0.8125rem',
                    letterSpacing: '0.5px'
                  }}>
                    {modeInfo.title}
                  </Typography>
                </Box>
                <Box sx={{ display: 'flex', alignItems: 'center' }}>
                  <Tooltip title="Refresh System Profile" arrow>
                    <IconButton 
                      size="small" 
                      onClick={onRefreshSystemProfile}
                      disabled={isProfileRefreshing}
                      sx={{ 
                        color: '#1976d2',
                        '&:hover': {
                          backgroundColor: alpha('#1976d2', 0.08)
                        },
                        '&:disabled': {
                          color: 'text.disabled'
                        }
                      }}
                    >
                      <Memory fontSize="small" />
                    </IconButton>
                  </Tooltip>
                  <Tooltip title="Run Performance Test" arrow>
                    <IconButton 
                      size="small" 
                      onClick={onRunPerformanceTest}
                      disabled={detectionState !== DetectionStates.READY}
                      sx={{ 
                        color: '#1976d2',
                        '&:hover': {
                          backgroundColor: alpha('#1976d2', 0.08)
                        },
                        '&:disabled': {
                          color: 'text.disabled'
                        }
                      }}
                    >
                      <Speed fontSize="small" />
                    </IconButton>
                  </Tooltip>
                  <IconButton 
                    size="small" 
                    onClick={toggleSystemModePanel}
                    sx={{ 
                      color: '#1976d2',
                      '&:hover': {
                        backgroundColor: alpha('#1976d2', 0.08)
                      }
                    }}
                  >
                    {systemModeExpanded ? <ExpandLess fontSize="small" /> : <ExpandMore fontSize="small" />}
                  </IconButton>
                </Box>
              </Box>

              {/* Expandable Content */}
              <Collapse in={systemModeExpanded}>
                <Box sx={{ p: 2, backgroundColor: 'background.paper' }}>
                  <Typography variant="body2" sx={{ 
                    mb: 2, 
                    color: 'text.secondary',
                    fontSize: '0.875rem',
                    lineHeight: 1.5
                  }}>
                    {modeInfo.description}
                  </Typography>
                  
                  <Stack spacing={1.5}>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <Typography variant="caption" color="text.secondary" sx={{ fontWeight: 500 }}>
                        Performance:
                      </Typography>
                      <Chip
                        size="small"
                        label={`${systemProfile.performance_score}/100`}
                        variant="outlined"
                        sx={{
                          borderColor: '#1976d2',
                          color: '#1976d2',
                          height: '24px',
                          fontSize: '0.75rem',
                          fontWeight: 500
                        }}
                      />
                    </Box>
                    
                    <Divider />
                    
                    <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                      <Typography variant="caption" color="text.secondary" sx={{ fontWeight: 500 }}>
                        CPU:
                      </Typography>
                      <Typography variant="body2" sx={{ fontWeight: 500, color: 'text.primary' }}>
                        {systemProfile.cpu_cores} cores
                      </Typography>
                    </Box>
                    
                    <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                      <Typography variant="caption" color="text.secondary" sx={{ fontWeight: 500 }}>
                        RAM:
                      </Typography>
                      <Typography variant="body2" sx={{ fontWeight: 500, color: 'text.primary' }}>
                        {systemProfile.available_memory_gb}GB
                      </Typography>
                    </Box>
                    
                    <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                      <Typography variant="caption" color="text.secondary" sx={{ fontWeight: 500 }}>
                        GPU:
                      </Typography>
                      <Typography variant="body2" sx={{ 
                        fontWeight: 500, 
                        color: 'text.primary',
                        maxWidth: '150px',
                        overflow: 'hidden',
                        textOverflow: 'ellipsis',
                        whiteSpace: 'nowrap'
                      }}>
                        {systemProfile.gpu_available ? systemProfile.gpu_name : 'None'}
                      </Typography>
                    </Box>
                  </Stack>
                </Box>
              </Collapse>
            </>
          )}
        </Paper>
      )}
    </Box>
  );
};

export default InfoPanel;