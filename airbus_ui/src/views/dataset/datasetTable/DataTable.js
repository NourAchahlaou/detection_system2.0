import React, { useState, useMemo, useCallback } from 'react';
import {
  TableRow,
  TableBody,
  TableCell,
  TableHead,
  Checkbox,
  Box,
  Avatar,
  Typography,
  Chip,
  Button,
  TablePagination,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  LinearProgress,
  Tooltip,
  CircularProgress,
  Menu,
  MenuItem,
} from "@mui/material";
import {
  Delete, 
  Visibility, 
  PhotoLibrary, 
  CheckCircle, 
  RadioButtonUnchecked,
  Image as ImageIcon,
  PlayArrow,
  Stop,
  MoreVert,
  Pause,
  RestartAlt,
  ExpandMore,
  Group as GroupIcon,
} from "@mui/icons-material";
import { ProductTable, ModernCard, StatusChip, ActionButton } from './StyledComponents';

export default function DataTable({ 
  datasets = [],
  selectedDatasets = [],
  selectAll = false,
  onSelectAll = () => {},
  onSelect = () => {},
  onView = () => {},
  onDelete = () => {},
  onTrain = () => {},
  trainingInProgress = false,
  trainingData = null,
  page = 0,
  pageSize = 10,
  totalCount = 0,
  onPageChange = () => {},
  onRowsPerPageChange = () => {},
  formatDate = (date) => date,
  onBatchTrain = () => {},
  onStopTraining = () => {},
  onPauseTraining = () => {},
  onResumeTraining = () => {}
}) {
  const [trainingMenuAnchor, setTrainingMenuAnchor] = useState(null);
  const [currentTrainingGroup, setCurrentTrainingGroup] = useState(null);

  // Group datasets by first 4 characters of label
  const groupedDatasets = useMemo(() => {
    const groups = {};
    
    if (!datasets || !Array.isArray(datasets)) {
      return groups;
    }
    
    datasets.forEach(piece => {
      const groupKey = piece.label?.length >= 4 ? piece.label.substring(0, 4) : piece.label || 'unknown';
      const groupName = `${groupKey}_group`;
      
      if (!groups[groupName]) {
        groups[groupName] = {
          name: groupName,
          prefix: groupKey,
          pieces: [],
          totalImages: 0,
          totalAnnotations: 0,
          annotatedCount: 0,
          trainedCount: 0,
          isFullyAnnotated: false,
          isFullyTrained: false,
          completionRate: { annotation: 0, training: 0 }
        };
      }
      
      groups[groupName].pieces.push(piece);
      groups[groupName].totalImages += piece.nbre_img || 0;
      groups[groupName].totalAnnotations += piece.total_annotations || 0;
      
      if (piece.is_annotated) {
        groups[groupName].annotatedCount++;
      }
      
      if (piece.is_yolo_trained) {
        groups[groupName].trainedCount++;
      }
    });
    
    // Calculate completion rates for each group
    Object.values(groups).forEach(group => {
      group.isFullyAnnotated = group.annotatedCount === group.pieces.length;
      group.isFullyTrained = group.trainedCount === group.pieces.length;
      group.completionRate.annotation = group.pieces.length > 0 ? Math.round((group.annotatedCount / group.pieces.length) * 100) : 0;
      group.completionRate.training = group.pieces.length > 0 ? Math.round((group.trainedCount / group.pieces.length) * 100) : 0;
    });
    
    return groups;
  }, [datasets]);

    const handleTrainingMenuClick = useCallback((event, group) => {
      event.preventDefault();
      event.stopPropagation();
      setTrainingMenuAnchor(event.currentTarget);
      setCurrentTrainingGroup(group);
    }, []);

    const handleTrainingMenuClose = useCallback(() => {
      setTrainingMenuAnchor(null);
      setCurrentTrainingGroup(null);
    }, []);

    // Check if a group is currently being trained
    const isGroupBeingTrained = useCallback((group) => {
      if (!trainingInProgress || !trainingData?.piece_labels) return false;
      return group.pieces.some(piece => 
        trainingData.piece_labels.includes(piece.label)
      );
    }, [trainingInProgress, trainingData]);

    // Get training progress for current session
    const getTrainingProgress = useCallback(() => {
      if (trainingData?.progress_percentage) {
        return Math.round(trainingData.progress_percentage);
      }
      return 0;
    }, [trainingData]);

    // FIXED: Enhanced handleTrainGroup with better debugging and error handling
    // ALSO UPDATE handleTrainGroup in DataTable.js to properly handle the return value:
const handleTrainGroup = useCallback(async (event, group) => {
  console.log('=== DEBUG handleTrainGroup START ===');
  
  if (event) {
    event.preventDefault();
    event.stopPropagation();
  }

  console.log('Step A: Event handling completed');
  console.log('Step B: onBatchTrain type:', typeof onBatchTrain);
  console.log('Step C: onBatchTrain function:', onBatchTrain);

  if (!onBatchTrain) {
    console.error('Step D: onBatchTrain is null/undefined');
    alert('Training function not available');
    return;
  }

  if (typeof onBatchTrain !== 'function') {
    console.error('Step D: onBatchTrain is not a function:', typeof onBatchTrain);
    alert('Invalid training function');
    return;
  }

  console.log('Step E: onBatchTrain validation passed');

  if (!group || !group.pieces) {
    console.error('Step F: Invalid group:', group);
    alert('Invalid group data');
    return;
  }

  const piecesToTrain = group.pieces.filter(piece => 
    piece.is_annotated && !piece.is_yolo_trained
  );
  
  console.log('Step G: Pieces to train count:', piecesToTrain.length);
  
  if (piecesToTrain.length === 0) {
    console.log('Step H: No trainable pieces');
    alert('No pieces ready for training');
    return;
  }

  if (trainingInProgress) {
    console.log('Step I: Training already in progress');
    alert('Training already in progress');
    return;
  }

  console.log('Step J: All validation passed, calling onBatchTrain...');
  try {
    const pieceLabels = piecesToTrain.map(piece => piece.label);
    console.log('About to call onBatchTrain with:', pieceLabels);
    
    // CRITICAL: Make sure you await AND return the result
    const result = await onBatchTrain(pieceLabels);
    console.log('Result from onBatchTrain:', result);
    
    // Handle the result
    if (result && result.success) {
      console.log('Training succeeded');
    } else {
      console.log('Training failed:', result ? result.error : 'No result');
      alert(`Training failed: ${result ? result.error : 'Unknown error'}`);
    }
    
    return result; // Return the result if needed
    
  } catch (error) {
    console.error('Exception in handleTrainGroup:', error);
    alert(`Error: ${error.message}`);
    return { success: false, error: error.message };
  }
}, [onBatchTrain, trainingInProgress]);

  // Group training action button with improved event handling
  const renderGroupTrainingButton = useCallback((group) => {
    const isCurrentlyTraining = isGroupBeingTrained(group);
    const trainableCount = group.pieces.filter(p => p.is_annotated && !p.is_yolo_trained).length;
    
    if (isCurrentlyTraining) {
      return (
        <Tooltip title="Training in progress">
          <Button 
            size="small"
            variant="contained"
            disabled
            startIcon={<CircularProgress size={12} />}
            sx={{ 
              textTransform: "none",
              minWidth: "100px",
              fontSize: "0.75rem",
              bgcolor: "#ff9800",
              color: "white",
              "&:disabled": {
                bgcolor: "rgba(255, 152, 0, 0.6)",
                color: "white"
              }
            }}
          >
            Training
          </Button>
        </Tooltip>
      );
    }

    if (group.isFullyTrained) {
      return (
        <Button 
          size="small"
          variant="outlined"
          disabled
          sx={{ 
            textTransform: "none",
            minWidth: "100px",
            fontSize: "0.75rem",
            opacity: 0.6
          }}
        >
          Trained
        </Button>
      );
    }

    if (trainableCount === 0) {
      return (
        <Tooltip title="No pieces ready for training (must be annotated and not already trained)">
          <Button 
            size="small"
            variant="outlined"
            disabled
            sx={{ 
              textTransform: "none",
              minWidth: "100px",
              fontSize: "0.75rem",
              opacity: 0.5
            }}
          >
            Not Ready
          </Button>
        </Tooltip>
      );
    }

    // If other training is in progress, disable this button
    if (trainingInProgress && !isCurrentlyTraining) {
      return (
        <Tooltip title="Another training session is active">
          <Button 
            size="small"
            variant="outlined"
            disabled
            sx={{ 
              textTransform: "none",
              minWidth: "100px",
              fontSize: "0.75rem",
              opacity: 0.5
            }}
          >
            Train Group
          </Button>
        </Tooltip>
      );
    }

    // FIXED: Active training button with proper event handling
    return (
      <Tooltip title={`Train ${trainableCount} pieces in this group`}>
        <Button 
          onClick={(event) => {
            console.log('🔴 Train Group button clicked!');
            handleTrainGroup(event, group);
          }}
          onMouseDown={(event) => {
            // Prevent accordion expansion on mouse down
            event.stopPropagation();
          }}
          size="small"
          variant="outlined"
          startIcon={<PlayArrow fontSize="small" />}
          sx={{ 
            textTransform: "none",
            minWidth: "100px",
            fontSize: "0.75rem",
            color: "#667eea",
            borderColor: "#667eea",
            "&:hover": {
              bgcolor: "rgba(102, 126, 234, 0.04)",
              borderColor: "#667eea"
            },
            // Add higher z-index to ensure button is clickable
            zIndex: 2,
            position: 'relative'
          }}
        >
          Train Group
        </Button>
      </Tooltip>
    );
  }, [isGroupBeingTrained, trainingInProgress, handleTrainGroup]);

  // FIXED: GroupHeader with improved button positioning and event handling
  const GroupHeader = useCallback(({ group }) => (
    <Box sx={{ 
      display: 'flex', 
      alignItems: 'center', 
      justifyContent: 'space-between', 
      width: '100%',
      py: 1
    }}>
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
        <Avatar sx={{ 
          bgcolor: isGroupBeingTrained(group) 
            ? "rgba(255, 152, 0, 0.1)" 
            : 'rgba(102, 126, 234, 0.1)', 
          color: isGroupBeingTrained(group) ? "#ff9800" : '#667eea',
          width: 32,
          height: 32
        }}>
          {isGroupBeingTrained(group) ? (
            <CircularProgress size={20} sx={{ color: "#ff9800" }} />
          ) : (
            <GroupIcon />
          )}
        </Avatar>
        <Box>
          <Typography variant="h6" sx={{ fontWeight: 600, color: '#333' }}>
            {group.name}
            {isGroupBeingTrained(group) && (
              <Chip 
                label="TRAINING" 
                size="small" 
                sx={{ 
                  ml: 1, 
                  bgcolor: "#ff9800", 
                  color: "white", 
                  fontSize: "0.65rem",
                  height: 20
                }} 
              />
            )}
          </Typography>
          <Typography variant="caption" color="text.secondary">
            {group.pieces.length} pieces • {group.totalImages} images • {group.totalAnnotations} annotations
          </Typography>
        </Box>
      </Box>
      
      {/* FIXED: Button container with proper event handling */}
      <Box 
        sx={{ 
          display: 'flex', 
          alignItems: 'center', 
          gap: 2,
          // Prevent accordion expansion when clicking in this area
          '& > *': {
            position: 'relative',
            zIndex: 2
          }
        }}
        onClick={(e) => e.stopPropagation()} // Prevent accordion toggle
        onMouseDown={(e) => e.stopPropagation()} // Prevent accordion toggle
      >
        <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', minWidth: 100 }}>
          <Typography variant="caption" color="text.secondary">Annotation</Typography>
          <LinearProgress 
            variant="determinate" 
            value={group.completionRate.annotation} 
            sx={{ 
              width: 80, 
              height: 6, 
              borderRadius: 3,
              backgroundColor: 'rgba(102, 126, 234, 0.1)',
              '& .MuiLinearProgress-bar': {
                backgroundColor: group.completionRate.annotation === 100 ? '#4caf50' : '#667eea'
              }
            }} 
          />
          <Typography variant="caption" color="text.secondary">
            {group.completionRate.annotation}%
          </Typography>
        </Box>
        
        <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', minWidth: 100 }}>
          <Typography variant="caption" color="text.secondary">Training</Typography>
          <LinearProgress 
            variant="determinate" 
            value={group.completionRate.training} 
            sx={{ 
              width: 80, 
              height: 6, 
              borderRadius: 3,
              backgroundColor: 'rgba(102, 126, 234, 0.1)',
              '& .MuiLinearProgress-bar': {
                backgroundColor: group.completionRate.training === 100 ? '#4caf50' : '#ff9800'
              }
            }} 
          />
          <Typography variant="caption" color="text.secondary">
            {group.completionRate.training}%
          </Typography>
        </Box>
        
        {renderGroupTrainingButton(group)}
        
        <Button
          size="small"
          endIcon={<MoreVert />}
          onClick={(e) => {
            e.stopPropagation();
            handleTrainingMenuClick(e, group);
          }}
          disabled={trainingInProgress}
          sx={{ 
            fontSize: "0.75rem", 
            minWidth: "60px",
            zIndex: 2,
            position: 'relative'
          }}
        >
          More
        </Button>
      </Box>
    </Box>
  ), [isGroupBeingTrained, renderGroupTrainingButton, handleTrainingMenuClick, trainingInProgress]);

  // Add safety check for datasets
  if (!datasets || !Array.isArray(datasets) || datasets.length === 0) {
    return (
      <ModernCard elevation={0}>
        <Box sx={{ p: 4, textAlign: 'center' }}>
          <Typography variant="h6" color="text.secondary">
            No datasets available
          </Typography>
        </Box>
      </ModernCard>
    );
  }

  return (
    <ModernCard elevation={0}>
      {/* Global Training Progress Bar */}
      {trainingInProgress && trainingData && (
        <Box sx={{ mb: 2, p: 2, bgcolor: "rgba(102, 126, 234, 0.04)", borderRadius: 2 }}>
          <Box display="flex" justifyContent="space-between" alignItems="center" mb={1}>
            <Typography variant="body2" fontWeight="600" color="#667eea">
              Training Session: {trainingData.session_name || 'Active Training'}
            </Typography>
            <Box display="flex" gap={1}>
              {onPauseTraining && typeof onPauseTraining === 'function' && (
                <Button 
                  size="small" 
                  startIcon={<Pause />} 
                  onClick={onPauseTraining}
                  sx={{ minWidth: "80px", fontSize: "0.75rem" }}
                >
                  Pause
                </Button>
              )}
              {onStopTraining && typeof onStopTraining === 'function' && (
                <Button 
                  size="small" 
                  startIcon={<Stop />} 
                  onClick={onStopTraining}
                  color="error"
                  sx={{ minWidth: "80px", fontSize: "0.75rem" }}
                >
                  Stop
                </Button>
              )}
            </Box>
          </Box>
          
          <LinearProgress 
            variant="determinate" 
            value={getTrainingProgress()} 
            sx={{ 
              height: 8, 
              borderRadius: 4,
              bgcolor: "rgba(102, 126, 234, 0.1)",
              "& .MuiLinearProgress-bar": {
                bgcolor: "#667eea"
              }
            }} 
          />
          
          <Box display="flex" justifyContent="space-between" mt={1}>
            <Typography variant="caption" color="text.secondary">
              Pieces: {trainingData.piece_labels?.join(', ') || 'N/A'}
            </Typography>
            <Typography variant="caption" color="text.secondary">
              Progress: {getTrainingProgress()}% (Epoch {trainingData.current_epoch || 0}/{trainingData.epochs || 30})
            </Typography>
          </Box>
        </Box>
      )}

      {/* Group Overview Header */}
      <Box sx={{ mb: 2, p: 2, bgcolor: 'rgba(102, 126, 234, 0.05)', borderRadius: 2 }}>
        <Typography variant="h6" sx={{ color: '#667eea', fontWeight: 600, mb: 1 }}>
          Group Overview
        </Typography>
        <Typography variant="body2" color="text.secondary">
          Pieces are automatically grouped by their label prefix (first 4 characters). Train entire groups efficiently.
        </Typography>
      </Box>

      {/* Grouped Datasets */}
      {Object.entries(groupedDatasets).map(([groupKey, group]) => (
        <Accordion 
          key={groupKey} 
          sx={{ 
            mb: 1, 
            boxShadow: 1,
            ...(isGroupBeingTrained(group) && {
              bgcolor: "rgba(102, 126, 234, 0.02)",
              borderLeft: "4px solid #667eea"
            })
          }}
        >
          <AccordionSummary 
            expandIcon={<ExpandMore />}
            sx={{
              // Ensure the expand icon works but buttons don't interfere
              '& .MuiAccordionSummary-content': {
                margin: '12px 0',
              },
              '& .MuiAccordionSummary-expandIconWrapper': {
                zIndex: 1
              }
            }}
          >
            <GroupHeader group={group} />
          </AccordionSummary>
          <AccordionDetails sx={{ p: 0 }}>
            <ProductTable>
              <TableHead>
                <TableRow sx={{ bgcolor: 'rgba(102, 126, 234, 0.02)' }}>
                  <TableCell padding="checkbox">
                    <Checkbox
                      checked={group.pieces.every(piece => selectedDatasets.includes(piece.id))}
                      indeterminate={group.pieces.some(piece => selectedDatasets.includes(piece.id)) && 
                                   !group.pieces.every(piece => selectedDatasets.includes(piece.id))}
                      onChange={() => {
                        const allSelected = group.pieces.every(piece => selectedDatasets.includes(piece.id));
                        group.pieces.forEach(piece => {
                          if (allSelected) {
                            if (selectedDatasets.includes(piece.id)) {
                              onSelect(piece.id);
                            }
                          } else {
                            if (!selectedDatasets.includes(piece.id)) {
                              onSelect(piece.id);
                            }
                          }
                        });
                      }}
                      disabled={isGroupBeingTrained(group)}
                      sx={{ color: "#667eea", '&.Mui-checked': { color: "#667eea" } }}
                    />
                  </TableCell>
                  <TableCell>Piece Details</TableCell>
                  <TableCell align="center">Images</TableCell>
                  <TableCell align="center">Annotations</TableCell>
                  <TableCell align="center">Annotation Status</TableCell>
                  <TableCell align="center">Training Status</TableCell>
                  <TableCell align="center">Created</TableCell>
                  <TableCell align="center">Actions</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {group.pieces.map((piece) => {
                  const isPieceBeingTrained = trainingInProgress && 
                    trainingData?.piece_labels?.includes(piece.label);
                  
                  return (
                    <TableRow 
                      key={piece.id} 
                      hover
                      sx={{
                        ...(isPieceBeingTrained && {
                          bgcolor: "rgba(255, 152, 0, 0.02)"
                        })
                      }}
                    >
                      <TableCell padding="checkbox">
                        <Checkbox
                          checked={selectedDatasets.includes(piece.id)}
                          onChange={() => onSelect(piece.id)}
                          disabled={isPieceBeingTrained}
                          sx={{ color: "#667eea", '&.Mui-checked': { color: "#667eea" } }}
                        />
                      </TableCell>
                      
                      <TableCell>
                        <Box display="flex" alignItems="center">
                          <Avatar 
                            sx={{ 
                              width: 40, 
                              height: 40, 
                              mr: 2, 
                              borderRadius: "8px",
                              bgcolor: isPieceBeingTrained 
                                ? "rgba(255, 152, 0, 0.1)" 
                                : "rgba(102, 126, 234, 0.1)",
                              color: isPieceBeingTrained ? "#ff9800" : "#667eea"
                            }}
                          >
                            {isPieceBeingTrained ? (
                              <CircularProgress size={20} sx={{ color: "#ff9800" }} />
                            ) : (
                              <PhotoLibrary />
                            )}
                          </Avatar>
                          <Box>
                            <Typography variant="body2" fontWeight="600" color="#333">
                              {piece.label}
                              {isPieceBeingTrained && (
                                <Chip 
                                  label="TRAINING" 
                                  size="small" 
                                  sx={{ 
                                    ml: 1, 
                                    bgcolor: "#ff9800", 
                                    color: "white", 
                                    fontSize: "0.65rem",
                                    height: 20
                                  }} 
                                />
                              )}
                            </Typography>
                            <Typography variant="caption" color="text.secondary">
                              ID: {piece.class_data_id}
                            </Typography>
                          </Box>
                        </Box>
                      </TableCell>
                      
                      <TableCell align="center">
                        <Box display="flex" alignItems="center" justifyContent="center" gap={0.5}>
                          <ImageIcon fontSize="small" color="action" />
                          <Typography variant="body2" fontWeight="600">
                            {piece.nbre_img}
                          </Typography>
                        </Box>
                      </TableCell>
                      
                      <TableCell align="center">
                        <Typography variant="body2" color="#667eea" fontWeight="600">
                          {piece.total_annotations}
                        </Typography>
                      </TableCell>
                      
                      <TableCell align="center">
                        <StatusChip 
                          variant={piece.is_annotated ? "completed" : "pending"}
                          icon={piece.is_annotated ? <CheckCircle /> : <RadioButtonUnchecked />}
                          label={piece.is_annotated ? "Annotated" : "Pending"}
                          size="small"
                        />
                      </TableCell>
                      
                      <TableCell align="center">
                        {isPieceBeingTrained ? (
                          <Tooltip title={`Training in progress (${getTrainingProgress()}%)`}>
                            <Box display="flex" alignItems="center" justifyContent="center" gap={1}>
                              <CircularProgress size={16} sx={{ color: "#ff9800" }} />
                              <Typography variant="caption" color="#ff9800" fontWeight="600">
                                Training
                              </Typography>
                            </Box>
                          </Tooltip>
                        ) : (
                          <StatusChip 
                            variant={piece.is_yolo_trained ? "trained" : "pending"}
                            icon={piece.is_yolo_trained ? <CheckCircle /> : <RadioButtonUnchecked />}
                            label={piece.is_yolo_trained ? "Trained" : "Not Trained"}
                            size="small"
                          />
                        )}
                      </TableCell>
                      
                      <TableCell align="center">
                        <Typography variant="caption" color="text.secondary">
                          {formatDate(piece.created_at)}
                        </Typography>
                      </TableCell>
                      
                      <TableCell align="center">
                        <Box display="flex" justifyContent="center" gap={0.5}>
                          <ActionButton variant="view" onClick={() => onView(piece)}>
                            <Visibility fontSize="small" />
                          </ActionButton>
                          <ActionButton 
                            variant="delete" 
                            onClick={() => onDelete(piece)}
                            disabled={isPieceBeingTrained}
                          >
                            <Delete fontSize="small" />
                          </ActionButton>
                        </Box>
                      </TableCell>
                    </TableRow>
                  );
                })}
              </TableBody>
            </ProductTable>
          </AccordionDetails>
        </Accordion>
      ))}
      
      <TablePagination
        component="div"
        count={totalCount}
        page={page}
        onPageChange={onPageChange}
        rowsPerPage={pageSize}
        onRowsPerPageChange={onRowsPerPageChange}
        rowsPerPageOptions={[5, 10, 25, 50]}
        sx={{
          borderTop: "1px solid rgba(102, 126, 234, 0.1)",
          "& .MuiTablePagination-selectLabel, & .MuiTablePagination-displayedRows": {
            color: "#667eea",
            fontWeight: "500"
          }
        }}
      />

      {/* Training Options Menu */}
      <Menu
        anchorEl={trainingMenuAnchor}
        open={Boolean(trainingMenuAnchor)}
        onClose={handleTrainingMenuClose}
      >
        {currentTrainingGroup && (
          <>
            <MenuItem 
              onClick={(e) => {
                e.stopPropagation();
                handleTrainGroup(e, currentTrainingGroup);
                handleTrainingMenuClose();
              }} 
              disabled={trainingInProgress || currentTrainingGroup.isFullyTrained}
            >
              <PlayArrow sx={{ mr: 1 }} fontSize="small" />
              Train This Group
            </MenuItem>
            <MenuItem 
              onClick={(e) => {
                e.stopPropagation();
                if (typeof onBatchTrain === 'function') {
                  const allNonTrained = currentTrainingGroup.pieces
                    .filter(p => !p.is_yolo_trained)
                    .map(p => p.label);
                  onBatchTrain(allNonTrained);
                  handleTrainingMenuClose();
                }
              }} 
              disabled={trainingInProgress || currentTrainingGroup.isFullyTrained}
            >
              <RestartAlt sx={{ mr: 1 }} fontSize="small" />
              Force Train All in Group
            </MenuItem>
          </>
        )}
      </Menu>
    </ModernCard>
  );
}