import React, { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import {
  CircularProgress,
  Backdrop,
  Alert,
  Snackbar,
} from "@mui/material";

// Import your real service
import { datasetService } from '../datasetService';

// Import components
import { Container } from './StyledComponents';
import HeaderActions from './HeaderActions';
import StatisticsPanel from './StatisticsPanel';
import FiltersPanel from './FiltersPanel';
import DataTable from './DataTable';
import ConfirmationDialog from './ConfirmationDialog';
import TrainingStatusComponent from './EnhancedTrainingStatus';

export default function DatasetComponenet({ 
  // Props from AppDatabasesetup
  datasets: propDatasets = [], 
  selectedDatasets = [],
  selectAll = false,
  onSelectAll = () => {},
  onSelect = () => {},
  onView = () => {},
  onDelete = () => {}, // Now expects piece labels array instead of piece object
  onBulkDelete = () => {},
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
  onResumeTraining = () => {},
  // Legacy props for compatibility
  data, 
  onTrainingStart, 
  sidebarOpen, 
  setSidebarOpen,
  onTrainingCheck
}) {
  
  // State management - Use parent datasets if available, otherwise local
  const [rawDatasets] = useState(propDatasets || data || []);
  const [statistics, setStatistics] = useState(null);
  const [availableGroups, setAvailableGroups] = useState([]);
  const [loading, setLoading] = useState(false);
  const [showFilters, setShowFilters] = useState(false);
  const [error, setError] = useState(null);
  
  // TrainingStatusComponent visibility state - Modified logic for persistence
  const [showTrainingStatus, setShowTrainingStatus] = useState(false);
  const [hasActiveOrPausedSession, setHasActiveOrPausedSession] = useState(false);
  
  // Training state - removed local training state since it's now passed from parent
  const [trainingProgress, setTrainingProgress] = useState(0);
  const [trainingPieces, setTrainingPieces] = useState([]);
  
  // FIXED: Add ref to track if additional data has been fetched to prevent multiple calls
  const additionalDataFetched = useRef(false);
  
  // Filter state
  const [filters, setFilters] = useState({
    search: '',
    status_filter: '',
    training_filter: '',
    group_filter: '',
    sort_by: 'created_at',
    sort_order: 'desc',
    date_from: '',
    date_to: '',
    min_images: '',
    max_images: ''
  });

  // Dialog state
  const [confirmationOpen, setConfirmationOpen] = useState(false);
  const [actionType, setActionType] = useState("");
  const [actionTarget, setActionTarget] = useState(null);

  // Notification state
  const [notification, setNotification] = useState({
    open: false,
    message: '',
    severity: 'success'
  });

  // FIXED: Use useMemo to filter datasets instead of setting state in useEffect
  const datasets = useMemo(() => {
    let filteredData = propDatasets || rawDatasets || [];
    
    // Apply search filter
    if (filters.search) {
      const searchLower = filters.search.toLowerCase();
      filteredData = filteredData.filter(item => 
        item.label?.toLowerCase().includes(searchLower) ||
        item.class_data_id?.toString().includes(searchLower)
      );
    }
    
    // Apply status filter
    if (filters.status_filter) {
      if (filters.status_filter === 'annotated') {
        filteredData = filteredData.filter(item => item.is_annotated);
      } else if (filters.status_filter === 'pending') {
        filteredData = filteredData.filter(item => !item.is_annotated);
      }
    }
    
    // Apply training filter
    if (filters.training_filter) {
      if (filters.training_filter === 'trained') {
        filteredData = filteredData.filter(item => item.is_yolo_trained);
      } else if (filters.training_filter === 'not_trained') {
        filteredData = filteredData.filter(item => !item.is_yolo_trained);
      }
    }
    
    // Apply group filter (first 4 characters of label)
    if (filters.group_filter) {
      filteredData = filteredData.filter(item => 
        item.label?.substring(0, 4) === filters.group_filter
      );
    }
    
    // Apply date filters
    if (filters.date_from) {
      const fromDate = new Date(filters.date_from);
      filteredData = filteredData.filter(item => 
        new Date(item.created_at) >= fromDate
      );
    }
    
    if (filters.date_to) {
      const toDate = new Date(filters.date_to);
      filteredData = filteredData.filter(item => 
        new Date(item.created_at) <= toDate
      );
    }
    
    // Apply image count filters
    if (filters.min_images) {
      const minImages = parseInt(filters.min_images);
      filteredData = filteredData.filter(item => 
        (item.nbre_img || 0) >= minImages
      );
    }
    
    if (filters.max_images) {
      const maxImages = parseInt(filters.max_images);
      filteredData = filteredData.filter(item => 
        (item.nbre_img || 0) <= maxImages
      );
    }
    
    // Apply sorting
    if (filters.sort_by) {
      filteredData = [...filteredData].sort((a, b) => {
        let aVal = a[filters.sort_by];
        let bVal = b[filters.sort_by];
        
        // Handle different data types
        if (filters.sort_by === 'created_at') {
          aVal = new Date(aVal).getTime();
          bVal = new Date(bVal).getTime();
        } else if (typeof aVal === 'string') {
          aVal = aVal.toLowerCase();
          bVal = bVal?.toLowerCase();
        }
        
        if (filters.sort_order === 'desc') {
          return bVal > aVal ? 1 : bVal < aVal ? -1 : 0;
        } else {
          return aVal > bVal ? 1 : aVal < bVal ? -1 : 0;
        }
      });
    }
    
    return filteredData;
  }, [propDatasets, rawDatasets, filters]);

  // Sync training pieces with training data from parent
  useEffect(() => {
    if (trainingData && trainingData.piece_labels) {
      setTrainingPieces(trainingData.piece_labels);
      setTrainingProgress(trainingData.progress || 0);
    } else {
      setTrainingPieces([]);
      setTrainingProgress(0);
    }
  }, [trainingData]);

  // UPDATED: Show/hide TrainingStatusComponent based on training state AND session data
  useEffect(() => {
    // Show TrainingStatusComponent if:
    // 1. Training is currently in progress, OR
    // 2. There's training data (session info) indicating a paused/resumable session, OR
    // 3. Component itself indicates there's an active/paused session
    const shouldShowTrainingStatus = trainingInProgress || 
      (trainingData && trainingData.session_info && !trainingData.completed_at) ||
      hasActiveOrPausedSession;
    
    setShowTrainingStatus(shouldShowTrainingStatus);
  }, [trainingInProgress, trainingData, hasActiveOrPausedSession]);

  // FIXED: Separate function for fetching additional data - only fetch once and cache result
  const fetchAdditionalData = useCallback(async () => {
    if (additionalDataFetched.current) {
      console.log('Additional data already fetched, skipping...');
      return;
    }
    
    try {
      console.log('Fetching additional data (statistics and groups)...');
      const [statsResponse, groupsResponse] = await Promise.all([
        datasetService.getDatasetStatistics(),
        datasetService.getAvailableGroups()
      ]);
      
      setStatistics(statsResponse.overview || statsResponse);
      setAvailableGroups(groupsResponse || []);
      additionalDataFetched.current = true;
      console.log('Additional data fetched successfully');
    } catch (error) {
      console.error("Error fetching additional data:", error);
      setError("Failed to fetch additional data. Please try again.");
    }
  }, []);

  // FIXED: Only fetch additional data once on mount
  useEffect(() => {
    fetchAdditionalData();
  }, [fetchAdditionalData]);

  // Notification handler
  const showNotification = (message, severity = 'success') => {
    setNotification({
      open: true,
      message,
      severity
    });
  };

  const hideNotification = useCallback(() => {
    setNotification({
      open: false,
      message: '',
      severity: 'success'
    });
  }, []);

  // Training handlers - forward to parent if available
  const handleTrain = async (piece) => {
    if (onTrain && typeof onTrain === 'function') {
      // Use parent's training handler
      onTrain(piece);
      return;
    }

    // Legacy fallback behavior
    try {
      const trainingInfo = {
        status: 'training',
        piece_labels: [piece.label],
        current_epoch: 1,
        total_epochs: 25,
        progress: 4,
        batch_size: 4,
        image_size: 640,
        device: 'cpu',
        total_images: piece.nbre_img,
        augmented_images: piece.nbre_img * 50,
        validation_images: Math.floor(piece.nbre_img * 0.2),
        losses: {
          box_loss: 0.0,
          cls_loss: 0.0,
          dfl_loss: 0.0,
        },
        metrics: {
          instances: piece.total_annotations,
          lr: 0.002,
          momentum: 0.9,
        },
        logs: [
          { level: 'INFO', message: `Starting training process for piece: ${piece.label}`, timestamp: new Date().toISOString() },
          { level: 'INFO', message: `Found ${piece.nbre_img} images for piece: ${piece.label}`, timestamp: new Date().toISOString() },
          { level: 'INFO', message: `Total annotations: ${piece.total_annotations}`, timestamp: new Date().toISOString() },
        ]
      };

      if (onTrainingStart) onTrainingStart(trainingInfo);
      
      await datasetService.trainPieceModel(piece.label);
      
      if (onTrainingCheck) {
        setTimeout(async () => {
          await onTrainingCheck();
        }, 3000);
      }
      
    } catch (error) {
      showNotification(`Failed to start training for ${piece.label}`, "error");
      
      if (onTrainingCheck) {
        setTimeout(async () => {
          await onTrainingCheck();
        }, 1000);
      }
    }
  };

  // FIXED: Add debouncing to prevent multiple rapid calls
  const [isTrainingStarting, setIsTrainingStarting] = useState(false);

  // Enhanced wrapper for handleBatchTrain that shows TrainingStatusComponent
  const handleBatchTrain = useCallback(async (pieceLabels, sequential = false) => {
    // FIXED: Prevent multiple simultaneous calls
    if (isTrainingStarting) {
      console.log('Training already starting, ignoring additional calls');
      return { success: false, error: 'Training already starting' };
    }

    console.log('=== DatasetComponent.handleBatchTrain WRAPPER ===');
    console.log('Received pieceLabels:', pieceLabels);
    console.log('Parent onBatchTrain:', typeof onBatchTrain);
    
    setIsTrainingStarting(true);
    
    try {
      if (onBatchTrain && typeof onBatchTrain === 'function') {
        console.log('Forwarding to parent onBatchTrain...');
        
        // Show TrainingStatusComponent immediately when starting training
        setShowTrainingStatus(true);
        setHasActiveOrPausedSession(true);
        
        const result = await onBatchTrain(pieceLabels, sequential);
        console.log('Parent returned:', result);
        
        // If training failed, hide the TrainingStatusComponent
        if (result && !result.success) {
          setShowTrainingStatus(false);
          setHasActiveOrPausedSession(false);
        }
        
        return result;
      }

      console.log('No parent onBatchTrain, using legacy fallback');
      // Legacy fallback behavior
      const nonTrainedPieces = datasets
        .filter(piece => !piece.is_yolo_trained)
        .map(piece => piece.label);
      
      if (nonTrainedPieces.length === 0) {
        showNotification("No pieces available for training", "warning");
        return { success: false, error: "No pieces available" };
      }
      
      const totalImages = datasets
        .filter(piece => !piece.is_yolo_trained)
        .reduce((sum, piece) => sum + piece.nbre_img, 0);
      
      const totalAnnotations = datasets
        .filter(piece => !piece.is_yolo_trained)
        .reduce((sum, piece) => sum + piece.total_annotations, 0);
      
      // Show TrainingStatusComponent
      setShowTrainingStatus(true);
      setHasActiveOrPausedSession(true);
      
      const trainingInfo = {
        status: 'training',
        piece_labels: nonTrainedPieces,
        current_epoch: 1,
        total_epochs: 25,
        progress: 4,
        batch_size: 4,
        image_size: 640,
        device: 'cpu',
        total_images: totalImages,
        augmented_images: totalImages * 50,
        validation_images: Math.floor(totalImages * 0.2),
        losses: {
          box_loss: 0.0,
          cls_loss: 0.0,
          dfl_loss: 0.0,
        },
        metrics: {
          instances: totalAnnotations,
          lr: 0.002,
          momentum: 0.9,
        },
        logs: [
          { level: 'INFO', message: `Starting training process for ${nonTrainedPieces.length} pieces`, timestamp: new Date().toISOString() },
          { level: 'INFO', message: `Total images: ${totalImages}`, timestamp: new Date().toISOString() },
          { level: 'INFO', message: `Total annotations: ${totalAnnotations}`, timestamp: new Date().toISOString() },
        ]
      };

      if (onTrainingStart) onTrainingStart(trainingInfo);
      
      await datasetService.trainAllPieces();
      
      if (onTrainingCheck) {
        setTimeout(async () => {
          await onTrainingCheck();
        }, 3000);
      }
      
      return { success: true };
      
    } catch (error) {
      showNotification("Failed to start training for all pieces", "error");
      setShowTrainingStatus(false);
      setHasActiveOrPausedSession(false);
      
      if (onTrainingCheck) {
        setTimeout(async () => {
          await onTrainingCheck();
        }, 1000);
      }
      
      return { success: false, error: error.message };
    } finally {
      // FIXED: Always reset the flag
      setTimeout(() => setIsTrainingStarting(false), 1000);
    }
  }, [onBatchTrain, datasets, onTrainingStart, onTrainingCheck, showNotification, isTrainingStarting]);

  const handleTrainAll = async () => {
    const nonTrainedPieces = datasets
      .filter(piece => !piece.is_yolo_trained)
      .map(piece => piece.label);
    
    if (nonTrainedPieces.length === 0) {
      showNotification("No pieces available for training", "warning");
      return;
    }
    
    await handleBatchTrain(nonTrainedPieces);
  };

  const handleStopTraining = async () => {
    if (onStopTraining && typeof onStopTraining === 'function') {
      onStopTraining();
      // Hide TrainingStatusComponent when stopping
      setShowTrainingStatus(false);
      setHasActiveOrPausedSession(false);
      return;
    }

    // Legacy fallback
    try {
      await datasetService.stopTraining();
      
      if (onTrainingCheck) {
        setTimeout(async () => {
          await onTrainingCheck();
        }, 1000);
      }
      
      setTrainingProgress(0);
      setTrainingPieces([]);
      setShowTrainingStatus(false);
      setHasActiveOrPausedSession(false);
      showNotification("Training stopped successfully", "info");
    } catch (error) {
      showNotification("Failed to stop training", "error");
    }
  };

  // Filter handlers
  const handleFilterChange = (field, value) => {
    setFilters(prev => ({ ...prev, [field]: value }));
  };

  const handleClearFilters = () => {
    setFilters({
      search: '',
      status_filter: '',
      training_filter: '',
      group_filter: '',
      sort_by: 'created_at',
      sort_order: 'desc',
      date_from: '',
      date_to: '',
      min_images: '',
      max_images: ''
    });
  };

  // Action handlers
  const handleView = (piece) => {
    if (onView && typeof onView === 'function') {
      onView(piece);
    } else {
      console.log("Viewing:", piece);
      showNotification(`Viewing details for ${piece.label}`, "info");
    }
  };

  // UPDATED: handleDelete - Now handles single piece deletion via bulk delete
  const handleDelete = (piece) => {
    console.log("Delete button clicked for piece:", piece);
    setActionType("delete");
    setActionTarget(piece); // FIXED: Store the entire piece object, not just the label
    setConfirmationOpen(true);
  };

  const handleBulkDelete = () => {
    setActionType("bulkDelete");
    setActionTarget(selectedDatasets);
    setConfirmationOpen(true);
  };

  // UPDATED: handleConfirmationClose - Modified to use bulk delete for single pieces
  const handleConfirmationClose = async (confirm) => {
    console.log("=== DELETION CONFIRMATION ===");
    console.log("Confirmed:", confirm);
    console.log("Action type:", actionType);
    console.log("Action target:", actionTarget);
    
    setConfirmationOpen(false);
    
    if (confirm) {
      try {
        setLoading(true);
        
        if (actionType === "delete" && actionTarget) {
          // FIXED: Handle single piece deletion properly
          console.log("Single piece deletion for:", actionTarget);
          
          // FIXED: Check if actionTarget is a piece object or just a string
          let pieceLabel;
          if (typeof actionTarget === 'string') {
            pieceLabel = actionTarget;
          } else if (actionTarget && actionTarget.label) {
            pieceLabel = actionTarget.label;
          } else {
            console.error("Invalid action target:", actionTarget);
            showNotification("Invalid piece for deletion", "error");
            return;
          }
          
          if (onDelete && typeof onDelete === 'function') {
            // Pass the piece label
            await onDelete(pieceLabel); // FIXED: Pass single label, not array
          } else {
            // Legacy fallback
            await datasetService.deleteBatchOfPieces([pieceLabel]);
          }
          showNotification(`Successfully deleted ${pieceLabel}`, "success");
          
        } else if (actionType === "bulkDelete" && actionTarget && actionTarget.length > 0) {
          // Bulk deletion remains the same
          console.log("Bulk deletion for piece IDs:", actionTarget);
          
          const piecesToDelete = datasets.filter(piece => actionTarget.includes(piece.id));
          const pieceLabels = piecesToDelete.map(piece => piece.label);
          
          console.log("Pieces to delete:", piecesToDelete);
          console.log("Piece labels:", pieceLabels);
          
          if (onBulkDelete && typeof onBulkDelete === 'function') {
            console.log("Using parent's onBulkDelete function");
            await onBulkDelete(pieceLabels);
          } else {
            console.log("Using datasetService.deleteBatchOfPieces");
            await datasetService.deleteBatchOfPieces(pieceLabels);
          }
          
          showNotification(`Successfully deleted ${actionTarget.length} pieces`, "success");
          
          // Clear selection after bulk delete
          if (onSelectAll) {
            onSelectAll();
          }
        }
        
        // FIXED: Don't refresh additional data after delete operations
        // The parent component should handle refreshing the main dataset
        
      } catch (error) {
        console.error("Delete operation failed:", error);
        showNotification(`Failed to delete: ${error.message}`, "error");
      } finally {
        setLoading(false);
      }
    }
    
    setActionTarget(null);
    setActionType("");
  };

  const localFormatDate = (dateString) => {
    if (formatDate && typeof formatDate === 'function') {
      return formatDate(dateString);
    }
    
    if (!dateString) return '-';
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  // Handler for TrainingStatusComponent state changes
  const handleTrainingStateChange = useCallback((isTrainingOrPaused) => {
    console.log('Training state change from component:', isTrainingOrPaused);
    setHasActiveOrPausedSession(isTrainingOrPaused);
    
    // Don't automatically hide the component - let the useEffect handle visibility
    // based on combined state (training + session + component state)
  }, []);

  // FIXED: Refresh function that only refreshes additional data, not datasets
  const handleRefresh = useCallback(async () => {
    // Reset the flag so additional data can be refetched
    additionalDataFetched.current = false;
    await fetchAdditionalData();
  }, [fetchAdditionalData]);

  return (
    <Container>
      <Backdrop open={loading} sx={{ zIndex: 1000, color: '#fff' }}>
        <CircularProgress color="inherit" />
      </Backdrop>

      <HeaderActions
        trainingInProgress={trainingInProgress}
        sidebarOpen={sidebarOpen}
        onToggleSidebar={() => setSidebarOpen && setSidebarOpen(!sidebarOpen)}
        showFilters={showFilters}
        onToggleFilters={() => setShowFilters(!showFilters)}
        selectedCount={selectedDatasets.length}
        onTrainAll={handleTrainAll}
        onStopTraining={handleStopTraining}
        onRefresh={handleRefresh}
        onBulkDelete={handleBulkDelete}
      />

      {/* Error Alert */}
      {error && (
        <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}

      <StatisticsPanel statistics={statistics} />
      
      {/* Show TrainingStatusComponent when training is active OR when there's a paused session */}
      {showTrainingStatus && (
        <TrainingStatusComponent 
          onTrainingStateChange={handleTrainingStateChange}
        />
      )}
      
      <FiltersPanel
        showFilters={showFilters}
        filters={filters}
        availableGroups={availableGroups}
        onFilterChange={handleFilterChange}
        onClearFilters={handleClearFilters}
      />

      <DataTable
        datasets={datasets}
        selectedDatasets={selectedDatasets}
        selectAll={selectAll}
        page={page}
        pageSize={pageSize}
        totalCount={totalCount || datasets.length}
        trainingInProgress={trainingInProgress}
        trainingData={trainingData}
        onSelectAll={onSelectAll}
        onSelect={onSelect}
        onView={handleView}
        onDelete={handleDelete} // This will trigger confirmation dialog for single pieces
        onTrain={handleTrain}
        onPageChange={onPageChange}
        onRowsPerPageChange={onRowsPerPageChange}
        formatDate={localFormatDate}
        onBatchTrain={handleBatchTrain}
        onStopTraining={onStopTraining}
        onPauseTraining={onPauseTraining}
        onResumeTraining={onResumeTraining}
      />

      <ConfirmationDialog
        open={confirmationOpen}
        actionType={actionType}
        selectedCount={actionType === "bulkDelete" ? actionTarget?.length || 0 : 1}
        targetName={actionType === "delete" ? actionTarget?.label : undefined}
        onClose={handleConfirmationClose}
      />

      {/* Notification Snackbar */}
      <Snackbar 
        open={notification.open} 
        autoHideDuration={6000} 
        onClose={hideNotification}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
      >
        <Alert onClose={hideNotification} severity={notification.severity}>
          {notification.message}
        </Alert>
      </Snackbar>
    </Container>
  );
};