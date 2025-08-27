import React, { useState, useEffect, useCallback } from 'react';
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
  const [datasets, setDatasets] = useState(propDatasets || data || []);
  const [statistics, setStatistics] = useState(null);
  const [availableGroups, setAvailableGroups] = useState([]);
  const [loading, setLoading] = useState(false);
  const [showFilters, setShowFilters] = useState(false);
  const [error, setError] = useState(null);
  
  // TrainingStatusComponent visibility state
  const [showTrainingStatus, setShowTrainingStatus] = useState(false);
  
  // Training state - removed local training state since it's now passed from parent
  const [trainingProgress, setTrainingProgress] = useState(0);
  const [trainingPieces, setTrainingPieces] = useState([]);
  
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

  // Update local datasets when prop changes
  useEffect(() => {
    if (propDatasets && propDatasets.length > 0) {
      setDatasets(propDatasets);
    }
  }, [propDatasets]);

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

  // Show/hide TrainingStatusComponent based on training state
  useEffect(() => {
    setShowTrainingStatus(trainingInProgress);
  }, [trainingInProgress]);

  // Fetch additional data (statistics, groups)
  const fetchAdditionalData = useCallback(async () => {
    if (propDatasets && propDatasets.length > 0) {
      try {
        const [statsResponse, groupsResponse] = await Promise.all([
          datasetService.getDatasetStatistics(),
          datasetService.getAvailableGroups()
        ]);
        
        setStatistics(statsResponse.overview || statsResponse);
        setAvailableGroups(groupsResponse || []);
      } catch (error) {
        console.error("Error fetching additional data:", error);
      }
      return;
    }

    // Legacy behavior - fetch all data if no parent datasets
    setLoading(true);
    setError(null);
    try {
      const params = {
        page: page + 1,
        page_size: pageSize,
        ...Object.fromEntries(
          Object.entries(filters).filter(([_, value]) => value !== '')
        )
      };
      
      const promises = [
        datasetService.getAllDatasetsWithFilters(params),
        datasetService.getDatasetStatistics(),
        datasetService.getAvailableGroups()
      ];
      
      const results = await Promise.all(promises);
      
      setDatasets(results[0].data || []);
      setStatistics(results[1].overview || results[1]);
      setAvailableGroups(results[2] || []);
      
    } catch (error) {
      console.error("Error fetching data:", error);
      setError("Failed to fetch data. Please try again.");
      showNotification("Failed to fetch data", "error");
    } finally {
      setLoading(false);
    }
  }, [propDatasets, page, pageSize, filters]);

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

  // Enhanced wrapper for handleBatchTrain that shows TrainingStatusComponent
  const handleBatchTrain = useCallback(async (pieceLabels, sequential = false) => {
    console.log('=== DatasetComponent.handleBatchTrain WRAPPER ===');
    console.log('Received pieceLabels:', pieceLabels);
    console.log('Parent onBatchTrain:', typeof onBatchTrain);
    
    if (onBatchTrain && typeof onBatchTrain === 'function') {
      console.log('Forwarding to parent onBatchTrain...');
      
      // Show TrainingStatusComponent immediately when starting training
      setShowTrainingStatus(true);
      
      const result = await onBatchTrain(pieceLabels, sequential);
      console.log('Parent returned:', result);
      
      // If training failed, hide the TrainingStatusComponent
      if (result && !result.success) {
        setShowTrainingStatus(false);
      }
      
      return result;
    }

    console.log('No parent onBatchTrain, using legacy fallback');
    // Legacy fallback behavior
    try {
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
      
      if (onTrainingCheck) {
        setTimeout(async () => {
          await onTrainingCheck();
        }, 1000);
      }
      
      return { success: false, error: error.message };
    }
  }, [onBatchTrain, datasets, onTrainingStart, onTrainingCheck, showNotification]);

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
          setDatasets(prevDatasets => 
            prevDatasets.filter(piece => piece.label !== pieceLabel)
          );
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
          
          setDatasets(prevDatasets => 
            prevDatasets.filter(piece => !actionTarget.includes(piece.id))
          );
        }
        
        showNotification(`Successfully deleted ${actionTarget.length} pieces`, "success");
        
        // Clear selection after bulk delete
        if (onSelectAll) {
          onSelectAll();
        }
      }
      
      // Refresh data
      await fetchAdditionalData();
      
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
  const handleTrainingStateChange = useCallback((isTraining) => {
    setShowTrainingStatus(isTraining);
  }, []);

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
        onRefresh={fetchAdditionalData}
        onBulkDelete={handleBulkDelete}
      />

      {/* Error Alert */}
      {error && (
        <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}

      <StatisticsPanel statistics={statistics} />
      
      {/* Show TrainingStatusComponent when training is active */}
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
}