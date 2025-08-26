import React, { useState, useEffect, useCallback } from 'react';
import {
  CircularProgress,
  Backdrop,
  Alert,
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

export default function DatasetComponenet({ 
  // Props from AppDatabasesetup
  datasets: propDatasets = [], // ✅ FIXED: Receive datasets from parent
  selectedDatasets = [],
  selectAll = false,
  onSelectAll = () => {},
  onSelect = () => {},
  onView = () => {},
  onDelete = () => {},
  onTrain = () => {}, // ✅ FIXED: Single piece training from parent
  trainingInProgress = false,
  trainingData = null,
  page = 0,
  pageSize = 10,
  totalCount = 0,
  onPageChange = () => {},
  onRowsPerPageChange = () => {},
  formatDate = (date) => date,
  onBatchTrain = () => {}, // ✅ FIXED: Batch training from parent
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
  
  // DEBUG: Log the received onBatchTrain function
  console.log('=== DatasetComponenet DEBUG ===');
  console.log('onBatchTrain received:', onBatchTrain);
  console.log('onBatchTrain type:', typeof onBatchTrain);
  console.log('onBatchTrain toString:', onBatchTrain.toString().substring(0, 100));
  
  // State management - Use parent datasets if available, otherwise local
  const [datasets, setDatasets] = useState(propDatasets || data || []);
  const [statistics, setStatistics] = useState(null);
  const [availableGroups, setAvailableGroups] = useState([]);
  const [loading, setLoading] = useState(false);
  const [showFilters, setShowFilters] = useState(false);
  const [error, setError] = useState(null);
  
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

  // ✅ FIXED: Update local datasets when prop changes
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

  // Fetch additional data (statistics, groups) - only if not using parent datasets
  const fetchAdditionalData = useCallback(async () => {
    if (propDatasets && propDatasets.length > 0) {
      // If using parent datasets, only fetch statistics and groups
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

  // ✅ FIXED: Training handlers - forward to parent if available
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

  // ✅ CRITICAL FIX: Create a proper wrapper for handleBatchTrain that forwards to parent
  const handleBatchTrain = useCallback(async (pieceLabels, sequential = false) => {
    console.log('=== DatasetComponenet.handleBatchTrain WRAPPER ===');
    console.log('Received pieceLabels:', pieceLabels);
    console.log('Parent onBatchTrain:', typeof onBatchTrain);
    
    if (onBatchTrain && typeof onBatchTrain === 'function') {
      console.log('Forwarding to parent onBatchTrain...');
      const result = await onBatchTrain(pieceLabels, sequential);
      console.log('Parent returned:', result);
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
      
      if (onTrainingCheck) {
        setTimeout(async () => {
          await onTrainingCheck();
        }, 1000);
      }
      
      return { success: false, error: error.message };
    }
  }, [onBatchTrain, datasets, onTrainingStart, onTrainingCheck, showNotification]);

  const handleTrainAll = async () => {
    // ✅ FIXED: Use the local wrapper which properly forwards to parent
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
    // ✅ FIXED: Use parent's stop handler if available
    if (onStopTraining && typeof onStopTraining === 'function') {
      onStopTraining();
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

  const handleDelete = (piece) => {
    if (onDelete && typeof onDelete === 'function') {
      onDelete(piece);
    } else {
      setActionType("delete");
      setActionTarget(piece);
      setConfirmationOpen(true);
    }
  };

  const handleBulkDelete = () => {
    setActionType("bulkDelete");
    setActionTarget(selectedDatasets);
    setConfirmationOpen(true);
  };

  const handleConfirmationClose = async (confirm) => {
    setConfirmationOpen(false);
    
    if (confirm) {
      try {
        setLoading(true);
        
        if (actionType === "delete" && actionTarget) {
          await datasetService.deletePieceByLabel(actionTarget.label);
          showNotification(`Successfully deleted ${actionTarget.label}`, "success");
        } else if (actionType === "bulkDelete" && actionTarget) {
          for (const id of actionTarget) {
            const piece = datasets.find(d => d.id === id);
            if (piece) {
              await datasetService.deletePieceByLabel(piece.label);
            }
          }
          showNotification(`Successfully deleted ${actionTarget.length} pieces`, "success");
        }
        
        fetchAdditionalData(); // Refresh data after deletion
      } catch (error) {
        showNotification("Failed to delete. Please try again.", "error");
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
      
      <FiltersPanel
        showFilters={showFilters}
        filters={filters}
        availableGroups={availableGroups}
        onFilterChange={handleFilterChange}
        onClearFilters={handleClearFilters}
      />

      {/* ✅ CRITICAL FIX: Pass the LOCAL wrapper function, not the parent function directly */}
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
        onDelete={handleDelete}
        onTrain={handleTrain}
        onPageChange={onPageChange}
        onRowsPerPageChange={onRowsPerPageChange}
        formatDate={localFormatDate}
        onBatchTrain={handleBatchTrain} // ✅ CRITICAL FIX: Pass the LOCAL wrapper
        onStopTraining={onStopTraining} // ✅ FIXED: Forward parent's stop handler
        onPauseTraining={onPauseTraining} // ✅ FIXED: Forward parent's pause handler
        onResumeTraining={onResumeTraining} // ✅ FIXED: Forward parent's resume handler
      />

      <ConfirmationDialog
        open={confirmationOpen}
        actionType={actionType}
        selectedCount={selectedDatasets.length}
        onClose={handleConfirmationClose}
      />
    </Container>
  );
}