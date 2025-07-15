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

export default function EnhancedDataTable({ 
  data, 
  onTrainingStart, 
  trainingInProgress, 
  sidebarOpen, 
  setSidebarOpen,
  trainingData,
  onTrainingCheck // New prop to check training status
}) {
  
  // State management
  const [datasets, setDatasets] = useState([]);
  const [statistics, setStatistics] = useState(null);
  const [availableGroups, setAvailableGroups] = useState([]);
  const [loading, setLoading] = useState(false);
  const [showFilters, setShowFilters] = useState(false);
  const [error, setError] = useState(null);
  
  // Training state - removed local training state since it's now passed from parent
  const [trainingProgress, setTrainingProgress] = useState(0);
  const [trainingPieces, setTrainingPieces] = useState([]);
  
  // Pagination state
  const [page, setPage] = useState(0);
  const [pageSize, setPageSize] = useState(10);
  const [totalCount, setTotalCount] = useState(0);
  
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
  
  // Selection state
  const [selectedDatasets, setSelectedDatasets] = useState([]);
  const [selectAll, setSelectAll] = useState(false);
  
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

  // Fetch data
  const fetchData = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const params = {
        page: page + 1, // API expects 1-based pagination
        page_size: pageSize,
        ...Object.fromEntries(
          Object.entries(filters).filter(([_, value]) => value !== '')
        )
      };
      
      const promises = [
        datasetService.getAllDatasetsWithFilters(params)
      ];
      
      // Only fetch statistics and groups on initial load
      if (page === 0) {
        promises.push(datasetService.getDatasetStatistics());
        if (availableGroups.length === 0) {
          promises.push(datasetService.getAvailableGroups());
        }
      }
      
      const results = await Promise.all(promises);
      const datasetsResponse = results[0];
      
      setDatasets(datasetsResponse.data || []);
      setTotalCount(datasetsResponse.pagination?.total_count || 0);
      
      if (results[1]) setStatistics(results[1].overview || results[1]);
      if (results[2]) setAvailableGroups(results[2] || []);
      
    } catch (error) {
      console.error("Error fetching data:", error);
      setError("Failed to fetch data. Please try again.");
      showNotification("Failed to fetch data", "error");
    } finally {
      setLoading(false);
    }
  }, [page, pageSize, filters, availableGroups.length]);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  // Notification handler
  const showNotification = (message, severity = 'success') => {
    setNotification({
      open: true,
      message,
      severity
    });
  };

  // Training handlers
  const handleTrain = async (piece) => {
    try {
      // Prepare training data
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

      // Call the parent's training start handler
      onTrainingStart(trainingInfo);
      
      // Start the actual training
      await datasetService.trainPieceModel(piece.label);
      
      // Check training status after starting
      if (onTrainingCheck) {
        setTimeout(async () => {
          await onTrainingCheck();
        }, 3000);
      }
      
    } catch (error) {
      showNotification(`Failed to start training for ${piece.label}`, "error");
      
      // If training failed to start, check status to update UI
      if (onTrainingCheck) {
        setTimeout(async () => {
          await onTrainingCheck();
        }, 1000);
      }
    }
  };

  const handleTrainAll = async () => {
    try {
      const nonTrainedPieces = datasets
        .filter(piece => !piece.is_yolo_trained)
        .map(piece => piece.label);
      
      if (nonTrainedPieces.length === 0) {
        showNotification("No pieces available for training", "warning");
        return;
      }
      
      // Set training data for multiple pieces
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

      // Call the parent's training start handler
      onTrainingStart(trainingInfo);
      
      // Start the actual training
      await datasetService.trainAllPieces();
      
      // Check training status after starting
      if (onTrainingCheck) {
        setTimeout(async () => {
          await onTrainingCheck();
        }, 3000);
      }
      
    } catch (error) {
      showNotification("Failed to start training for all pieces", "error");
      
      // If training failed to start, check status to update UI
      if (onTrainingCheck) {
        setTimeout(async () => {
          await onTrainingCheck();
        }, 1000);
      }
    }
  };

  const handleStopTraining = async () => {
    try {
      await datasetService.stopTraining();
      
      // After stopping, check status to update UI
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
    setPage(0); // Reset to first page when filters change
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
    setPage(0);
  };

  // Pagination handlers
  const handleChangePage = (event, newPage) => {
    setPage(newPage);
  };

  const handleChangeRowsPerPage = (event) => {
    setPageSize(parseInt(event.target.value, 10));
    setPage(0);
  };

  // Selection handlers
  const handleSelectAll = () => {
    if (selectAll) {
      setSelectedDatasets([]);
      setSelectAll(false);
    } else {
      setSelectedDatasets(datasets.map(dataset => dataset.id));
      setSelectAll(true);
    }
  };

  const handleSelect = (id) => {
    setSelectedDatasets(prevSelected => {
      const newSelected = prevSelected.includes(id) 
        ? prevSelected.filter(item => item !== id) 
        : [...prevSelected, id];
      
      // Update selectAll state
      setSelectAll(newSelected.length === datasets.length);
      return newSelected;
    });
  };

  // Action handlers
  const handleView = (piece) => {
    console.log("Viewing:", piece);
    // Navigate to detail view or open modal
    showNotification(`Viewing details for ${piece.label}`, "info");
  };

  const handleDelete = (piece) => {
    setActionType("delete");
    setActionTarget(piece);
    setConfirmationOpen(true);
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
          // For bulk delete, you might need to delete each piece individually
          for (const id of actionTarget) {
            const piece = datasets.find(d => d.id === id);
            if (piece) {
              await datasetService.deletePieceByLabel(piece.label);
            }
          }
          showNotification(`Successfully deleted ${actionTarget.length} pieces`, "success");
          setSelectedDatasets([]);
          setSelectAll(false);
        }
        
        fetchData(); // Refresh data after deletion
      } catch (error) {
        showNotification("Failed to delete. Please try again.", "error");
      } finally {
        setLoading(false);
      }
    }
    
    setActionTarget(null);
    setActionType("");
  };

  const formatDate = (dateString) => {
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
        onToggleSidebar={() => setSidebarOpen(!sidebarOpen)}
        showFilters={showFilters}
        onToggleFilters={() => setShowFilters(!showFilters)}  // ✅ Correct prop name and toggle function
        selectedCount={selectedDatasets.length}
        onTrainAll={handleTrainAll}
        onStopTraining={handleStopTraining}
        onRefresh={fetchData}
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

      <DataTable
        datasets={datasets}
        selectedDatasets={selectedDatasets}
        selectAll={selectAll}
        page={page}
        pageSize={pageSize}
        totalCount={totalCount}
        trainingInProgress={trainingInProgress}
        onSelectAll={handleSelectAll}
        onSelect={handleSelect}
        onView={handleView}
        onDelete={handleDelete}
        onTrain={handleTrain}
        onChangePage={handleChangePage}
        onChangeRowsPerPage={handleChangeRowsPerPage}
        formatDate={formatDate}
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