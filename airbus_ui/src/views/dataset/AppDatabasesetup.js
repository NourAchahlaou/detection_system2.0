// UPDATED: AppDatabasesetup component with unified delete handling
import { Box, styled, Typography, CircularProgress, Snackbar, Alert } from "@mui/material";
import DatasetComponenet from "./datasetTable/Dataset";
import NoData from "../sessions/NoData";
import TrainingProgressSidebar from "./TrainingProgressSidebar";
import { useState, useEffect, useRef, useCallback } from "react";
import { datasetService } from "./datasetService";
import { useNavigate } from "react-router-dom";

const Container = styled("div")(({ theme }) => ({
  margin: "30px",
  [theme.breakpoints.down("sm")]: { margin: "16px" },
  "& .breadcrumb": {
    marginBottom: "30px",
    [theme.breakpoints.down("sm")]: { marginBottom: "16px" }
  }
}));

const LoadingContainer = styled(Box)({
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  minHeight: "400px",
  flexDirection: "column",
  gap: 2,
  color: "#666",
});

const ErrorContainer = styled(Box)(({ theme }) => ({
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  minHeight: "400px",
  flexDirection: "column",
  gap: 2,
  color: "#f44336",
  textAlign: "center",
  padding: "40px",
  backgroundColor: "rgba(244, 67, 54, 0.04)",
  borderRadius: "16px",
  border: "1px solid rgba(244, 67, 54, 0.1)",
  margin: "20px 0",
}));

export default function AppDatabasesetup() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const navigate = useNavigate();

  // Training sidebar state
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [trainingData, setTrainingData] = useState(null);
  const [trainingInProgress, setTrainingInProgress] = useState(false);
  
  // Enhanced training state
  const [currentSessionId, setCurrentSessionId] = useState(null);
  const [trainingHistory, setTrainingHistory] = useState([]);
  
  // Selection state
  const [selectedDatasets, setSelectedDatasets] = useState([]);
  const [selectAll, setSelectAll] = useState(false);
  
  // Pagination state
  const [page, setPage] = useState(0);
  const [pageSize, setPageSize] = useState(10);
  
  // Notification state
  const [notification, setNotification] = useState({ open: false, message: '', severity: 'info' });
  
  const hasCheckedInitialStatus = useRef(false);
  const trainingStatusInterval = useRef(null);

  // Function to show notifications
  const showNotification = useCallback((message, severity = 'info') => {
    console.log(`🔔 Notification: ${message} (${severity})`);
    setNotification({ open: true, message, severity });
  }, []);

  const hideNotification = useCallback(() => {
    setNotification({ open: false, message: '', severity: 'info' });
  }, []);

  // Enhanced training status check with session management
  const checkTrainingStatus = useCallback(async () => {
    try {
      console.log('🔍 Checking training status...');
      const status = await datasetService.getTrainingStatus(true);
      console.log('Training status response:', status);

      if (status?.data?.is_training && status?.data?.session_info) {
        console.log('✅ Training is active');
        setTrainingInProgress(true);
        setTrainingData(status.data.session_info);
        setCurrentSessionId(status.data.session_info.id);
        setSidebarOpen(true);
        
        if (!trainingStatusInterval.current) {
          startTrainingPolling();
        }
      } else {
        console.log('❌ No active training');
        setTrainingInProgress(false);
        setTrainingData(null);
        setCurrentSessionId(null);
        setSidebarOpen(false);
        
        stopTrainingPolling();
      }
    } catch (error) {
      console.error("❌ Error checking training status:", error);
      setTrainingInProgress(false);
      setTrainingData(null);
      setCurrentSessionId(null);
      setSidebarOpen(false);
      stopTrainingPolling();
    }
  }, []);

  // Start polling for training updates
  const startTrainingPolling = useCallback(() => {
    if (trainingStatusInterval.current) {
      console.log('⚠️ Training polling already active');
      return;
    }
    
    console.log('🔄 Starting training polling...');
    trainingStatusInterval.current = setInterval(async () => {
      try {
        const status = await datasetService.getTrainingStatus(true);
        
        if (status?.data?.is_training && status?.data?.session_info) {
          setTrainingData(status.data.session_info);
        } else {
          console.log('🎉 Training completed!');
          setTrainingInProgress(false);
          setTrainingData(null);
          setCurrentSessionId(null);
          setSidebarOpen(false);
          stopTrainingPolling();
          showNotification('Training session completed!', 'success');
          
          fetchData();
        }
      } catch (error) {
        console.error("❌ Error during training polling:", error);
        stopTrainingPolling();
      }
    }, 3000);
  }, [showNotification]);

  // Stop polling
  const stopTrainingPolling = useCallback(() => {
    if (trainingStatusInterval.current) {
      console.log('⏹️ Stopping training polling...');
      clearInterval(trainingStatusInterval.current);
      trainingStatusInterval.current = null;
    }
  }, []);

  // Fetch training sessions history
  const fetchTrainingHistory = useCallback(async () => {
    try {
      console.log('📜 Fetching training history...');
      const sessions = await datasetService.getTrainingSessions({ limit: 10 });
      setTrainingHistory(sessions.data?.sessions || []);
      console.log('Training history fetched:', sessions.data?.sessions?.length || 0, 'sessions');
    } catch (error) {
      console.error("❌ Error fetching training history:", error);
    }
  }, []);

  // Initial training status check on component mount
  useEffect(() => {
    if (!hasCheckedInitialStatus.current) {
      console.log('🚀 Component mounted, checking initial status...');
      checkTrainingStatus();
      fetchTrainingHistory();
      hasCheckedInitialStatus.current = true;
    }
  }, [checkTrainingStatus, fetchTrainingHistory]);

  // Cleanup polling on component unmount
  useEffect(() => {
    return () => {
      console.log('🧹 Component unmounting, cleaning up...');
      stopTrainingPolling();
    };
  }, [stopTrainingPolling]);

  // Fetch datasets data
  const fetchData = useCallback(async () => {
    console.log('📊 Fetching datasets data...');
    setLoading(true);
    setError(null);
    
    try {
      const pieces = await datasetService.getAllDatasets();
      const dataArray = Array.isArray(pieces) ? pieces : Object.values(pieces || {});
      console.log('📊 Datasets fetched:', dataArray.length, 'pieces');
      setData(dataArray);

      if (!dataArray || dataArray.length === 0) {
        console.log('⚠️ No data available, redirecting to 204');
        navigate("/204");
      }
    } catch (error) {
      console.error("❌ Error fetching data:", error);
      setError(error.response?.data?.detail || "Failed to fetch data");
    } finally {
      setLoading(false);
    }
  }, [navigate]);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  // Enhanced training handlers
  const handleTrainingStart = useCallback(async (piece) => {
    try {
      console.log('🚀 Starting single piece training for:', piece.label);
      showNotification(`Starting training for ${piece.label}...`, 'info');
      
      const result = await datasetService.trainPieceModel([piece.label]);
      console.log('Training start result:', result);
      
      if (result.data) {
        setCurrentSessionId(result.data.session_id);
        setTrainingInProgress(true);
        setSidebarOpen(true);
        startTrainingPolling();
        showNotification(`Training started for ${piece.label}!`, 'success');
      }
    } catch (error) {
      console.error("❌ Failed to start training:", error);
      showNotification(
        error.response?.data?.detail?.message || `Failed to start training for ${piece.label}`, 
        'error'
      );
    }
  }, [showNotification, startTrainingPolling]);

  const handleBatchTrain = useCallback(async (pieceLabels, sequential = false) => {
    console.log('=== DEBUG handleBatchTrain START ===');
    console.log('Input pieceLabels:', pieceLabels);
    
    try {
      if (!pieceLabels || !Array.isArray(pieceLabels) || pieceLabels.length === 0) {
        const result = { success: false, error: 'No pieces provided' };
        showNotification('No pieces provided for training', 'error');
        return result;
      }

      if (trainingInProgress) {
        const result = { success: false, error: 'Training in progress' };
        showNotification('Training is already in progress', 'warning');
        return result;
      }

      showNotification(`Starting training for ${pieceLabels.length} pieces...`, 'info');

      const apiResult = await datasetService.trainMultiplePieces(pieceLabels);
      console.log('API call completed, result:', apiResult);

      if (apiResult && apiResult.data) {
        if (apiResult.data.session_id) {
          setCurrentSessionId(apiResult.data.session_id);
          setTrainingInProgress(true);
          setSidebarOpen(true);
          startTrainingPolling();
        }
        
        showNotification(`Training started successfully!`, 'success');
        return { success: true, data: apiResult.data };
      } else {
        const result = { success: false, error: 'Invalid API response format' };
        showNotification('Invalid response from training service', 'error');
        return result;
      }

    } catch (error) {
      console.log('Exception caught:', error);
      const errorMessage = error.response?.data?.detail || error.message || 'Unknown error occurred';
      showNotification(`Training failed: ${errorMessage}`, 'error');
      return { success: false, error: errorMessage };
    }
  }, [showNotification, startTrainingPolling, trainingInProgress]);

  const handleTrainingStop = useCallback(async () => {
    try {
      console.log('🛑 Stopping training...');
      await datasetService.stopTraining();
      setTrainingInProgress(false);
      setSidebarOpen(false);
      setTrainingData(null);
      setCurrentSessionId(null);
      stopTrainingPolling();
      showNotification('Training stopped successfully', 'success');
      
      fetchData();
    } catch (error) {
      console.error("❌ Failed to stop training:", error);
      showNotification('Failed to stop training', 'error');
    }
  }, [showNotification, stopTrainingPolling, fetchData]);

  const handleTrainingPause = useCallback(async () => {
    try {
      console.log('⏸️ Pausing training...');
      await datasetService.pauseTraining();
      showNotification('Training paused', 'info');
      checkTrainingStatus();
    } catch (error) {
      console.error("❌ Failed to pause training:", error);
      showNotification('Failed to pause training', 'error');
    }
  }, [showNotification, checkTrainingStatus]);

  const handleTrainingResume = useCallback(async (sessionId) => {
    try {
      console.log('▶️ Resuming training for session:', sessionId);
      await datasetService.resumeTrainingSession(sessionId);
      showNotification('Training resumed', 'success');
      startTrainingPolling();
    } catch (error) {
      console.error("❌ Failed to resume training:", error);
      showNotification('Failed to resume training', 'error');
    }
  }, [showNotification, startTrainingPolling]);

  const handleRefreshTraining = useCallback(async () => {
    try {
      console.log('🔄 Refreshing training data...');
      await checkTrainingStatus();
      await fetchTrainingHistory();
      showNotification('Training status refreshed', 'info');
    } catch (error) {
      console.error("❌ Failed to refresh training data:", error);
      showNotification('Failed to refresh training data', 'error');
    }
  }, [checkTrainingStatus, fetchTrainingHistory, showNotification]);

  const handleSidebarClose = useCallback(() => {
    setSidebarOpen(false);
  }, []);

  const handlePageRefresh = useCallback(() => {
    console.log('🔄 Manual page refresh triggered');
    checkTrainingStatus();
    fetchData();
  }, [checkTrainingStatus, fetchData]);

  // DataTable handler functions
  const handleSelectAll = useCallback(() => {
    if (selectAll) {
      setSelectedDatasets([]);
    } else {
      setSelectedDatasets(data ? data.map(item => item.id) : []);
    }
    setSelectAll(!selectAll);
  }, [selectAll, data]);

  const handleSelect = useCallback((id) => {
    setSelectedDatasets(prev => {
      if (prev.includes(id)) {
        return prev.filter(item => item !== id);
      } else {
        return [...prev, id];
      }
    });
  }, []);

  const handleView = useCallback((piece) => {
    console.log('👁️ View piece:', piece);
  }, []);

  // UPDATED: handleDelete now expects an array of piece labels (unified delete function)
// FIXED: handleDelete in AppDatabasesetup.js
// FIXED: handleDelete in AppDatabasesetup.js
const handleDelete = useCallback(async (pieceInput) => {
  console.log('Delete pieces called with:', pieceInput);
  
  try {
    // FIXED: Normalize input - handle both single piece and array formats
    let labelsToDelete;
    
    if (Array.isArray(pieceInput)) {
      // Array of labels or piece objects
      labelsToDelete = pieceInput.map(item => 
        typeof item === 'string' ? item : item.label
      ).filter(Boolean);
    } else if (typeof pieceInput === 'string') {
      // Single label string
      labelsToDelete = [pieceInput];
    } else if (pieceInput && typeof pieceInput === 'object' && pieceInput.label) {
      // Single piece object - extract label
      labelsToDelete = [pieceInput.label];
    } else {
      console.error('Invalid input for deletion:', pieceInput);
      showNotification('Invalid pieces for deletion', 'error');
      return;
    }

    console.log(`Deleting ${labelsToDelete.length} pieces:`, labelsToDelete);
    
    // FIXED: Ensure we pass a clean array of label strings
    await datasetService.deleteBatchOfPieces(labelsToDelete);
    
    // Update local state to remove deleted pieces
    setData(prevData => 
      prevData.filter(item => !labelsToDelete.includes(item.label))
    );
    
    // Clear selection if any of the deleted pieces were selected
    setSelectedDatasets(prevSelected => {
      const deletedPieceIds = data
        .filter(item => labelsToDelete.includes(item.label))
        .map(item => item.id);
      return prevSelected.filter(id => !deletedPieceIds.includes(id));
    });
    
    // Update selectAll if needed
    if (selectedDatasets.length > 0) {
      setSelectAll(false);
    }
    
    // Refresh data to ensure consistency
    await fetchData();
    
    const message = labelsToDelete.length === 1 
      ? `Successfully deleted piece: ${labelsToDelete[0]}` 
      : `Successfully deleted ${labelsToDelete.length} pieces`;
    showNotification(message, 'success');
    
  } catch (error) {
    console.error('Failed to delete pieces:', error);
    const message = labelsToDelete && labelsToDelete.length === 1
      ? `Failed to delete piece: ${error.message}` 
      : `Failed to delete pieces: ${error.message}`;
    showNotification(message, 'error');
  }
}, [data, selectedDatasets, fetchData, showNotification]);

  // UPDATED: handleBulkDelete - Now just forwards to handleDelete since they use the same logic
  const handleBulkDelete = useCallback(async (pieceLabels) => {
    console.log('📦 Bulk delete pieces:', pieceLabels);
    
    try {
      // Call the unified delete handler
      await handleDelete(pieceLabels);
      
      // Clear selection after bulk delete
      setSelectedDatasets([]);
      setSelectAll(false);
      
    } catch (error) {
      console.error('Failed to bulk delete pieces:', error);
      throw error; // Re-throw to let the child component handle the error display
    }
  }, [handleDelete]);

  // Pagination handlers
  const handlePageChange = useCallback((event, newPage) => {
    console.log('📄 Page change:', newPage);
    setPage(newPage);
  }, []);

  const handleRowsPerPageChange = useCallback((event) => {
    console.log('📄 Rows per page change:', event.target.value);
    setPageSize(parseInt(event.target.value, 10));
    setPage(0);
  }, []);

  const formatDate = useCallback((dateString) => {
    if (!dateString) return 'N/A';
    try {
      return new Date(dateString).toLocaleDateString();
    } catch (error) {
      console.error('❌ Error formatting date:', error);
      return dateString;
    }
  }, []);

  // Debug logging for render
  console.log('🎨 === PARENT COMPONENT RENDER ===');
  console.log('📊 Data available:', !!data);
  console.log('📊 Data length:', data?.length || 0);
  console.log('🔧 handleBatchTrain type:', typeof handleBatchTrain);
  console.log('🔧 handleDelete type:', typeof handleDelete);
  console.log('⚠️ Current training status:', trainingInProgress);

  if (loading) {
    return (
      <Container>
        <LoadingContainer>
          <CircularProgress sx={{ color: '#667eea' }} size={48} />
          <Typography variant="h6" sx={{ opacity: 0.8, mt: 2, fontWeight: "600" }}>
            Loading database...
          </Typography>
          <Typography variant="body2" sx={{ opacity: 0.6 }}>
            Fetching your datasets and pieces
          </Typography>
        </LoadingContainer>
      </Container>
    );
  }

  if (error) {
    return (
      <Container>
        <ErrorContainer>
          <Typography variant="h5" sx={{ fontWeight: "600", mb: 1 }}>
            Unable to Load Database
          </Typography>
          <Typography variant="body1" sx={{ opacity: 0.8, mb: 2 }}>
            {error}
          </Typography>
          <Typography variant="body2" sx={{ opacity: 0.6 }}>
            Please check your connection and try again
          </Typography>
        </ErrorContainer>
      </Container>
    );
  }

  return (
    <Container>
      {data && Array.isArray(data) && data.length > 0 ? (
        <>
          <DatasetComponenet 
            datasets={data}
            selectedDatasets={selectedDatasets}
            selectAll={selectAll}
            onSelectAll={handleSelectAll}
            onSelect={handleSelect}
            onView={handleView}
            onDelete={handleDelete} // ✅ UPDATED: Now handles array of piece labels
            onBulkDelete={handleBulkDelete} // ✅ UPDATED: Forwards to handleDelete
            onTrain={handleTrainingStart}
            trainingInProgress={trainingInProgress}
            trainingData={trainingData}
            page={page}
            pageSize={pageSize}
            totalCount={data ? data.length : 0}
            onPageChange={handlePageChange}
            onRowsPerPageChange={handleRowsPerPageChange}
            formatDate={formatDate}
            onBatchTrain={handleBatchTrain}
            onStopTraining={handleTrainingStop}
            onPauseTraining={handleTrainingPause}
            onResumeTraining={handleTrainingResume}
            // Legacy props for compatibility
            data={data}
            onTrainingStart={handleTrainingStart}
            sidebarOpen={sidebarOpen}
            setSidebarOpen={setSidebarOpen}
            onTrainingCheck={checkTrainingStatus}
          />
          <TrainingProgressSidebar
            isOpen={sidebarOpen}
            onClose={handleSidebarClose}
            trainingData={trainingData}
            onStopTraining={handleTrainingStop}
            onRefresh={handleRefreshTraining}
          />
        </>
      ) : (
        <NoData />
      )}
      
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