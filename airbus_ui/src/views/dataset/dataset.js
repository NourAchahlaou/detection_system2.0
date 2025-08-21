import { Box, styled, Typography, CircularProgress, Snackbar, Alert } from "@mui/material";
import DataTable from "./datasetTable/Dataset";
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
  
  // Selection state - Add this for DataTable
  const [selectedDatasets, setSelectedDatasets] = useState([]);
  const [selectAll, setSelectAll] = useState(false);
  
  // Pagination state - Add this for DataTable
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
      const status = await datasetService.getTrainingStatus(true); // Include logs
      console.log('Training status response:', status);

      if (status?.data?.is_training && status?.data?.session_info) {
        console.log('✅ Training is active');
        setTrainingInProgress(true);
        setTrainingData(status.data.session_info);
        setCurrentSessionId(status.data.session_info.id);
        setSidebarOpen(true);
        
        // Start polling for updates when training is active
        if (!trainingStatusInterval.current) {
          startTrainingPolling();
        }
      } else {
        console.log('❌ No active training');
        setTrainingInProgress(false);
        setTrainingData(null);
        setCurrentSessionId(null);
        setSidebarOpen(false);
        
        // Stop polling when training is not active
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

  // Start polling for training updates (only when training is active)
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
          // Training finished
          console.log('🎉 Training completed!');
          setTrainingInProgress(false);
          setTrainingData(null);
          setCurrentSessionId(null);
          setSidebarOpen(false);
          stopTrainingPolling();
          showNotification('Training session completed!', 'success');
          
          // Refresh dataset data to show updated training status
          fetchData();
        }
      } catch (error) {
        console.error("❌ Error during training polling:", error);
        stopTrainingPolling();
      }
    }, 3000); // Poll every 3 seconds when training is active
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
      // Convert object to array if needed
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

  // Enhanced training handlers - ALL USING useCallback to ensure stable references
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

  // 🔥 COMPLETELY REWRITTEN handleBatchTrain with extensive debugging
// In your AppDatabasesetup.js, replace the handleBatchTrain function with this corrected version:

// CORRECTED handleBatchTrain function - ensures it always returns a result
// FIXED: handleBatchTrain function in AppDatabasesetup.js
const handleBatchTrain = useCallback(async (pieceLabels, sequential = false) => {
  console.log('=== DEBUG handleBatchTrain START ===');
  console.log('Input pieceLabels:', pieceLabels);
  console.log('Input sequential:', sequential);
  
  try {
    console.log('Step 1: Function entered successfully');
    
    // Basic validation - ALWAYS return a result object
    if (!pieceLabels) {
      console.log('Step 2: No pieceLabels provided - returning error');
      const result = { success: false, error: 'No pieces provided' };
      console.log('Returning result:', result);
      showNotification('No pieces provided for training', 'error');
      return result;
    }

    if (!Array.isArray(pieceLabels)) {
      console.log('Step 2: pieceLabels not an array - returning error');
      const result = { success: false, error: 'Invalid format' };
      console.log('Returning result:', result);
      showNotification('Invalid piece format', 'error');
      return result;
    }

    if (pieceLabels.length === 0) {
      console.log('Step 2: Empty pieceLabels array - returning error');
      const result = { success: false, error: 'Empty selection' };
      console.log('Returning result:', result);
      showNotification('No pieces selected for training', 'error');
      return result;
    }

    console.log('Step 3: Basic validation passed');

    // Check training in progress
    if (trainingInProgress) {
      console.log('Step 4: Training already in progress - returning error');
      const result = { success: false, error: 'Training in progress' };
      console.log('Returning result:', result);
      showNotification('Training is already in progress', 'warning');
      return result;
    }

    console.log('Step 5: No training in progress, continuing');

    // Check datasetService
    if (!datasetService) {
      console.log('Step 6: datasetService not available - returning error');
      const result = { success: false, error: 'Service unavailable' };
      console.log('Returning result:', result);
      showNotification('Training service unavailable', 'error');
      return result;
    }

    console.log('Step 7: datasetService available');

    if (typeof datasetService.trainMultiplePieces !== 'function') {
      console.log('Step 8: trainMultiplePieces method not found - returning error');
      console.log('Available methods:', Object.keys(datasetService));
      const result = { success: false, error: 'Method unavailable' };
      console.log('Returning result:', result);
      showNotification('Training method not available', 'error');
      return result;
    }

    console.log('Step 9: trainMultiplePieces method available');
    console.log('About to call API with:', pieceLabels);

    // Show notification BEFORE API call
    showNotification(`Starting training for ${pieceLabels.length} pieces...`, 'info');

    // Make API call with proper error handling
    console.log('Step 10: Making API call...');
    const apiResult = await datasetService.trainMultiplePieces(pieceLabels);
    console.log('Step 11: API call completed, result:', apiResult);

    // Handle API response
    if (!apiResult) {
      console.log('Step 12: API returned null/undefined');
      const result = { success: false, error: 'No response from training service' };
      console.log('Returning result:', result);
      showNotification('No response from training service', 'error');
      return result;
    }

    if (apiResult.data) {
      console.log('Step 12: API returned valid data');
      
      // Update UI state if session ID is provided
      if (apiResult.data.session_id) {
        console.log('Step 13: Setting up training session');
        setCurrentSessionId(apiResult.data.session_id);
        setTrainingInProgress(true);
        setSidebarOpen(true);
        startTrainingPolling();
      }
      
      showNotification(`Training started successfully!`, 'success');
      const result = { success: true, data: apiResult.data };
      console.log('Step 14: Returning success result:', result);
      return result;
    } else {
      console.log('Step 12: API returned invalid data format');
      console.log('API Result structure:', apiResult);
      const result = { success: false, error: 'Invalid API response format' };
      console.log('Returning result:', result);
      showNotification('Invalid response from training service', 'error');
      return result;
    }

  } catch (error) {
    console.log('Step ERROR: Exception caught:', error);
    console.log('Error name:', error.name);
    console.log('Error message:', error.message);
    console.log('Error stack:', error.stack);
    
    const errorMessage = error.response?.data?.detail || error.message || 'Unknown error occurred';
    showNotification(`Training failed: ${errorMessage}`, 'error');
    
    const result = { success: false, error: errorMessage };
    console.log('Returning error result:', result);
    return result;
  } finally {
    console.log('=== DEBUG handleBatchTrain END ===');
  }
}, [showNotification, startTrainingPolling, trainingInProgress, datasetService]);

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
      
      // Refresh data to show updated status
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
      checkTrainingStatus(); // Refresh status
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

  // Manual refresh function
  const handlePageRefresh = useCallback(() => {
    console.log('🔄 Manual page refresh triggered');
    checkTrainingStatus();
    fetchData();
  }, [checkTrainingStatus, fetchData]);

  // Add missing DataTable handler functions - ALL with useCallback
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
    // Implement view functionality
    console.log('👁️ View piece:', piece);
  }, []);

  const handleDelete = useCallback((piece) => {
    // Implement delete functionality
    console.log('🗑️ Delete piece:', piece);
  }, []);

  // FIXED: Proper pagination handlers with correct signatures
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

  // Add debug logging for render
  console.log('🎨 === PARENT COMPONENT RENDER ===');
  console.log('📊 Data available:', !!data);
  console.log('📊 Data length:', data?.length || 0);
  console.log('🔧 handleBatchTrain type:', typeof handleBatchTrain);
  console.log('🔧 handleBatchTrain is function:', typeof handleBatchTrain === 'function');
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
          <DataTable 
            datasets={data}
            selectedDatasets={selectedDatasets}
            selectAll={selectAll}
            onSelectAll={handleSelectAll}
            onSelect={handleSelect}
            onView={handleView}
            onDelete={handleDelete}
            onTrain={handleTrainingStart}
            trainingInProgress={trainingInProgress}
            trainingData={trainingData}
            page={page}
            pageSize={pageSize}
            totalCount={data ? data.length : 0}
            onPageChange={handlePageChange}
            onRowsPerPageChange={handleRowsPerPageChange}
            formatDate={formatDate}
            onBatchTrain={handleBatchTrain} // 🔥 ENHANCED CALLBACK WITH FULL DEBUGGING
            onStopTraining={handleTrainingStop}
            onPauseTraining={handleTrainingPause}
            onResumeTraining={handleTrainingResume}
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