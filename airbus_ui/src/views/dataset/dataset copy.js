import { Box, styled, Typography, CircularProgress } from "@mui/material";
import DataTable from "./datasetTable/Dataset";
import NoData from "../sessions/NoData";
import TrainingProgressSidebar from "./TrainingProgressSidebar";
import { useState, useEffect, useRef } from "react";
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
  
  // Remove polling-related refs and state
  const hasCheckedInitialStatus = useRef(false);

  // Function to check training status (called only when needed)
  const checkTrainingStatus = async () => {
    try {
      const status = await datasetService.getTrainingStatus();

      if (status?.data?.is_training) {
        setTrainingInProgress(true);
        setTrainingData(status.data.session_info); // ✅ Pass only the session_info
        setSidebarOpen(true);
      } else {
        setTrainingInProgress(false);
        setTrainingData(null);
        setSidebarOpen(false);
      }
    } catch (error) {
      console.error("Error checking training status:", error);
      setTrainingInProgress(false);
      setTrainingData(null);
      setSidebarOpen(false);
    }
  };


  // Initial training status check on component mount (only once)
  useEffect(() => {
    if (!hasCheckedInitialStatus.current) {
      checkTrainingStatus();
      hasCheckedInitialStatus.current = true;
    }
  }, []);

  // Fetch datasets data
  useEffect(() => {
    const fetchData = async () => {
      setLoading(true);
      setError(null);
      
      try {
        const pieces = await datasetService.getAllDatasets();
        setData(pieces);

        // Handle cases where pieces are empty
        if (!pieces || Object.keys(pieces).length === 0) {
          navigate("/204");
        }
      } catch (error) {
        console.error("Error fetching data:", error);
        setError(error.response?.data?.detail || "Failed to fetch data");
        // Comment out navigation to see error state
        // navigate("/204");
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, [navigate]);

  // Training handlers
  const handleTrainingStart = async (trainingInfo) => {
    setTrainingData(trainingInfo);
    setTrainingInProgress(true);
    setSidebarOpen(true);
    
    // Check training status once after starting training
    setTimeout(() => {
      checkTrainingStatus();
    }, 1000);
  };

  const handleTrainingStop = async () => {
    try {
      await datasetService.stopTraining();
      setTrainingInProgress(false);
      setSidebarOpen(false);
      setTrainingData(null);
    } catch (error) {
      console.error("Failed to stop training:", error);
    }
  };

  const handleRefreshTraining = async () => {
    try {
      const status = await datasetService.getTrainingStatus();
      if (status?.data?.session_info) {
        setTrainingData(status.data.session_info); // ✅ Fix here too
      }
    } catch (error) {
      console.error("Failed to refresh training data:", error);
    }
  };


  const handleSidebarClose = () => {
    setSidebarOpen(false);
  };

  // Function to manually refresh training status (for page refresh)
  const handlePageRefresh = () => {
    checkTrainingStatus();
  };

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
      {data && Object.keys(data).length > 0 ? (
        <>
          <DataTable 
            data={data} 
            onTrainingStart={handleTrainingStart}
            trainingInProgress={trainingInProgress}
            sidebarOpen={sidebarOpen}
            setSidebarOpen={setSidebarOpen}
            trainingData={trainingData}
            onRefreshTraining={handlePageRefresh}
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
    </Container>
  );
}