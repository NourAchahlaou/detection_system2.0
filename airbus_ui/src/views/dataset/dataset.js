import { Box, styled, Typography, CircularProgress } from "@mui/material";
import DataTable from "./DatasetTable";
import NoData from "../sessions/NoData";
import TrainingProgressSidebar from "./TrainingProgressSidebar"; // Add this import
import { useState, useEffect } from "react";
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

  // Training sidebar state - Add these states
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [trainingData, setTrainingData] = useState(null);
  const [trainingInProgress, setTrainingInProgress] = useState(false);

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

  // Training handlers - Add these functions
  const handleTrainingStart = (trainingInfo) => {
    setTrainingData(trainingInfo);
    setTrainingInProgress(true);
    setSidebarOpen(true);
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

  const handleRefreshTraining = () => {
    // Refresh training data - in a real app, this would fetch from API
    console.log("Refreshing training data...");
  };

  const handleSidebarClose = () => {
    setSidebarOpen(false);
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