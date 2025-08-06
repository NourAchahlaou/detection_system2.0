// DetectionLotsOverview.jsx - FIXED: Proper navigation and piece label display
import React, { useState, useEffect, useCallback } from "react";
import { 
  Box, 
  Typography, 
  Tabs, 
  Tab, 
  styled, 
  Card,
  Chip,
  CircularProgress,
  Button,
  Dialog,
  IconButton,
  Tooltip,
  Menu,
  MenuItem,
  Alert,
  Stack,
  Snackbar
} from "@mui/material";
import { 
  Inventory, 
  CheckCircle,
  Warning,
  RadioButtonUnchecked,
  Visibility,
  Edit,
  Add,
  MoreVert,
  PlayArrow,
  Info,
  AccessTime,
  Label,
  Numbers,
  Category,
  Refresh,
  ErrorOutline
} from "@mui/icons-material";
import { useNavigate } from "react-router-dom";
// Import your detection service
import { detectionService } from "./service/DetectionService"; // Update this path
import DetectionLotForm from "./components/DetectionLotForm"; // Import your form component

// Styled components following your theme (keeping original styling)
const Container = styled("div")(({ theme }) => ({
  margin: "30px",
  [theme.breakpoints.down("sm")]: { margin: "16px" },
}));

const HeaderBox = styled(Box)({
  paddingBottom: "24px",
  borderBottom: "1px solid rgba(102, 126, 234, 0.1)",
  marginBottom: "24px",
  textAlign: "center",
});

const StyledTabs = styled(Tabs)({
  marginBottom: "24px",
  '& .MuiTabs-indicator': {
    backgroundColor: "#667eea",
    height: "3px",
    borderRadius: "3px",
  },
  '& .MuiTab-root': {
    textTransform: "none",
    fontSize: "1rem",
    fontWeight: "600",
    color: "#666",
    minHeight: "48px",
    '&.Mui-selected': {
      color: "#667eea",
    },
  },
});

const CardsGridContainer = styled('div')(({ theme }) => ({
  display: 'grid',
  gap: '20px',
  width: '100%',
  gridTemplateColumns: '1fr',
  [theme.breakpoints.up('sm')]: {
    gridTemplateColumns: 'repeat(2, 1fr)',
  },
  [theme.breakpoints.up('md')]: {
    gridTemplateColumns: 'repeat(3, 1fr)',
  },
  [theme.breakpoints.up('lg')]: {
    gridTemplateColumns: 'repeat(4, 1fr)',
  },
  [theme.breakpoints.up('xl')]: {
    gridTemplateColumns: 'repeat(5, 1fr)',
  },
}));

const LotCard = styled(Card)(({ theme }) => ({
  padding: "20px",
  cursor: "pointer",
  height: "100%",
  display: "flex",
  flexDirection: "column",
  border: "2px solid rgba(102, 126, 234, 0.1)",
  borderRadius: "16px",
  transition: "all 0.3s cubic-bezier(0.4, 0, 0.2, 1)",
  position: "relative",
  overflow: "hidden",
  boxShadow: "0 2px 12px rgba(0, 0, 0, 0.08)",
  minWidth: "0",
  "&:hover": {
    transform: "translateY(-4px)",
    boxShadow: "0 12px 32px rgba(102, 126, 234, 0.2)",
    border: "2px solid #667eea",
    backgroundColor: "rgba(78, 105, 221, 0.45)",
  },
}));

const CardHeader = styled(Box)({
  display: "flex",
  alignItems: "flex-start",
  justifyContent: "space-between",
  marginBottom: "16px",
  gap: "12px",
});

const IconContainer = styled(Box)({
  width: "48px",
  height: "48px",
  borderRadius: "12px",
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  backgroundColor: "rgba(102, 126, 234, 0.15)",
  color: "#667eea",
  marginBottom: "12px",
});

const LotTitle = styled(Typography)({
  fontSize: "1.1rem",
  fontWeight: "700",
  color: "#333",
  marginBottom: "8px",
  overflow: "hidden",
  textOverflow: "ellipsis",
  whiteSpace: "nowrap",
  "&:hover": {
    color: "#e2e2e2",
  },
});

const StatsContainer = styled(Box)({
  display: "flex",
  flexDirection: "column",
  gap: "8px",
  marginTop: "auto",
});

const StatsRow = styled(Box)({
  display: "flex",
  alignItems: "center",
  justifyContent: "space-between",
});

const StatusChip = styled(Chip)(({ variant }) => ({
  fontSize: "0.75rem",
  fontWeight: "600",
  height: "24px",
  backgroundColor: variant === 'completed' 
    ? "rgba(76, 175, 80, 0.15)" 
    : variant === 'pending'
    ? "rgba(255, 152, 0, 0.15)"
    : "rgba(244, 67, 54, 0.15)",
  color: variant === 'completed' 
    ? "#4caf50" 
    : variant === 'pending'
    ? "#ff9800"
    : "#f44336",
  "& .MuiChip-icon": {
    fontSize: "14px",
  },
}));

const ActionButton = styled(Button)(({ variant }) => ({
  textTransform: "none",
  fontWeight: "600",
  borderRadius: "6px",
  fontSize: "0.8rem",
  padding: "4px 12px",
  minWidth: "auto",
  backgroundColor: variant === 'primary' ? "#667eea" : "transparent",
  color: variant === 'primary' ? "white" : "#667eea",
  border: variant === 'primary' ? "none" : "1px solid rgba(102, 126, 234, 0.3)",
  "&:hover": {
    backgroundColor: variant === 'primary' ? "#5a67d8" : "rgba(102, 126, 234, 0.08)",
  },
}));

const LoadingContainer = styled(Box)({
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  minHeight: "300px",
  flexDirection: "column",
  gap: 2,
  color: "#666",
});

const EmptyState = styled(Box)({
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  minHeight: "300px",
  flexDirection: "column",
  gap: 2,
  color: "#666",
  textAlign: "center",
});

const HeaderActions = styled(Box)({
  display: "flex",
  gap: "12px",
  alignItems: "center",
  justifyContent: "center",
  marginTop: "16px",
});

function TabPanel({ children, value, index }) {
  return (
    <div hidden={value !== index}>
      {value === index && children}
    </div>
  );
}

export default function DetectionLotsOverview() {
  const [tabValue, setTabValue] = useState(0);
  const [allLots, setAllLots] = useState([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState(null);
  const [stats, setStats] = useState({
    total: 0,
    completed: 0,
    pending: 0,
    notStarted: 0
  });
  
  // NEW: Piece labels cache
  const [pieceLabels, setPieceLabels] = useState(new Map());
  const [loadingLabels, setLoadingLabels] = useState(new Set());
  
  // Form state
  const [showCreateForm, setShowCreateForm] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  
  // Menu state
  const [anchorEl, setAnchorEl] = useState(null);
  const [selectedLot, setSelectedLot] = useState(null);
  
  // Snackbar state
  const [snackbar, setSnackbar] = useState({
    open: false,
    message: '',
    severity: 'success'
  });
  
  const navigate = useNavigate();

  useEffect(() => {
    fetchAllLots();
  }, []);

  const showSnackbar = (message, severity = 'success') => {
    setSnackbar({
      open: true,
      message,
      severity
    });
  };

  const closeSnackbar = () => {
    setSnackbar(prev => ({ ...prev, open: false }));
  };

  // NEW: Function to fetch piece label by ID
  const fetchPieceLabel = useCallback(async (pieceId) => {
    if (pieceLabels.has(pieceId) || loadingLabels.has(pieceId)) {
      return pieceLabels.get(pieceId);
    }

    setLoadingLabels(prev => new Set(prev).add(pieceId));

    try {
      const response = await fetch(`/api/artifact_keeper/captureImage/piece_label_byid/${pieceId}`);
      
      if (response.ok) {
        const label = await response.text();
        // Remove quotes if the API returns a quoted string
        const cleanLabel = label.replace(/^"|"$/g, '');
        
        setPieceLabels(prev => new Map(prev).set(pieceId, cleanLabel));
        setLoadingLabels(prev => {
          const newSet = new Set(prev);
          newSet.delete(pieceId);
          return newSet;
        });
        
        return cleanLabel;
      } else {
        throw new Error(`Failed to fetch piece label for ID ${pieceId}`);
      }
    } catch (error) {
      console.error(`Error fetching piece label for ID ${pieceId}:`, error);
      setLoadingLabels(prev => {
        const newSet = new Set(prev);
        newSet.delete(pieceId);
        return newSet;
      });
      
      // Return fallback label
      const fallback = `Piece ${pieceId}`;
      setPieceLabels(prev => new Map(prev).set(pieceId, fallback));
      return fallback;
    }
  }, [pieceLabels, loadingLabels]);

  // NEW: Function to get piece label with loading state
  const getPieceLabel = useCallback((pieceId) => {
    if (pieceLabels.has(pieceId)) {
      return pieceLabels.get(pieceId);
    }
    
    if (loadingLabels.has(pieceId)) {
      return 'Loading...';
    }
    
    // Trigger fetch
    fetchPieceLabel(pieceId);
    return `Piece ${pieceId}`; // Fallback while loading
  }, [pieceLabels, loadingLabels, fetchPieceLabel]);

  const fetchAllLots = async () => {
    try {
      setLoading(true);
      setError(null);
      
      // Ensure detection service is initialized
      await detectionService.ensureInitialized();
      
      // Fetch lots using the real service
      const response = await detectionService.streamManager.getAllDetectionLots();
      
      if (response.success) {
        const lotsData = response.lots || [];
        
        console.log("Fetched lots data:", lotsData);
        
        setAllLots(lotsData);
        
        // NEW: Fetch piece labels for all unique piece IDs
        const uniquePieceIds = [...new Set(lotsData.map(lot => lot.expected_piece_id))];
        console.log("Fetching labels for piece IDs:", uniquePieceIds);
        
        // Fetch labels in parallel
        await Promise.all(
          uniquePieceIds.map(pieceId => fetchPieceLabel(pieceId))
        );
        
        // Calculate stats
        const totalLots = lotsData.length;
        const completedLots = lotsData.filter(lot => lot.is_target_match === true).length;
        const pendingLots = lotsData.filter(lot => lot.is_target_match === false).length;
        const notStartedLots = lotsData.filter(lot => lot.is_target_match === null).length;
        
        setStats({
          total: totalLots,
          completed: completedLots,
          pending: pendingLots,
          notStarted: notStartedLots
        });
        
        showSnackbar(`Successfully loaded ${totalLots} detection lots`);
      } else {
        throw new Error(response.message || 'Failed to fetch detection lots');
      }
      
    } catch (error) {
      console.error("Error fetching lots:", error);
      setError(error.message);
      setAllLots([]);
      setStats({ total: 0, completed: 0, pending: 0, notStarted: 0 });
      showSnackbar(`Error loading lots: ${error.message}`, 'error');
    } finally {
      setLoading(false);
    }
  };

  const handleRefresh = async () => {
    setRefreshing(true);
    try {
      await fetchAllLots();
    } catch (error) {
      showSnackbar(`Error refreshing: ${error.message}`, 'error');
    } finally {
      setRefreshing(false);
    }
  };

  const handleTabChange = (event, newValue) => {
    setTabValue(newValue);
  };

  const handleLotClick = (lot) => {
    // Navigate to lot details or detection interface
    navigate(`/detection?lotId=${lot.lot_id}`);
  };

  const handleMenuOpen = (event, lot) => {
    event.stopPropagation();
    setAnchorEl(event.currentTarget);
    setSelectedLot(lot);
  };

  const handleMenuClose = () => {
    setAnchorEl(null);
    setSelectedLot(null);
  };

  const handleViewLot = async () => {
    if (selectedLot) {
      try {
        const sessions = await detectionService.streamManager.getLotDetectionSessions(selectedLot.lot_id);
        console.log('Lot sessions:', sessions);
        // Navigate to detailed view or show modal
        navigate(`/detection/lot/${selectedLot.lot_id}/sessions`);
        showSnackbar(`Viewing ${sessions.totalSessions} sessions for lot ${sessions.lotName}`);
      } catch (error) {
        console.error('Error viewing lot:', error);
        showSnackbar(`Error viewing lot: ${error.message}`, 'error');
      }
    }
    handleMenuClose();
  };

  // FIXED: Only navigate to detection page, don't start detection
  const handleSelectForDetection = () => {
    if (selectedLot) {
      // Navigate to detection interface with pre-selected lot (no mode parameter)
      navigate(`/detection?lotId=${selectedLot.lot_id}`);
      showSnackbar(`Selected lot ${selectedLot.lot_name} for detection. Please choose your camera and start detection.`);
    }
    handleMenuClose();
  };

  const handleCreateLot = async (formData) => {
    try {
      setIsSubmitting(true);
      
      // Use the real service to create lot
      const result = await detectionService.streamManager.createDetectionLot(
        formData.lotName,
        formData.expectedPieceId,
        formData.expectedPieceNumber
      );
      
      if (result.success) {
        console.log('Lot created successfully:', result);
        showSnackbar(`Successfully created lot: ${formData.lotName}`);
        await fetchAllLots(); // Refresh the list
        setShowCreateForm(false);
      } else {
        throw new Error(result.message || 'Failed to create lot');
      }
    } catch (error) {
      console.error('Error creating lot:', error);
      showSnackbar(`Error creating lot: ${error.message}`, 'error');
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleEditLot = async () => {
    if (selectedLot) {
      // Navigate to edit page or open edit modal
      navigate(`/detection/lot/${selectedLot.lot_id}/edit`);
    }
    handleMenuClose();
  };

  const handleUpdateLotStatus = async (lot, newStatus) => {
    try {
      const result = await detectionService.streamManager.updateLotTargetMatchStatus(
        lot.lot_id, 
        newStatus
      );
      
      if (result.success) {
        showSnackbar(`Updated lot status: ${newStatus ? 'Completed' : 'Pending'}`);
        await fetchAllLots(); // Refresh the list
      } else {
        throw new Error(result.message || 'Failed to update lot status');
      }
    } catch (error) {
      console.error('Error updating lot status:', error);
      showSnackbar(`Error updating lot status: ${error.message}`, 'error');
    }
  };

  const getStatusInfo = (lot) => {
    if (lot.is_target_match === null) {
      return {
        variant: 'not-started',
        label: 'Not Started',
        icon: <RadioButtonUnchecked />,
        color: '#999'
      };
    } else if (lot.is_target_match === false) {
      return {
        variant: 'pending',
        label: 'Pending',
        icon: <Warning />,
        color: '#ff9800'
      };
    } else {
      return {
        variant: 'completed',
        label: 'Completed',
        icon: <CheckCircle />,
        color: '#4caf50'
      };
    }
  };

  const formatDate = (dateString) => {
    if (!dateString) return 'N/A';
    return new Date(dateString).toLocaleString();
  };

  const renderLotCard = (lot) => {
    const statusInfo = getStatusInfo(lot);
    // FIXED: Use piece label instead of piece ID
    const pieceLabel = getPieceLabel(lot.expected_piece_id);
    
    return (
      <Card key={lot.lot_id} elevation={0} onClick={() => handleLotClick(lot)}>
        <LotCard>
          <CardHeader>
            <Box sx={{ display: "flex", alignItems: "center", gap: 1, flex: 1, minWidth: 0 }}>
              <IconContainer>
                <Inventory fontSize="medium" />
              </IconContainer>
              <Box sx={{ minWidth: 0, flex: 1 }}>
                <LotTitle title={lot.lot_name}>{lot.lot_name}</LotTitle>
                <Box sx={{ display: "flex", alignItems: "center", gap: 1, mb: 1 }}>
                  <Label sx={{ fontSize: 14, color: "#667eea" }} />
                  <Typography variant="caption" sx={{ color: "#666", fontWeight: "500" }}>
                    {pieceLabel}
                  </Typography>
                </Box>
                <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                  <Numbers sx={{ fontSize: 14, color: "#667eea" }} />
                  <Typography variant="caption" sx={{ color: "#666", fontWeight: "500" }}>
                    Expected: {lot.expected_piece_number}
                  </Typography>
                </Box>
              </Box>
            </Box>
            
            <IconButton 
              size="small" 
              onClick={(e) => handleMenuOpen(e, lot)}
              sx={{ alignSelf: 'flex-start' }}
            >
              <MoreVert fontSize="small" />
            </IconButton>
          </CardHeader>
          
          <Box sx={{ mb: 2 }}>
            <Box sx={{ display: "flex", alignItems: "center", gap: 1, mb: 1 }}>
              <AccessTime sx={{ fontSize: 14, color: "#667eea" }} />
              <Typography variant="caption" sx={{ color: "#666" }}>
                Created: {formatDate(lot.created_at)}
              </Typography>
            </Box>
            {lot.last_detection_at && (
              <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                <PlayArrow sx={{ fontSize: 14, color: "#667eea" }} />
                <Typography variant="caption" sx={{ color: "#666" }}>
                  Last detection: {formatDate(lot.last_detection_at)}
                </Typography>
              </Box>
            )}
          </Box>
          
          <StatsContainer>
            <StatsRow>
              <StatusChip
                variant={statusInfo.variant}
                icon={statusInfo.icon}
                label={statusInfo.label}
                size="small"
              />
              
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                <Typography variant="caption" sx={{ color: "#666" }}>
                  {lot.sessions_count || 0} sessions
                </Typography>
              </Box>
            </StatsRow>
            
            <StatsRow>
              <ActionButton
                variant="primary"
                size="small"
                startIcon={<PlayArrow />}
                onClick={(e) => {
                  e.stopPropagation();
                  // FIXED: Only navigate, don't start detection
                  navigate(`/detection?lotId=${lot.lot_id}`);
                  showSnackbar(`Selected lot ${lot.lot_name} for detection. Please choose your camera and start detection.`);
                }}
              >
                Select for Detection
              </ActionButton>
              
              <ActionButton
                variant="secondary"
                size="small"
                startIcon={<Visibility />}
                onClick={(e) => {
                  e.stopPropagation();
                  navigate(`/detection/lot/${lot.lot_id}/sessions`);
                }}
              >
                View
              </ActionButton>
            </StatsRow>
          </StatsContainer>
        </LotCard>
      </Card>
    );
  };

  const getFilteredLots = () => {
    switch (tabValue) {
      case 0:
        return allLots;
      case 1:
        return allLots.filter(lot => lot.is_target_match === null);
      case 2:
        return allLots.filter(lot => lot.is_target_match === false);
      case 3:
        return allLots.filter(lot => lot.is_target_match === true);
      default:
        return allLots;
    }
  };

  if (loading) {
    return (
      <Container>
        <LoadingContainer>
          <CircularProgress sx={{ color: '#667eea' }} size={48} />
          <Typography variant="h6" sx={{ opacity: 0.8, mt: 2 }}>
            Loading detection lots...
          </Typography>
        </LoadingContainer>
      </Container>
    );
  }

  if (error) {
    return (
      <Container>
        <LoadingContainer>
          <ErrorOutline sx={{ color: '#f44336', fontSize: 48, mb: 2 }} />
          <Typography variant="h6" sx={{ opacity: 0.8, mb: 2 }}>
            Error Loading Detection Lots
          </Typography>
          <Typography variant="body2" sx={{ opacity: 0.7, mb: 3, textAlign: 'center' }}>
            {error}
          </Typography>
          <Button 
            variant="contained" 
            onClick={handleRefresh}
            sx={{ backgroundColor: '#667eea' }}
          >
            Retry
          </Button>
        </LoadingContainer>
      </Container>
    );
  }

  const filteredLots = getFilteredLots();

  return (
    <Container>
      <HeaderBox>
        <Typography variant="h4" sx={{ color: '#333', fontWeight: '700', mb: 2 }}>
          Detection Lots Management
        </Typography>
        
        <Box sx={{ 
          display: 'flex', 
          gap: 3, 
          mt: 3, 
          flexWrap: 'wrap',
          justifyContent: 'center'
        }}>
          <Box sx={{ textAlign: 'center' }}>
            <Typography variant="h4" sx={{ color: '#667eea', fontWeight: '700' }}>
              {stats.total}
            </Typography>
            <Typography variant="caption" sx={{ color: '#666', textTransform: 'uppercase', letterSpacing: 1 }}>
              Total Lots
            </Typography>
          </Box>
          <Box sx={{ textAlign: 'center' }}>
            <Typography variant="h4" sx={{ color: '#4caf50', fontWeight: '700' }}>
              {stats.completed}
            </Typography>
            <Typography variant="caption" sx={{ color: '#666', textTransform: 'uppercase', letterSpacing: 1 }}>
              Completed
            </Typography>
          </Box>
          <Box sx={{ textAlign: 'center' }}>
            <Typography variant="h4" sx={{ color: '#ff9800', fontWeight: '700' }}>
              {stats.pending}
            </Typography>
            <Typography variant="caption" sx={{ color: '#666', textTransform: 'uppercase', letterSpacing: 1 }}>
              Pending
            </Typography>
          </Box>
          <Box sx={{ textAlign: 'center' }}>
            <Typography variant="h4" sx={{ color: '#999', fontWeight: '700' }}>
              {stats.notStarted}
            </Typography>
            <Typography variant="caption" sx={{ color: '#666', textTransform: 'uppercase', letterSpacing: 1 }}>
              Not Started
            </Typography>
          </Box>
        </Box>

        <HeaderActions>
          <Button
            variant="contained"
            startIcon={<Add />}
            onClick={() => setShowCreateForm(true)}
            sx={{ 
              backgroundColor: '#667eea', 
              '&:hover': { backgroundColor: '#5a67d8' },
              fontWeight: '600'
            }}
          >
            Create New Lot
          </Button>
          
          <Button
            variant="outlined"
            startIcon={refreshing ? <CircularProgress size={16} /> : <Refresh />}
            onClick={handleRefresh}
            disabled={refreshing}
            sx={{ 
              borderColor: '#667eea', 
              color: '#667eea',
              '&:hover': { borderColor: '#5a67d8', backgroundColor: 'rgba(102, 126, 234, 0.08)' }
            }}
          >
            {refreshing ? 'Refreshing...' : 'Refresh'}
          </Button>
        </HeaderActions>
      </HeaderBox>

      <StyledTabs value={tabValue} onChange={handleTabChange}>
        <Tab label={`All Lots (${stats.total})`} />
        <Tab label={`Not Started (${stats.notStarted})`} />
        <Tab label={`Pending (${stats.pending})`} />
        <Tab label={`Completed (${stats.completed})`} />
      </StyledTabs>

      <TabPanel value={tabValue} index={0}>
        <CardsGridContainer>
          {filteredLots.map(renderLotCard)}
        </CardsGridContainer>
        
        {filteredLots.length === 0 && (
          <EmptyState>
            <Inventory sx={{ fontSize: 64, opacity: 0.4, mb: 2 }} />
            <Typography variant="h6" sx={{ opacity: 0.9, mb: 1 }}>
              No Lots Found
            </Typography>
            <Typography variant="body2" sx={{ opacity: 0.7 }}>
              Create your first detection lot to get started
            </Typography>
          </EmptyState>
        )}
      </TabPanel>

      <TabPanel value={tabValue} index={1}>
        <CardsGridContainer>
          {filteredLots.map(renderLotCard)}
        </CardsGridContainer>
        
        {filteredLots.length === 0 && (
          <EmptyState>
            <CheckCircle sx={{ fontSize: 64, opacity: 0.4, mb: 2, color: '#4caf50' }} />
            <Typography variant="h6" sx={{ opacity: 0.9, mb: 1 }}>
              All Lots Have Been Started!
            </Typography>
            <Typography variant="body2" sx={{ opacity: 0.7 }}>
              All detection lots have detection sessions
            </Typography>
          </EmptyState>
        )}
      </TabPanel>

      <TabPanel value={tabValue} index={2}>
        <CardsGridContainer>
          {filteredLots.map(renderLotCard)}
        </CardsGridContainer>
        
        {filteredLots.length === 0 && (
          <EmptyState>
            <CheckCircle sx={{ fontSize: 64, opacity: 0.4, mb: 2, color: '#4caf50' }} />
            <Typography variant="h6" sx={{ opacity: 0.9, mb: 1 }}>
              No Pending Lots!
            </Typography>
            <Typography variant="body2" sx={{ opacity: 0.7 }}>
              All lots are either completed or not started
            </Typography>
          </EmptyState>
        )}
      </TabPanel>

      <TabPanel value={tabValue} index={3}>
        <CardsGridContainer>
          {filteredLots.map(renderLotCard)}
        </CardsGridContainer>
        
        {filteredLots.length === 0 && (
          <EmptyState>
            <Warning sx={{ fontSize: 64, opacity: 0.4, mb: 2, color: '#ff9800' }} />
            <Typography variant="h6" sx={{ opacity: 0.9, mb: 1 }}>
              No Completed Lots Yet
            </Typography>
            <Typography variant="body2" sx={{ opacity: 0.7 }}>
              Complete some detection sessions to see results here
            </Typography>
          </EmptyState>
        )}
      </TabPanel>

      {/* Context Menu */}
      <Menu
        anchorEl={anchorEl}
        open={Boolean(anchorEl)}
        onClose={handleMenuClose}
        anchorOrigin={{
          vertical: 'bottom',
          horizontal: 'right',
        }}
transformOrigin={{
          vertical: 'top',
          horizontal: 'right',
        }}
      >
        <MenuItem onClick={handleViewLot}>
          <Visibility sx={{ mr: 1, fontSize: 20 }} />
          View Details
        </MenuItem>
        <MenuItem onClick={handleSelectForDetection}>
          <PlayArrow sx={{ mr: 1, fontSize: 20 }} />
          Use for Detection
        </MenuItem>
        <MenuItem onClick={handleEditLot}>
          <Edit sx={{ mr: 1, fontSize: 20 }} />
          Edit Lot
        </MenuItem>
        {selectedLot && selectedLot.is_target_match === false && (
          <MenuItem onClick={() => handleUpdateLotStatus(selectedLot, true)}>
            <CheckCircle sx={{ mr: 1, fontSize: 20, color: '#4caf50' }} />
            Mark as Completed
          </MenuItem>
        )}
        {selectedLot && selectedLot.is_target_match === true && (
          <MenuItem onClick={() => handleUpdateLotStatus(selectedLot, false)}>
            <Warning sx={{ mr: 1, fontSize: 20, color: '#ff9800' }} />
            Mark as Pending
          </MenuItem>
        )}
      </Menu>

      {/* Create Lot Form */}
      <DetectionLotForm
        isOpen={showCreateForm}
        onClose={() => setShowCreateForm(false)}
        onCreateLot={handleCreateLot}
        onRefreshLots={handleRefresh}
        isSubmitting={isSubmitting}
        existingLots={allLots}
        // Default values for form
        cameraId={1}
        targetLabel="default"
        detectionOptions={{
          quality: 85,
          detectionFps: null,
          priority: 1,
          enableAdaptiveQuality: false
        }}
      />

      {/* Snackbar for notifications */}
      <Snackbar
        open={snackbar.open}
        autoHideDuration={6000}
        onClose={closeSnackbar}
        message={snackbar.message}
        action={
          <Button color="inherit" size="small" onClick={closeSnackbar}>
            Close
          </Button>
          }
        sx={{
          '& .MuiSnackbarContent-root': {
            backgroundColor: snackbar.severity === 'error' ? '#f44336' : '#4caf50'
          }
        }}
      />
    </Container>
  );
}