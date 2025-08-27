import React, { useState, useEffect, useCallback } from 'react';
import {
  Delete, Visibility, PhotoLibrary, CheckCircle, RadioButtonUnchecked,
  Search, FilterList, Image as ImageIcon,
  Refresh, Clear, PlayArrow, Stop, ModelTraining
} from "@mui/icons-material";
import {
  Box,
  Card,
  Table,
  Avatar,
  styled,
  TableRow,

  TableBody,
  TableCell,
  TableHead,
  IconButton,
  Button,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  Typography,
  Chip,
  Checkbox,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Grid,

  TablePagination,
  Collapse,
  InputAdornment,
  CircularProgress,
  Backdrop,
  Alert,
} from "@mui/material";

// Import your real service
import { datasetService } from './datasetService';



// STYLED COMPONENTS (keeping the same styled components from your original code)
const Container = styled("div")(({ theme }) => ({
  margin: "30px",
  [theme.breakpoints.down("sm")]: { margin: "16px" },
  position: "relative", // Add this for sidebar positioning
}));

const HeaderBox = styled(Box)({
  paddingBottom: "24px",
  borderBottom: "1px solid rgba(102, 126, 234, 0.1)",
  marginBottom: "24px",
  display: "flex",
  alignItems: "center",
  justifyContent: "space-between",
  flexWrap: "wrap",
  gap: "16px",
});

const Title = styled(Typography)({
  fontSize: "1.5rem",
  fontWeight: "700",
  color: "#333",
  textTransform: "none",
});

const FilterCard = styled(Card)(({ theme }) => ({
  marginBottom: "24px",
  border: "1px solid rgba(102, 126, 234, 0.1)",
  borderRadius: "12px",
  overflow: "hidden",
  backgroundColor: "#fff",
}));

const ModernCard = styled(Card)(({ theme }) => ({
  marginBottom: "24px",
  border: "2px solid rgba(102, 126, 234, 0.1)",
  borderRadius: "16px",
  boxShadow: "0 2px 12px rgba(0, 0, 0, 0.08)",
  overflow: "hidden",
  backgroundColor: "#fff",
  transition: "all 0.3s cubic-bezier(0.4, 0, 0.2, 1)",
  "&:hover": {
    boxShadow: "0 12px 32px rgba(102, 126, 234, 0.15)",
    border: "2px solid rgba(102, 126, 234, 0.2)",
  },
}));

const ProductTable = styled(Table)(() => ({
  minWidth: 900,
  "& .MuiTableCell-root": {
    borderBottom: "1px solid rgba(102, 126, 234, 0.08)",
    padding: "16px 12px",
    fontSize: "0.875rem",
  },
  "& .MuiTableHead-root .MuiTableCell-root": {
    backgroundColor: "rgba(102, 126, 234, 0.02)",
    fontWeight: "600",
    color: "#667eea",
    textTransform: "uppercase",
    fontSize: "0.75rem",
    letterSpacing: "1px",
  },
  "& .MuiTableRow-root:hover": {
    backgroundColor: "rgba(102, 126, 234, 0.02)",
  },
}));

const StatusChip = styled(Chip)(({ variant }) => ({
  fontSize: "0.75rem",
  fontWeight: "600",
  height: "28px",
  minWidth: "90px",
  backgroundColor: variant === 'completed' 
    ? "rgba(76, 175, 80, 0.15)" 
    : variant === 'trained'
    ? "rgba(102, 126, 234, 0.15)"
    : "rgba(244, 67, 54, 0.15)",
  color: variant === 'completed' 
    ? "#4caf50" 
    : variant === 'trained'
    ? "#667eea"
    : "#f44336",
  border: `1px solid ${variant === 'completed' 
    ? "rgba(76, 175, 80, 0.3)" 
    : variant === 'trained'
    ? "rgba(102, 126, 234, 0.3)"
    : "rgba(244, 67, 54, 0.3)"}`,
}));

const ActionButton = styled(IconButton)(({ variant }) => ({
  width: "32px",
  height: "32px",
  margin: "0 2px",
  borderRadius: "6px",
  border: "1px solid rgba(102, 126, 234, 0.2)",
  backgroundColor: "transparent",
  color: variant === 'delete' ? "#f44336" : variant === 'train' ? "#667eea" : "#666",
  transition: "all 0.2s cubic-bezier(0.4, 0, 0.2, 1)",
  "&:hover": {
    backgroundColor: variant === 'delete' 
      ? "rgba(244, 67, 54, 0.08)" 
      : variant === 'train' 
      ? "rgba(102, 126, 234, 0.08)"
      : "rgba(0, 0, 0, 0.04)",
    borderColor: variant === 'delete' ? "#f44336" : variant === 'train' ? "#667eea" : "#999",
  },
}));

const TrainButton = styled(Button)(({ theme }) => ({
  textTransform: "none",
  fontSize: "0.875rem",
  marginLeft: theme.spacing(2),
  borderRadius: "8px",
}));

const StatsContainer = styled(Box)({
  display: "flex",
  gap: "16px",
  marginBottom: "24px",
  flexWrap: "wrap",
});

const StatCard = styled(Card)({
  padding: "16px",
  minWidth: "140px",
  textAlign: "center",
  border: "1px solid rgba(102, 126, 234, 0.1)",
  borderRadius: "12px",
  backgroundColor: "rgba(102, 126, 234, 0.02)",
});

const StatValue = styled(Typography)({
  fontSize: "1.8rem",
  fontWeight: "700",
  color: "#667eea",
  lineHeight: 1,
});

const StatLabel = styled(Typography)({
  fontSize: "0.75rem",
  color: "#666",
  textTransform: "uppercase",
  letterSpacing: "1px",
  marginTop: "4px",
});
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
          await datasetService.deleteBatchOfPieces(actionTarget.label);
          showNotification(`Successfully deleted ${actionTarget.label}`, "success");
        } else if (actionType === "bulkDelete" && actionTarget) {
          // For bulk delete, you might need to delete each piece individually
          for (const id of actionTarget) {
            const piece = datasets.find(d => d.id === id);
            if (piece) {
              await datasetService.deleteBatchOfPieces(piece.label);
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

      <HeaderBox>
     
        <Box sx={{ display: 'flex', gap: 2, alignItems: 'center', flexWrap: 'wrap' }}>
          <TrainButton 
            variant="contained" 
            color="primary" 
            onClick={handleTrainAll}
            disabled={trainingInProgress}
            startIcon={trainingInProgress ? <CircularProgress size={16} color="inherit" /> : <PlayArrow />}
          >
            {trainingInProgress ? "Training..." : "Train All"}
          </TrainButton>
          {trainingInProgress && (
            <TrainButton 
              variant="outlined" 
              color="error" 
              onClick={handleStopTraining}
              startIcon={<Stop />}
            >
              Stop Training
            </TrainButton>
          )}
          {trainingInProgress && (
            <Button
              variant="outlined"
              onClick={() => setSidebarOpen(!sidebarOpen)}
              startIcon={<ModelTraining />}
              sx={{ textTransform: "none" }}
            >
              {sidebarOpen ? "Hide Progress" : "Show Progress"}
            </Button>
          )}
          <Button
            startIcon={<FilterList />}
            onClick={() => setShowFilters(!showFilters)}
            variant={showFilters ? "contained" : "outlined"}
            sx={{ textTransform: "none" }}
          >
            Filters
          </Button>
          <Button
            startIcon={<Refresh />}
            onClick={fetchData}
            variant="outlined"
            sx={{ textTransform: "none" }}
          >
            Refresh
          </Button>
          {selectedDatasets.length > 0 && (
            <Button 
              variant="contained"
              color="error"
              onClick={handleBulkDelete}
              sx={{ textTransform: "none" }}
            >
              Delete Selected ({selectedDatasets.length})
            </Button>
          )}
        </Box>
      </HeaderBox>

      {/* Error Alert */}
      {error && (
        <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}

      {/* Statistics Cards */}
      {statistics && (
        <StatsContainer>
          <StatCard>
            <StatValue>{statistics.total_pieces}</StatValue>
            <StatLabel>Total Pieces</StatLabel>
          </StatCard>
          <StatCard>
            <StatValue>{statistics.total_images}</StatValue>
            <StatLabel>Total Images</StatLabel>
          </StatCard>
          <StatCard>
            <StatValue>{statistics.total_annotations}</StatValue>
            <StatLabel>Annotations</StatLabel>
          </StatCard>
          <StatCard>
            <StatValue>{statistics.annotation_completion_rate}%</StatValue>
            <StatLabel>Annotated</StatLabel>
          </StatCard>
          <StatCard>
            <StatValue>{statistics.training_completion_rate}%</StatValue>
            <StatLabel>Trained</StatLabel>
          </StatCard>
          <StatCard>
            <StatValue>{statistics.recent_pieces}</StatValue>
            <StatLabel>Recent</StatLabel>
          </StatCard>
        </StatsContainer>
      )}
      
      {/* Filters Panel */}
      <Collapse in={showFilters}>
        <FilterCard>
          <Box sx={{ p: 3 }}>
            <Typography variant="h6" sx={{ mb: 2, fontWeight: 600, color: "#667eea" }}>
              Search & Filter Options
            </Typography>
            <Grid container spacing={3}>
              <Grid item xs={12} md={6}>
                <TextField
                  fullWidth
                  label="Search"
                  value={filters.search}
                  onChange={(e) => handleFilterChange('search', e.target.value)}
                  placeholder="Search by label or class ID..."
                  InputProps={{
                    startAdornment: (
                      <InputAdornment position="start">
                        <Search />
                      </InputAdornment>
                    ),
                  }}
                />
              </Grid>
              <Grid item xs={12} md={3}>
                <FormControl fullWidth>
                  <InputLabel>Annotation Status</InputLabel>
                  <Select
                    value={filters.status_filter}
                    onChange={(e) => handleFilterChange('status_filter', e.target.value)}
                    label="Annotation Status"
                  >
                    <MenuItem value="">All</MenuItem>
                    <MenuItem value="annotated">Annotated</MenuItem>
                    <MenuItem value="not_annotated">Not Annotated</MenuItem>
                  </Select>
                </FormControl>
              </Grid>
              <Grid item xs={12} md={3}>
                <FormControl fullWidth>
                  <InputLabel>Training Status</InputLabel>
                  <Select
                    value={filters.training_filter}
                    onChange={(e) => handleFilterChange('training_filter', e.target.value)}
                    label="Training Status"
                  >
                    <MenuItem value="">All</MenuItem>
                    <MenuItem value="trained">Trained</MenuItem>
                    <MenuItem value="not_trained">Not Trained</MenuItem>
                  </Select>
                </FormControl>
              </Grid>
              <Grid item xs={12} md={3}>
                <FormControl fullWidth>
                  <InputLabel>Group</InputLabel>
                  <Select
                    value={filters.group_filter}
                    onChange={(e) => handleFilterChange('group_filter', e.target.value)}
                    label="Group"
                  >
                    <MenuItem value="">All Groups</MenuItem>
                    {availableGroups.map(group => (
                      <MenuItem key={group} value={group}>{group}</MenuItem>
                    ))}
                  </Select>
                </FormControl>
              </Grid>
              <Grid item xs={12} md={3}>
                <TextField
                  fullWidth
                  label="Date From"
                  type="date"
                  value={filters.date_from}
                  onChange={(e) => handleFilterChange('date_from', e.target.value)}
                  InputLabelProps={{ shrink: true }}
                />
              </Grid>
              <Grid item xs={12} md={3}>
                <TextField
                  fullWidth
                  label="Date To"
                  type="date"
                  value={filters.date_to}
                  onChange={(e) => handleFilterChange('date_to', e.target.value)}
                  InputLabelProps={{ shrink: true }}
                />
              </Grid>
              <Grid item xs={12} md={3}>
                <TextField
                  fullWidth
                  label="Min Images"
                  type="number"
                  value={filters.min_images}
                  onChange={(e) => handleFilterChange('min_images', e.target.value)}
                />
              </Grid>
              <Grid item xs={12} md={3}>
                <TextField
                  fullWidth
                  label="Max Images"
                  type="number"
                  value={filters.max_images}
                  onChange={(e) => handleFilterChange('max_images', e.target.value)}
                />
              </Grid>
              <Grid item xs={12}>
                <Box sx={{ display: 'flex', gap: 2, justifyContent: 'flex-end' }}>
                  <Button
                    startIcon={<Clear />}
                    onClick={handleClearFilters}
                    variant="outlined"
                    sx={{ textTransform: "none" }}
                  >
                    Clear Filters
                  </Button>
                </Box>
              </Grid>
            </Grid>
          </Box>
        </FilterCard>
      </Collapse>

      {/* Data Table */}
      <ModernCard elevation={0}>
        <ProductTable>
          <TableHead>
            <TableRow>
              <TableCell padding="checkbox">
                <Checkbox
                  checked={selectAll}
                  onChange={handleSelectAll}
                  sx={{ color: "#667eea", '&.Mui-checked': { color: "#667eea" } }}
                />
              </TableCell>
              <TableCell>Piece Details</TableCell>
              <TableCell align="center">Group</TableCell>
              <TableCell align="center">Images</TableCell>
              <TableCell align="center">Annotations</TableCell>
              <TableCell align="center">Status</TableCell>
              <TableCell align="center">Training</TableCell>
              <TableCell align="center">Created</TableCell>
              <TableCell align="center">Actions</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {datasets.map((piece) => (
              <TableRow key={piece.id} hover>
                <TableCell padding="checkbox">
                  <Checkbox
                    checked={selectedDatasets.includes(piece.id)}
                    onChange={() => handleSelect(piece.id)}
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
                        bgcolor: "rgba(102, 126, 234, 0.1)",
                        color: "#667eea"
                      }}
                    >
                      <PhotoLibrary />
                    </Avatar>
                    <Box>
                      <Typography variant="body2" fontWeight="600" color="#333">
                        {piece.label}
                      </Typography>
                      <Typography variant="caption" color="text.secondary">
                        ID: {piece.class_data_id}
                      </Typography>
                    </Box>
                  </Box>
                </TableCell>
                
                <TableCell align="center">
                  <Chip 
                    label={piece.group} 
                    size="small"
                    sx={{ 
                      bgcolor: "rgba(102, 126, 234, 0.1)",
                      color: "#667eea",
                      fontWeight: "600"
                    }}
                  />
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
                  <StatusChip 
                    variant={piece.is_yolo_trained ? "trained" : "pending"}
                    icon={piece.is_yolo_trained ? <CheckCircle /> : <RadioButtonUnchecked />}
                    label={piece.is_yolo_trained ? "Trained" : "Not Trained"}
                    size="small"
                  />
                </TableCell>
                
                <TableCell align="center">
                  <Typography variant="caption" color="text.secondary">
                    {formatDate(piece.created_at)}
                  </Typography>
                </TableCell>
                
                <TableCell align="center">
                  <Box display="flex" justifyContent="center" gap={0.5}>
                    <ActionButton variant="view" onClick={() => handleView(piece)}>
                      <Visibility fontSize="small" />
                    </ActionButton>
                    <ActionButton variant="delete" onClick={() => handleDelete(piece)}>
                      <Delete fontSize="small" />
                    </ActionButton>
                    <Button 
                      onClick={() => handleTrain(piece)} 
                      size="small"
                      variant="outlined"
                      disabled={trainingInProgress}
                      sx={{ 
                        textTransform: "none",
                        minWidth: "60px",
                        fontSize: "0.75rem",
                        py: 0.5
                      }}
                    >
                      {trainingInProgress ? "Training..." : "Train"}
                    </Button>
                  </Box>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </ProductTable>
        
        <TablePagination
          component="div"
          count={totalCount}
          page={page}
          onPageChange={handleChangePage}
          rowsPerPage={pageSize}
          onRowsPerPageChange={handleChangeRowsPerPage}
          rowsPerPageOptions={[5, 10, 25, 50]}
          sx={{
            borderTop: "1px solid rgba(102, 126, 234, 0.1)",
            "& .MuiTablePagination-selectLabel, & .MuiTablePagination-displayedRows": {
              color: "#667eea",
              fontWeight: "500"
            }
          }}
        />
      </ModernCard>

      {/* Confirmation Dialog */}
      <Dialog
        open={confirmationOpen}
        onClose={() => setConfirmationOpen(false)}
        PaperProps={{
          sx: { borderRadius: "16px", padding: "8px" }
        }}
      >
        <DialogTitle sx={{ fontWeight: "600", color: "#333" }}>
          Delete Selected Items
        </DialogTitle>
        <DialogContent>
          <Typography>
            Are you sure you want to delete {selectedDatasets.length} selected item{selectedDatasets.length !== 1 ? 's' : ''}? 
            This action cannot be undone.
          </Typography>
        </DialogContent>
        <DialogActions sx={{ gap: 1, padding: "16px 24px" }}>
          <Button 
            onClick={() => handleConfirmationClose(false)}
            sx={{ textTransform: "none" }}
          >
            Cancel
          </Button>
          <Button 
            onClick={() => handleConfirmationClose(true)} 
            color="error"
            variant="contained"
            sx={{ textTransform: "none" }}
          >
            Delete
          </Button>
        </DialogActions>
      </Dialog>
    </Container>
  );
}