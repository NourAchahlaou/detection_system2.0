import {  Delete, Visibility, PhotoLibrary, CheckCircle, RadioButtonUnchecked } from "@mui/icons-material";
import {
  Box,
  Card,
  Table,
  Avatar,
  styled,
  TableRow,
  useTheme,
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

  Checkbox
} from "@mui/material";

import { useEffect, useState } from "react";
import axios from "axios";
import TrainingProgressModal from "./ProgressBar";

// STYLED COMPONENTS - Following the modern theme
const Container = styled("div")(({ theme }) => ({
  margin: "30px",
  [theme.breakpoints.down("sm")]: { margin: "16px" },
}));

const HeaderBox = styled(Box)({
  paddingBottom: "24px",
  borderBottom: "1px solid rgba(102, 126, 234, 0.1)",
  marginBottom: "24px",
  display: "flex",
  alignItems: "center",
  justifyContent: "space-between",
});

const Title = styled(Typography)({
  fontSize: "1.5rem",
  fontWeight: "700",
  color: "#333",
  textTransform: "none",
});

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

const GroupHeader = styled(Box)(({ theme }) => ({
  padding: "20px 24px",
  backgroundColor: "rgba(102, 126, 234, 0.04)",
  borderBottom: "1px solid rgba(102, 126, 234, 0.1)",
  display: "flex",
  alignItems: "center",
  justifyContent: "space-between",
}));

const GroupTitle = styled(Typography)({
  fontSize: "1.2rem",
  fontWeight: "600",
  color: "#667eea",
  display: "flex",
  alignItems: "center",
  gap: "12px",
});

const ProductTable = styled(Table)(() => ({
  minWidth: 400,
  "& .MuiTableCell-root": {
    borderBottom: "1px solid rgba(102, 126, 234, 0.08)",
    padding: "16px 24px",
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
  "& .MuiChip-icon": {
    fontSize: "14px",
  },
}));

const ActionButton = styled(IconButton)(({ variant, theme }) => ({
  width: "36px",
  height: "36px",
  margin: "0 4px",
  borderRadius: "8px",
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
    transform: "translateY(-1px)",
  },
  "&:disabled": {
    backgroundColor: "rgba(0, 0, 0, 0.04)",
    color: "rgba(0, 0, 0, 0.26)",
    border: "1px solid rgba(0, 0, 0, 0.12)",
  },
}));

const TrainButton = styled(Button)(({ theme, training }) => ({
  textTransform: "none",
  fontSize: "0.875rem",
  fontWeight: "600",
  padding: "8px 16px",
  borderRadius: "8px",
  minWidth: "100px",
  backgroundColor: training ? "rgba(102, 126, 234, 0.08)" : "#667eea",
  color: training ? "#667eea" : "white",
  border: training ? "1px solid rgba(102, 126, 234, 0.3)" : "none",
  "&:hover": {
    backgroundColor: training ? "rgba(102, 126, 234, 0.12)" : "#5a67d8",
    transform: training ? "none" : "translateY(-1px)",
    boxShadow: training ? "none" : "0 4px 12px rgba(102, 126, 234, 0.3)",
  },
  "&:disabled": {
    backgroundColor: "rgba(0, 0, 0, 0.04)",
    color: "rgba(0, 0, 0, 0.26)",
  },
}));

const SelectAllButton = styled(Button)(({ theme, selected }) => ({
  textTransform: "none",
  fontSize: "0.875rem",
  fontWeight: "600",
  padding: "8px 16px",
  borderRadius: "8px",
  backgroundColor: selected ? "rgba(244, 67, 54, 0.08)" : "rgba(102, 126, 234, 0.08)",
  color: selected ? "#f44336" : "#667eea",
  border: `1px solid ${selected ? "rgba(244, 67, 54, 0.3)" : "rgba(102, 126, 234, 0.3)"}`,
  "&:hover": {
    backgroundColor: selected ? "rgba(244, 67, 54, 0.12)" : "rgba(102, 126, 234, 0.12)",
  },
}));

const ImageAvatar = styled(Avatar)(({ theme }) => ({
  width: "48px",
  height: "48px",
  borderRadius: "12px",
  border: "2px solid rgba(102, 126, 234, 0.2)",
  marginRight: "16px",
}));

const StatsContainer = styled(Box)({
  display: "flex",
  gap: "24px",
  marginBottom: "24px",
  flexWrap: "wrap",
});

const StatCard = styled(Card)(({ theme }) => ({
  padding: "20px",
  minWidth: "140px",
  textAlign: "center",
  border: "1px solid rgba(102, 126, 234, 0.1)",
  borderRadius: "12px",
  backgroundColor: "rgba(102, 126, 234, 0.02)",
}));

const StatValue = styled(Typography)({
  fontSize: "2rem",
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

export default function DataTable() {
  const { palette } = useTheme();
  
  const [datasets, setDatasets] = useState([]);
  const [trainingInProgress, setTrainingInProgress] = useState(null);
  const [progress, setProgress] = useState(0);
  const [modalOpen, setModalOpen] = useState(false);
  const [currentPieceLabel, setCurrentPieceLabel] = useState("");
  const [selectedDatasets, setSelectedDatasets] = useState([]);
  const [selectAll, setSelectAll] = useState(false);
  const [confirmationOpen, setConfirmationOpen] = useState(false);
  const [actionType, setActionType] = useState("");

  useEffect(() => {
    axios.get("http://localhost:8000/piece/datasets")
      .then((response) => {
        setDatasets(Object.values(response.data));
      })
      .catch((error) => {
        console.error("Error fetching datasets:", error);
      });
  }, []);

  const handleTrain = (pieceLabel) => {
    setTrainingInProgress(pieceLabel);
    setCurrentPieceLabel(pieceLabel);
    setModalOpen(true);

    axios.post(`http://localhost:8000/detection/train/${pieceLabel}`)
      .then((response) => {
        const interval = setInterval(() => {
          setProgress((prevProgress) => {
            if (prevProgress === 100) {
              clearInterval(interval);
              setTrainingInProgress(null);
              setProgress(0);
              setModalOpen(false);
              return 100;
            }
            return Math.min(prevProgress + 10, 100);
          });
        }, 1000);
      })
      .catch((error) => {
        console.error("Error starting training:", error);
        setTrainingInProgress(null);
        setProgress(0);
        setModalOpen(false);
      });
  };

  const handleSelectAll = () => {
    if (selectAll) {
      setSelectedDatasets([]);
      setSelectAll(false);
    } else {
      setSelectedDatasets(datasets.map(dataset => dataset.label));
      setSelectAll(true);
    }
  };

  const handleSelect = (label) => {
    setSelectedDatasets(prevSelected => 
      prevSelected.includes(label) 
        ? prevSelected.filter(item => item !== label) 
        : [...prevSelected, label]
    );
  };

  const handleView = (label) => {
    console.log("handleView", label);
  };

  const handleDelete = (label) => {
    axios.delete(`http://localhost:8000/piece/delete_piece/${label}`)
      .then(() => {
        setDatasets(prevDatasets => prevDatasets.filter(dataset => dataset.label !== label));
        setSelectedDatasets(prevSelected => prevSelected.filter(item => item !== label));
        window.location.reload();
      })
      .catch((error) => {
        console.error("Error deleting piece:", error);
      });
  };
  
  const handleConfirmationClose = (confirm) => {
    setConfirmationOpen(false);
    if (confirm) {
      if (actionType === "delete") {
        if (selectAll) {
          handleDeleteAll();
        } else {
          selectedDatasets.forEach(label => handleDelete(label));
        }
      }
      setSelectAll(false);
    }
  };
  
  const handleDeleteAll = () => {
    axios.delete("http://localhost:8000/piece/delete_all_pieces")
      .then(() => {
        setDatasets([]);
        setSelectedDatasets([]);
      })
      .catch((error) => {
        console.error("Error deleting all pieces:", error);
      });
  };

  const handleBulkDelete = () => {
    setActionType("delete");
    setConfirmationOpen(true);
  };

  // Group datasets by their common identifier
  const groupedDatasets = datasets.reduce((groups, dataset) => {
    const groupLabel = dataset.label.split(".").slice(0, 2).join(".");
    if (!groups[groupLabel]) {
      groups[groupLabel] = { label: groupLabel, pieces: [] };
    }
    groups[groupLabel].pieces.push(dataset);
    return groups;
  }, {});

  // Calculate statistics
  const totalPieces = datasets.length;
  const annotatedPieces = datasets.filter(d => d.is_annotated).length;
  const trainedPieces = datasets.filter(d => d.is_yolo_trained).length;
  const totalImages = datasets.reduce((sum, d) => sum + d.images.length, 0);

  return (
    <Container>
      <HeaderBox>
        <Title>Dataset Management</Title>
        <Box sx={{ display: 'flex', gap: 2 }}>
          <SelectAllButton 
            onClick={handleSelectAll}
            selected={selectAll}
          >
            {selectAll ? "Deselect All" : "Select All"}
          </SelectAllButton>
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

      <StatsContainer>
        <StatCard>
          <StatValue>{totalPieces}</StatValue>
          <StatLabel>Total Pieces</StatLabel>
        </StatCard>
        <StatCard>
          <StatValue>{totalImages}</StatValue>
          <StatLabel>Total Images</StatLabel>
        </StatCard>
        <StatCard>
          <StatValue>{annotatedPieces}</StatValue>
          <StatLabel>Annotated</StatLabel>
        </StatCard>
        <StatCard>
          <StatValue>{trainedPieces}</StatValue>
          <StatLabel>Trained</StatLabel>
        </StatCard>
      </StatsContainer>

      {Object.values(groupedDatasets).map((group, index) => (
        <ModernCard key={index} elevation={0}>
          <GroupHeader>
            <GroupTitle>
              <PhotoLibrary sx={{ fontSize: 20 }} />
              {group.label}
              <Typography variant="caption" sx={{ 
                ml: 2, 
                px: 1.5, 
                py: 0.5, 
                backgroundColor: "rgba(102, 126, 234, 0.1)",
                borderRadius: "12px",
                color: "#667eea",
                fontWeight: "600"
              }}>
                {group.pieces.length} pieces
              </Typography>
            </GroupTitle>
          </GroupHeader>
          
          <ProductTable>
            <TableHead>
              <TableRow>
                <TableCell width="5%">Select</TableCell>
                <TableCell width="35%">Piece Details</TableCell>
                <TableCell width="15%" align="center">Images</TableCell>
                <TableCell width="15%" align="center">Annotation</TableCell>
                <TableCell width="15%" align="center">Training</TableCell>
                <TableCell width="15%" align="center">Actions</TableCell>
              </TableRow>
            </TableHead>

            <TableBody>
              {group.pieces.map((piece) => (
                <TableRow key={piece.id} hover>
                  <TableCell>
                    <Checkbox
                      checked={selectedDatasets.includes(piece.label)}
                      onChange={() => handleSelect(piece.label)}
                      sx={{ 
                        color: "#667eea",
                        '&.Mui-checked': { color: "#667eea" }
                      }}
                    />
                  </TableCell>
                  
                  <TableCell>
                    <Box display="flex" alignItems="center">
                      <ImageAvatar 
                        src={piece.images[0]?.url} 
                        variant="rounded"
                      >
                        <PhotoLibrary />
                      </ImageAvatar>
                      <Box>
                        <Typography variant="body1" fontWeight="600" color="#333">
                          {piece.label}
                        </Typography>
                        <Typography variant="caption" color="text.secondary">
                          ID: {piece.id}
                        </Typography>
                      </Box>
                    </Box>
                  </TableCell>
                  
                  <TableCell align="center">
                    <Typography variant="h6" color="#667eea" fontWeight="700">
                      {piece.images.length}
                    </Typography>
                  </TableCell>
                  
                  <TableCell align="center">
                    <StatusChip 
                      variant={piece.is_annotated ? "completed" : "pending"}
                      icon={piece.is_annotated ? <CheckCircle /> : <RadioButtonUnchecked />}
                      label={piece.is_annotated ? "Completed" : "Pending"}
                    />
                  </TableCell>
                  
                  <TableCell align="center">
                    <StatusChip 
                      variant={piece.is_yolo_trained ? "trained" : "pending"}
                      icon={piece.is_yolo_trained ? <CheckCircle /> : <RadioButtonUnchecked />}
                      label={piece.is_yolo_trained ? "Trained" : "Not Trained"}
                    />
                  </TableCell>
                  
                  <TableCell align="center">
                    <Box display="flex" justifyContent="center" alignItems="center" gap={1}>
                      <ActionButton variant="view" onClick={() => handleView(piece.label)}>
                        <Visibility fontSize="small" />
                      </ActionButton>
                      
                      <ActionButton variant="delete" onClick={() => handleDelete(piece.label)}>
                        <Delete fontSize="small" />
                      </ActionButton>
                      
                      <TrainButton 
                        onClick={() => handleTrain(piece.label)} 
                        training={trainingInProgress === piece.label}
                        disabled={trainingInProgress === piece.label}
                        size="small"
                      >
                        {trainingInProgress === piece.label ? "Training..." : "Train"}
                      </TrainButton>
                    </Box>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </ProductTable>
        </ModernCard>
      ))}

      <TrainingProgressModal 
        open={modalOpen}
        onClose={() => setModalOpen(false)}
        progress={progress}
      />

      <Dialog
        open={confirmationOpen}
        onClose={() => setConfirmationOpen(false)}
        PaperProps={{
          sx: {
            borderRadius: "16px",
            padding: "8px",
          }
        }}
      >
        <DialogTitle sx={{ fontWeight: "600", color: "#333" }}>
          Delete Selected Items
        </DialogTitle>
        <DialogContent>
          <Typography>
            Are you sure you want to delete {selectAll ? "all items" : `${selectedDatasets.length} selected item${selectedDatasets.length !== 1 ? 's' : ''}`}? 
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