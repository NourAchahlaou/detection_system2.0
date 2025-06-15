import React, { useState, useEffect } from "react";
import { 
  Box, 
  Typography, 
  Tabs, 
  Tab, 
  styled, 
  Grid,
  Card,
  Chip,
  CircularProgress,
  Button
} from "@mui/material";
import { 
  CropFree, 
  PhotoLibrary, 
  CheckCircle,
  RadioButtonUnchecked,
  Visibility,
  Edit
} from "@mui/icons-material";
import { useNavigate } from "react-router-dom";
import api from "../../utils/UseAxios";

// STYLED COMPONENTS - Following your theme
const Container = styled("div")(({ theme }) => ({
  margin: "30px",
  [theme.breakpoints.down("sm")]: { margin: "16px" },
}));

const HeaderBox = styled(Box)({
  paddingBottom: "24px",
  borderBottom: "1px solid rgba(102, 126, 234, 0.1)",
  marginBottom: "24px",
});

const HeaderTitle = styled(Typography)({
  fontSize: "1.8rem",
  fontWeight: "700",
  color: "#333",
  marginBottom: "8px",
});

const HeaderSubtitle = styled(Typography)({
  fontSize: "1rem",
  color: "#666",
  fontWeight: "400",
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

const PieceCard = styled(Card)(({ theme }) => ({
  padding: "24px",
  cursor: "pointer",
  minHeight: "200px",
  display: "flex",
  flexDirection: "column",
  justifyContent: "space-between",
  backgroundColor: "#f8f9ff",
  border: "2px solid rgba(102, 126, 234, 0.1)",
  borderRadius: "16px",
  transition: "all 0.3s cubic-bezier(0.4, 0, 0.2, 1)",
  position: "relative",
  overflow: "hidden",
  boxShadow: "0 2px 12px rgba(0, 0, 0, 0.08)",
  "&:hover": {
    transform: "translateY(-4px)",
    boxShadow: "0 12px 32px rgba(102, 126, 234, 0.2)",
    border: "2px solid #667eea",
    backgroundColor: "#f0f4ff",
  },
}));

const IconContainer = styled(Box)({
  width: "56px",
  height: "56px",
  borderRadius: "16px",
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  backgroundColor: "rgba(102, 126, 234, 0.15)",
  color: "#667eea",
  marginBottom: "16px",
});

const PieceTitle = styled(Typography)({
  fontSize: "1.2rem",
  fontWeight: "700",
  color: "#333",
  marginBottom: "12px",
  overflow: "hidden",
  textOverflow: "ellipsis",
  whiteSpace: "nowrap",
});

const StatsContainer = styled(Box)({
  display: "flex",
  flexDirection: "column",
  gap: "12px",
  marginTop: "auto",
});

const StatsRow = styled(Box)({
  display: "flex",
  alignItems: "center",
  justifyContent: "space-between",
});

const ImagePreview = styled("img")({
  width: "48px",
  height: "48px",
  objectFit: "cover",
  borderRadius: "8px",
  border: "2px solid rgba(102, 126, 234, 0.2)",
});

const StatusChip = styled(Chip)(({ variant }) => ({
  fontSize: "0.75rem",
  fontWeight: "600",
  height: "28px",
  backgroundColor: variant === 'completed' 
    ? "rgba(76, 175, 80, 0.15)" 
    : variant === 'partial'
    ? "rgba(255, 152, 0, 0.15)"
    : "rgba(244, 67, 54, 0.15)",
  color: variant === 'completed' 
    ? "#4caf50" 
    : variant === 'partial'
    ? "#ff9800"
    : "#f44336",
  "& .MuiChip-icon": {
    fontSize: "16px",
  },
}));

const ActionButton = styled(Button)(({ variant }) => ({
  textTransform: "none",
  fontWeight: "600",
  borderRadius: "8px",
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

function TabPanel({ children, value, index }) {
  return (
    <div hidden={value !== index}>
      {value === index && children}
    </div>
  );
}

export default function PiecesOverview() {
  const [tabValue, setTabValue] = useState(0);
  const [nonAnnotatedPieces, setNonAnnotatedPieces] = useState([]);
  const [annotatedPieces, setAnnotatedPieces] = useState([]);
  const [loading, setLoading] = useState(true);
  const [stats, setStats] = useState({
    total: 0,
    annotated: 0,
    nonAnnotated: 0,
    partial: 0
  });
  
  const navigate = useNavigate();

  useEffect(() => {
    fetchAllPieces();
  }, []);

  const fetchAllPieces = async () => {
    try {
      setLoading(true);
      
      // Fetch non-annotated pieces
      const nonAnnotatedResponse = await api.get("/api/annotation/annotations/get_Img_nonAnnotated");
      const nonAnnotatedData = nonAnnotatedResponse.data;
      
      // Fetch all pieces to get annotated ones
      // This assumes you have an endpoint that returns all pieces
      // If not, you might need to create one or modify this logic
      const allPiecesResponse = await api.get("/api/annotation/annotations/get_all_pieces");
      const allPiecesData = allPiecesResponse.data || [];
      
      // Filter annotated pieces (pieces not in non-annotated list)
      const nonAnnotatedLabels = new Set(nonAnnotatedData.map(piece => piece.piece_label));
      const annotatedData = allPiecesData.filter(piece => !nonAnnotatedLabels.has(piece.piece_label));
      
      setNonAnnotatedPieces(nonAnnotatedData);
      setAnnotatedPieces(annotatedData);
      
      // Calculate stats
      const totalPieces = nonAnnotatedData.length + annotatedData.length;
      const partiallyAnnotated = nonAnnotatedData.filter(piece => piece.annotated_count > 0).length;
      
      setStats({
        total: totalPieces,
        annotated: annotatedData.length,
        nonAnnotated: nonAnnotatedData.length - partiallyAnnotated,
        partial: partiallyAnnotated
      });
      
    } catch (error) {
      console.error("Error fetching pieces:", error);
      // If the all pieces endpoint doesn't exist, just use non-annotated data
      if (error.response?.status === 404) {
        const nonAnnotatedResponse = await api.get("/api/annotation/annotations/get_Img_nonAnnotated");
        const nonAnnotatedData = nonAnnotatedResponse.data;
        
        setNonAnnotatedPieces(nonAnnotatedData);
        setAnnotatedPieces([]);
        
        const partiallyAnnotated = nonAnnotatedData.filter(piece => piece.annotated_count > 0).length;
        setStats({
          total: nonAnnotatedData.length,
          annotated: 0,
          nonAnnotated: nonAnnotatedData.length - partiallyAnnotated,
          partial: partiallyAnnotated
        });
      }
    } finally {
      setLoading(false);
    }
  };

  const handleTabChange = (event, newValue) => {
    setTabValue(newValue);
  };

  const handlePieceClick = (pieceLabel) => {
    // Navigate to annotation page with the piece pre-selected
    navigate(`/annotation?piece=${encodeURIComponent(pieceLabel)}`);
  };

  const getStatusInfo = (piece) => {
    if (piece.annotated_count === 0) {
      return {
        variant: 'not-started',
        label: 'Not Started',
        icon: <RadioButtonUnchecked />,
        progress: 0
      };
    } else if (piece.annotated_count < piece.nbr_img) {
      return {
        variant: 'partial',
        label: `${piece.annotated_count}/${piece.nbr_img} Done`,
        icon: <Edit />,
        progress: (piece.annotated_count / piece.nbr_img) * 100
      };
    } else {
      return {
        variant: 'completed',
        label: 'Completed',
        icon: <CheckCircle />,
        progress: 100
      };
    }
  };

  const renderPieceCard = (piece) => {
    const statusInfo = getStatusInfo(piece);
    
    return (
      <Grid item xs={12} sm={6} md={4} lg={3} key={piece.piece_label}>
        <PieceCard elevation={0} onClick={() => handlePieceClick(piece.piece_label)}>
          <Box>
            <IconContainer>
              <CropFree fontSize="large" />
            </IconContainer>
            
            <PieceTitle>{piece.piece_label}</PieceTitle>
            
            <StatsContainer>
              <StatsRow>
                <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                  <PhotoLibrary sx={{ fontSize: 16, color: "#667eea" }} />
                  <Typography variant="body2" sx={{ color: "#666", fontWeight: "500" }}>
                    {piece.nbr_img} images
                  </Typography>
                </Box>
                
                {piece.url && (
                  <ImagePreview
                    src={piece.url}
                    alt={piece.piece_label}
                    onError={(e) => {
                      e.target.style.display = 'none';
                    }}
                  />
                )}
              </StatsRow>
              
              <StatsRow>
                <StatusChip
                  variant={statusInfo.variant}
                  icon={statusInfo.icon}
                  label={statusInfo.label}
                  size="small"
                />
                
                <ActionButton
                  variant={statusInfo.variant === 'completed' ? 'secondary' : 'primary'}
                  size="small"
                  startIcon={statusInfo.variant === 'completed' ? <Visibility /> : <Edit />}
                >
                  {statusInfo.variant === 'completed' ? 'View' : 'Annotate'}
                </ActionButton>
              </StatsRow>
            </StatsContainer>
          </Box>
        </PieceCard>
      </Grid>
    );
  };

  if (loading) {
    return (
      <Container>
        <LoadingContainer>
          <CircularProgress sx={{ color: '#667eea' }} size={48} />
          <Typography variant="h6" sx={{ opacity: 0.8, mt: 2 }}>
            Loading pieces...
          </Typography>
        </LoadingContainer>
      </Container>
    );
  }

  return (
    <Container>
      <HeaderBox>
        <HeaderTitle>Pieces Management</HeaderTitle>
        <HeaderSubtitle>
          Manage and track annotation progress across all pieces
        </HeaderSubtitle>
        
        {/* Stats Summary */}
        <Box sx={{ 
          display: 'flex', 
          gap: 3, 
          mt: 3, 
          flexWrap: 'wrap',
          justifyContent: { xs: 'center', sm: 'flex-start' }
        }}>
          <Box sx={{ textAlign: 'center' }}>
            <Typography variant="h4" sx={{ color: '#667eea', fontWeight: '700' }}>
              {stats.total}
            </Typography>
            <Typography variant="caption" sx={{ color: '#666', textTransform: 'uppercase', letterSpacing: 1 }}>
              Total Pieces
            </Typography>
          </Box>
          <Box sx={{ textAlign: 'center' }}>
            <Typography variant="h4" sx={{ color: '#4caf50', fontWeight: '700' }}>
              {stats.annotated}
            </Typography>
            <Typography variant="caption" sx={{ color: '#666', textTransform: 'uppercase', letterSpacing: 1 }}>
              Completed
            </Typography>
          </Box>
          <Box sx={{ textAlign: 'center' }}>
            <Typography variant="h4" sx={{ color: '#ff9800', fontWeight: '700' }}>
              {stats.partial}
            </Typography>
            <Typography variant="caption" sx={{ color: '#666', textTransform: 'uppercase', letterSpacing: 1 }}>
              In Progress
            </Typography>
          </Box>
          <Box sx={{ textAlign: 'center' }}>
            <Typography variant="h4" sx={{ color: '#f44336', fontWeight: '700' }}>
              {stats.nonAnnotated}
            </Typography>
            <Typography variant="caption" sx={{ color: '#666', textTransform: 'uppercase', letterSpacing: 1 }}>
              Not Started
            </Typography>
          </Box>
        </Box>
      </HeaderBox>

      <StyledTabs value={tabValue} onChange={handleTabChange}>
        <Tab label={`All Pieces (${stats.total})`} />
        <Tab label={`Need Annotation (${stats.nonAnnotated + stats.partial})`} />
        <Tab label={`Completed (${stats.annotated})`} />
      </StyledTabs>

      <TabPanel value={tabValue} index={0}>
        {/* All Pieces */}
        <Grid container spacing={3}>
          {[...nonAnnotatedPieces, ...annotatedPieces].map(renderPieceCard)}
        </Grid>
        
        {[...nonAnnotatedPieces, ...annotatedPieces].length === 0 && (
          <EmptyState>
            <CropFree sx={{ fontSize: 64, opacity: 0.4, mb: 2 }} />
            <Typography variant="h6" sx={{ opacity: 0.9, mb: 1 }}>
              No Pieces Found
            </Typography>
            <Typography variant="body2" sx={{ opacity: 0.7 }}>
              No pieces are available in the system
            </Typography>
          </EmptyState>
        )}
      </TabPanel>

      <TabPanel value={tabValue} index={1}>
        {/* Need Annotation */}
        <Grid container spacing={3}>
          {nonAnnotatedPieces.map(renderPieceCard)}
        </Grid>
        
        {nonAnnotatedPieces.length === 0 && (
          <EmptyState>
            <CheckCircle sx={{ fontSize: 64, opacity: 0.4, mb: 2, color: '#4caf50' }} />
            <Typography variant="h6" sx={{ opacity: 0.9, mb: 1 }}>
              All Pieces Completed!
            </Typography>
            <Typography variant="body2" sx={{ opacity: 0.7 }}>
              All pieces have been fully annotated
            </Typography>
          </EmptyState>
        )}
      </TabPanel>

      <TabPanel value={tabValue} index={2}>
        {/* Completed */}
        <Grid container spacing={3}>
          {annotatedPieces.map(renderPieceCard)}
        </Grid>
        
        {annotatedPieces.length === 0 && (
          <EmptyState>
            <RadioButtonUnchecked sx={{ fontSize: 64, opacity: 0.4, mb: 2 }} />
            <Typography variant="h6" sx={{ opacity: 0.9, mb: 1 }}>
              No Completed Pieces
            </Typography>
            <Typography variant="body2" sx={{ opacity: 0.7 }}>
              No pieces have been fully annotated yet
            </Typography>
          </EmptyState>
        )}
      </TabPanel>
    </Container>
  );
}