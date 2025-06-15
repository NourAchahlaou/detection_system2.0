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

// UPDATED: Custom Grid Container with CSS Grid for precise control
const CardsGridContainer = styled('div')(({ theme }) => ({
  display: 'grid',
  gap: '20px',
  width: '100%',
  
  // Mobile first approach
  gridTemplateColumns: '1fr',
  
  // Small screens (tablets) - 2 cards per row
  [theme.breakpoints.up('sm')]: {
    gridTemplateColumns: 'repeat(2, 1fr)',
  },
  
  // Medium screens - 3 cards per row
  [theme.breakpoints.up('md')]: {
    gridTemplateColumns: 'repeat(3, 1fr)',
  },
  
  // Large screens - 4 cards per row (this is what you want)
  [theme.breakpoints.up('lg')]: {
    gridTemplateColumns: 'repeat(4, 1fr)',
  },
  
  // Extra large screens - 5 cards per row
  [theme.breakpoints.up('xl')]: {
    gridTemplateColumns: 'repeat(5, 1fr)',
  },
  
  // Fallback for very large screens using pixel values
  '@media (min-width: 1200px)': {
    gridTemplateColumns: 'repeat(4, 1fr)',
  },
  
  '@media (min-width: 1536px)': {
    gridTemplateColumns: 'repeat(5, 1fr)',
  },
}));

const PieceCard = styled(Card)(({ theme }) => ({
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
  minWidth: "0", // Important for text truncation
  "&:hover": {
    transform: "translateY(-4px)",
    boxShadow: "0 12px 32px rgba(102, 126, 234, 0.2)",
    border: "2px solid #667eea",
    backgroundColor: "rgba(78, 105, 221, 0.45)",
  },
}));

// Alternative approach: Custom Grid Item with explicit sizing
const CustomGridItem = styled('div')(({ theme }) => ({
  width: '100%',
  
  // Use CSS Grid's auto-fit with minmax for responsive behavior
  '@media (max-width: 599px)': {
    width: '100%',
  },
  '@media (min-width: 600px) and (max-width: 899px)': {
    width: '100%',
  },
  '@media (min-width: 900px) and (max-width: 1199px)': {
    width: '100%',
  },
  '@media (min-width: 1200px)': {
    width: '100%',
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

const PieceTitle = styled(Typography)({
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

const ImagePreview = styled("img")({
  width: "40px",
  height: "40px",
  objectFit: "cover",
  borderRadius: "6px",
  border: "2px solid rgba(102, 126, 234, 0.2)",
});

const StatusChip = styled(Chip)(({ variant }) => ({
  fontSize: "0.75rem",
  fontWeight: "600",
  height: "24px",
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

function TabPanel({ children, value, index }) {
  return (
    <div hidden={value !== index}>
      {value === index && children}
    </div>
  );
}

export default function PiecesOverview() {
  const [tabValue, setTabValue] = useState(0);
  const [allPieces, setAllPieces] = useState([]);
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
      
      const response = await api.get("/api/annotation/annotations/get_all_pieces");
      const piecesData = response.data || [];
      
      console.log("Fetched pieces data:", piecesData);
      
      setAllPieces(piecesData);
      
      const totalPieces = piecesData.length;
      const fullyAnnotatedPieces = piecesData.filter(piece => piece.is_fully_annotated).length;
      const partiallyAnnotatedPieces = piecesData.filter(piece => 
        piece.annotated_count > 0 && !piece.is_fully_annotated
      ).length;
      const notStartedPieces = piecesData.filter(piece => piece.annotated_count === 0).length;
      
      setStats({
        total: totalPieces,
        annotated: fullyAnnotatedPieces,
        partial: partiallyAnnotatedPieces,
        nonAnnotated: notStartedPieces
      });
      
    } catch (error) {
      console.error("Error fetching pieces:", error);
      
      try {
        console.log("Trying fallback endpoint...");
        const fallbackResponse = await api.get("/api/annotation/annotations/get_Img_nonAnnotated");
        const nonAnnotatedData = fallbackResponse.data || [];
        
        console.log("Fallback data:", nonAnnotatedData);
        
        const convertedData = nonAnnotatedData.map(piece => ({
          piece_label: piece.piece_label,
          nbr_img: piece.nbr_img,
          annotated_count: piece.annotated_count || 0,
          url: piece.url,
          is_fully_annotated: false
        }));
        
        setAllPieces(convertedData);
        
        const partiallyAnnotated = convertedData.filter(piece => piece.annotated_count > 0).length;
        setStats({
          total: convertedData.length,
          annotated: 0,
          partial: partiallyAnnotated,
          nonAnnotated: convertedData.length - partiallyAnnotated
        });
      } catch (fallbackError) {
        console.error("Fallback fetch also failed:", fallbackError);
        setAllPieces([]);
        setStats({ total: 0, annotated: 0, partial: 0, nonAnnotated: 0 });
      }
    } finally {
      setLoading(false);
    }
  };

  const handleTabChange = (event, newValue) => {
    setTabValue(newValue);
  };

  const handlePieceClick = (pieceLabel) => {
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

  const renderPieceCard = (piece, index) => {
    const statusInfo = getStatusInfo(piece);
    
    return (
      <CustomGridItem key={piece.piece_label}>
        <PieceCard elevation={0} onClick={() => handlePieceClick(piece.piece_label)}>
          <CardHeader>
            <Box sx={{ display: "flex", alignItems: "center", gap: 1, flex: 1, minWidth: 0 }}>
              <IconContainer>
                <CropFree fontSize="medium" />
              </IconContainer>
              <Box sx={{ minWidth: 0, flex: 1 }}>
                <PieceTitle title={piece.piece_label}>{piece.piece_label}</PieceTitle>
                <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                  <PhotoLibrary sx={{ fontSize: 14, color: "#667eea" }} />
                  <Typography variant="caption" sx={{ color: "#666", fontWeight: "500" }}>
                    {piece.nbr_img} images
                  </Typography>
                </Box>
              </Box>
            </Box>
            
            {piece.url && (
              <ImagePreview
                src={piece.url}
                alt={piece.piece_label}
                onError={(e) => {
                  console.log(`Failed to load image: ${piece.url}`);
                  e.target.style.display = 'none';
                }}
                onLoad={() => {
                  console.log(`Successfully loaded image: ${piece.url}`);
                }}
              />
            )}
          </CardHeader>
          
          <StatsContainer>
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
        </PieceCard>
      </CustomGridItem>
    );
  };

  const getFilteredPieces = () => {
    switch (tabValue) {
      case 0:
        return allPieces;
      case 1:
        return allPieces.filter(piece => !piece.is_fully_annotated);
      case 2:
        return allPieces.filter(piece => piece.is_fully_annotated);
      default:
        return allPieces;
    }
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

  const filteredPieces = getFilteredPieces();

  return (
    <Container>
      <HeaderBox>       
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
        <CardsGridContainer>
          {filteredPieces.map(renderPieceCard)}
        </CardsGridContainer>
        
        {filteredPieces.length === 0 && (
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
        <CardsGridContainer>
          {filteredPieces.map(renderPieceCard)}
        </CardsGridContainer>
        
        {filteredPieces.length === 0 && (
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
        <CardsGridContainer>
          {filteredPieces.map(renderPieceCard)}
        </CardsGridContainer>
        
        {filteredPieces.length === 0 && (
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