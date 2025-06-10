import React, { useEffect, useState } from "react";
import { 
  CropFree, 
  PhotoLibrary, 
  CheckCircle 
} from "@mui/icons-material";
import { 
  Card, 
  Grid, 
  styled, 
  Box, 
  Typography,
  Chip,
  CircularProgress
} from "@mui/material";
import api from "../../utils/UseAxios";

// STYLED COMPONENTS - Updated to match capture theme
const ContentBox = styled("div")(() => ({
  display: "flex",
  flexWrap: "wrap",
  alignItems: "center",
  width: "100%",
}));

const PieceCard = styled(Card)(({ theme, selected }) => ({
  padding: "20px",
  cursor: "pointer",
  minWidth: "280px",
  height: "140px",
  display: "flex",
  flexDirection: "column",
  justifyContent: "space-between",
  backgroundColor: selected ? "#f0f4ff" : "#f5f5f5",
  border: selected ? "3px solid #667eea" : "2px solid rgba(102, 126, 234, 0.2)",
  borderRadius: "12px",
  transition: "all 0.3s cubic-bezier(0.4, 0, 0.2, 1)",
  position: "relative",
  overflow: "hidden",
  marginRight: "16px",
  flexShrink: 0,
  boxShadow: selected 
    ? "0 8px 25px rgba(102, 126, 234, 0.3)" 
    : "0 2px 8px rgba(0, 0, 0, 0.1)",
  "&:hover": {
    transform: "translateY(-2px)",
    boxShadow: "0 8px 25px rgba(102, 126, 234, 0.25)",
    border: "3px solid #667eea",
    backgroundColor: "#f0f4ff",
  },
  "&:last-child": {
    marginRight: 0,
  },
}));

const IconContainer = styled(Box)(({ theme }) => ({
  width: "48px",
  height: "48px",
  borderRadius: "12px",
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  backgroundColor: "rgba(102, 126, 234, 0.1)",
  color: "#667eea",
  marginBottom: "12px",
}));

const PieceTitle = styled(Typography)({
  fontSize: "1.1rem",
  fontWeight: "600",
  color: "#333",
  marginBottom: "8px",
  overflow: "hidden",
  textOverflow: "ellipsis",
  whiteSpace: "nowrap",
});

const StatsContainer = styled(Box)({
  display: "flex",
  alignItems: "center",
  justifyContent: "space-between",
  marginTop: "auto",
});

const ImagePreview = styled("img")({
  width: "40px",
  height: "40px",
  objectFit: "cover",
  borderRadius: "6px",
  border: "2px solid rgba(102, 126, 234, 0.2)",
});

const ImageCount = styled(Chip)(({ theme }) => ({
  backgroundColor: "rgba(102, 126, 234, 0.1)",
  color: "#667eea",
  fontSize: "0.75rem",
  fontWeight: "600",
  height: "24px",
  "& .MuiChip-icon": {
    color: "#667eea",
    fontSize: "16px",
  },
}));

const LoadingContainer = styled(Box)({
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  minHeight: "140px",
  minWidth: "280px",
  flexDirection: "column",
  gap: 2,
  color: "#666",
});

const EmptyState = styled(Box)({
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  minHeight: "140px",
  minWidth: "280px",
  flexDirection: "column",
  gap: 2,
  color: "#666",
  textAlign: "center",
});

export default function NonAnnotated({ onPieceSelect }) {
  const [pieces, setPieces] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selectedPiece, setSelectedPiece] = useState('');

  useEffect(() => {
    async function fetchPieces() {
      try {
        setLoading(true);
        const response = await api.get("/api/annotation/annotations/get_Img_nonAnnotated");
        setPieces(response.data);
        
        // Auto-select the first piece if available
        if (response.data.length > 0) {
          const firstPiece = response.data[0].piece_label;
          setSelectedPiece(firstPiece);
          onPieceSelect(firstPiece);
        }
      } catch (error) {
        console.error("Error fetching pieces:", error.response?.data?.detail || error.message);
      } finally {
        setLoading(false);
      }
    }

    fetchPieces();
  }, [onPieceSelect]);

  const handlePieceClick = (pieceLabel) => {
    setSelectedPiece(pieceLabel);
    onPieceSelect(pieceLabel);
  };

  if (loading) {
    return (
      <Grid container spacing={3} sx={{ mb: 3, flexWrap: 'nowrap' }}>
        <Grid item>
          <LoadingContainer>
            <CircularProgress sx={{ color: '#667eea' }} size={32} />
            <Typography variant="body2" sx={{ opacity: 0.8 }}>
              Loading pieces...
            </Typography>
          </LoadingContainer>
        </Grid>
      </Grid>
    );
  }

  if (pieces.length === 0) {
    return (
      <Grid container spacing={3} sx={{ mb: 3, flexWrap: 'nowrap' }}>
        <Grid item>
          <EmptyState>
            <CropFree sx={{ fontSize: 48, opacity: 0.4, mb: 1 }} />
            <Typography variant="body1" sx={{ opacity: 0.9, mb: 0.5 }}>
              No Pieces Available
            </Typography>
            <Typography variant="body2" sx={{ opacity: 0.7 }}>
              All pieces have been annotated
            </Typography>
          </EmptyState>
        </Grid>
      </Grid>
    );
  }

  return (
    <Grid container spacing={3} sx={{ mb: 3, flexWrap: 'nowrap' }}>
      {pieces.map((piece) => (
        <Grid item key={piece.piece_label}>
          <PieceCard 
            elevation={0}
            selected={selectedPiece === piece.piece_label}
            onClick={() => handlePieceClick(piece.piece_label)}
          >
            <ContentBox>
              <IconContainer>
                <CropFree fontSize="medium" />
              </IconContainer>
              
              <PieceTitle>
                {piece.piece_label}
              </PieceTitle>
            </ContentBox>
            
            <StatsContainer>
              <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                {piece.url && (
                  <ImagePreview
                    src={piece.url}
                    alt={piece.piece_label}
                    onError={(e) => {
                      e.target.style.display = 'none';
                    }}
                  />
                )}
                
                <ImageCount
                  icon={<PhotoLibrary />}
                  label={`${piece.nbr_img} images`}
                  size="small"
                />
              </Box>
              
              {piece.annotated_count > 0 && (
                <Chip
                  icon={<CheckCircle />}
                  label={`${piece.annotated_count} done`}
                  size="small"
                  sx={{
                    backgroundColor: "rgba(76, 175, 80, 0.1)",
                    color: "#4caf50",
                    fontSize: "0.75rem",
                    fontWeight: "600",
                    height: "24px",
                    "& .MuiChip-icon": {
                      color: "#4caf50",
                      fontSize: "16px",
                    },
                  }}
                />
              )}
            </StatsContainer>
          </PieceCard>
        </Grid>
      ))}
    </Grid>
  );
}