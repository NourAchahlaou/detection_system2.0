import React, {  useState, useEffect } from "react";
import { Card, Grid, Box, styled, Stack, Typography } from "@mui/material";
import NonAnnotated from "./NonAnnotated";
import SidenavImageDisplay from "./annotation/components/SidenavImageDisplay";
import Simple from "./Simple";
import api from "../../utils/UseAxios";
import { useNavigate } from "react-router-dom";

// STYLED COMPONENTS - Updated to match capture template exactly
const Container = styled("div")(({ theme }) => ({
  margin: "30px",
  [theme.breakpoints.down("sm")]: { margin: "16px" },
  "& .breadcrumb": {
    marginBottom: "30px",
    [theme.breakpoints.down("sm")]: { marginBottom: "16px" },
  },
}));



// Updated to match VideoCard styling exactly
const AnnotationCard = styled(Card)(({ theme }) => ({
  width: "900px", // Match VideoCard width exactly
  height: "480px", // Match VideoCard height exactly
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  backgroundColor: "#f5f5f5",
  border: "2px solid #667eea", // Always active border like camera feed
  borderRadius: "12px",
  position: "relative",
  overflow: "hidden",
  padding: 0,
  margin: 0,
  // Override any Material-UI Card default styles
  '& .MuiCard-root': {
    padding: 0,
  },
  [theme.breakpoints.down("md")]: {
    width: "100%",
    maxWidth: "700px", // Match VideoCard responsive behavior
  },
  [theme.breakpoints.down("sm")]: {
    height: "300px", // Match VideoCard mobile height
  },
}));

// Header styles moved from SidenavImageDisplay
const HeaderBox = styled(Box)({
  paddingTop: "16px",
  borderBottom: "2px solid rgba(102, 126, 234, 0.1)",
  
});

const HeaderTitle = styled(Typography)({
  fontSize: "1.1rem",
  fontWeight: "600",
  color: "#333",
  textAlign: "center",
});

// New styled component for centering content
const CenteredContainer = styled(Box)({
  display: "flex",
  flexDirection: "column",
  alignItems: "center",
  justifyContent: "center",
  height: "100%",
  minHeight: "600px", // Ensure minimum height for proper centering
});

// Sidebar container with proper centering
const SidebarContainer = styled(Box)({
  display: "flex",
  alignItems: "center",
  justifyContent: "center", 
  height: "100%",
  minHeight: "600px", // Match the annotation card area height
  padding: "0 16px",
});

export default function AppImageAnnotaion() {
  const [selectedPieceLabel, setSelectedPieceLabel] = useState('');
  const [selectedImageUrl, setSelectedImageUrl] = useState('');
  const [initialPiece, setInitialPiece] = useState(null);
  const [annotatedImages, setAnnotatedImages] = useState([]);
  const [totalImages, setTotalImages] = useState(0);
  const navigate = useNavigate();

  const handlePieceSelect = (pieceLabel) => {
    setSelectedPieceLabel(pieceLabel);
  };

  const handleImageSelect = (url) => {
    setSelectedImageUrl(url);
  };

  const handleFirstImageLoad = (url) => {
    setSelectedImageUrl(url);
  };

  const handleImageCountUpdate = (count) => {
    setTotalImages(count);
  };

  useEffect(() => {
    async function fetchInitialPiece() {
      try {
        const response = await api.get("/api/annotation/annotations/get_Img_nonAnnotated");
        const pieces = response.data;
        if (pieces.length > 0) {
          const firstPiece = pieces[0];
          setInitialPiece(firstPiece.piece_label);
          setSelectedPieceLabel(firstPiece.piece_label);
        } else {
          navigate("/204");
        }
      } catch (error) {
        console.error("Error fetching initial piece:", error.response?.data?.detail || error.message);
        navigate("/204");
      }
    }

    fetchInitialPiece();
  }, [navigate]);

  const handleAnnotationSubmit = async () => {
    try {
      await api.post("/api/annotation/annotations", { imageUrl: selectedImageUrl });
      setAnnotatedImages((prev) => [...prev, selectedImageUrl]);
    } catch (error) {
      console.error("Error saving annotation:", error.response?.data?.detail || error.message);
    }
  };

  return (
    <Box sx={{ width: '100%', maxWidth: { sm: '100%', md: '1700px' }, margin: '0 auto' }}>
      {/* Header Section - Keep outside the main grid */}
      <HeaderBox>
        <HeaderTitle>
          {selectedPieceLabel ? `${selectedPieceLabel} Images` : "Select a Piece"}
        </HeaderTitle>
        {totalImages > 0 && (
          <Typography
            variant="body2"
            sx={{
              color: "#666",
              textAlign: "center",
              mt: 1,
              fontSize: "0.8rem",
            }}
          >
            {totalImages} image{totalImages !== 1 ? 's' : ''} • {annotatedImages.length} annotated
          </Typography>
        )}
      </HeaderBox>

      {/* Main Content Grid - Single Row Layout */}
      <Grid
        container
        spacing={2}
        columns={12}
        sx={{ 
          mb: (theme) => theme.spacing(2),
          minHeight: "600px", // Ensure consistent height
          alignItems: "center" // Center both grid items vertically
        }}
      >
        {/* Annotation Card Column */}
        <Grid size={{ xs: 12, md: 9 }}>
          <CenteredContainer>
            <AnnotationCard>
              <Simple 
                imageUrl={selectedImageUrl} 
                annotated={annotatedImages.includes(selectedImageUrl)} 
                pieceLabel={selectedPieceLabel}
              />
            </AnnotationCard>
          </CenteredContainer>
        </Grid>
        
        {/* Sidebar Column */}
        <Grid size={{ xs: 12, md: 3 }}>
          <SidebarContainer>
            <Box sx={{ width: "100%", height: "480px" }}> {/* Match annotation card height */}
              <SidenavImageDisplay
                pieceLabel={selectedPieceLabel}
                onImageSelect={handleImageSelect}
                onFirstImageLoad={handleFirstImageLoad}
                annotatedImages={annotatedImages}
                onImageCountUpdate={handleImageCountUpdate}
              />
            </Box>
          </SidebarContainer>
        </Grid>
      </Grid>
    </Box>
  );
}