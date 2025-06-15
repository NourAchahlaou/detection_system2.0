import React, { useState, useEffect, useRef } from "react";
import { Card, Grid, Box, styled, Stack, Typography } from "@mui/material";
import { CheckCircle, RadioButtonUnchecked } from "@mui/icons-material";
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
  const [selectedImageId, setSelectedImageId] = useState(null);
  const [initialPiece, setInitialPiece] = useState(null);
  const [annotatedImages, setAnnotatedImages] = useState([]);
  const [totalImages, setTotalImages] = useState(0);
  const [currentImageIndex, setCurrentImageIndex] = useState(0);
  const [allImages, setAllImages] = useState([]);
  
  // ADDED: Force refresh trigger for SidenavImageDisplay
  const [refreshTrigger, setRefreshTrigger] = useState(0);
  const sidenavRef = useRef(null);
  
  const navigate = useNavigate();

  // FIXED: Handle image selection from sidebar
  const handleImageSelect = (url, imageId, index) => {
    setSelectedImageUrl(url);
    setSelectedImageId(imageId);
    if (typeof index === 'number') {
      setCurrentImageIndex(index);
    }
  };

  // FIXED: Handle first image load - don't call handleImageSelect to avoid loops
  const handleFirstImageLoad = (url, imageId) => {
    setSelectedImageUrl(url);
    setSelectedImageId(imageId);
    setCurrentImageIndex(0);
  };

  // Handle image count updates
  const handleImageCountUpdate = (count) => {
    setTotalImages(count);
  };

  // FIXED: Handle when images are fetched from SidenavImageDisplay - only set allImages
  const handleImagesLoaded = (images) => {
    setAllImages(images);
  };

  // FIXED: Handle annotation saved - mark image as annotated and trigger refresh
  const handleAnnotationSaved = async (imageUrl, imageId) => {
    // Mark current image as annotated locally first for immediate UI update
    setAnnotatedImages(prev => {
      if (!prev.includes(imageUrl)) {
        return [...prev, imageUrl];
      }
      return prev;
    });

    // ADDED: Force SidenavImageDisplay to refresh its data
    await refreshImageData();
    
    // Trigger a refresh of the sidebar component
    setRefreshTrigger(prev => prev + 1);
  };

  // FIXED: Move to next image - simplified logic
  const moveToNextImage = () => {
    if (allImages.length === 0) return;

    // Simple logic: just move to the next image in sequence
    let nextIndex = (currentImageIndex + 1) % allImages.length;
    
    const nextImage = allImages[nextIndex];
    if (nextImage) {
      setSelectedImageUrl(nextImage.url);
      setSelectedImageId(nextImage.name);
      setCurrentImageIndex(nextIndex);
    }
  };

  // FIXED: Enhanced function to refresh image data from backend
  const refreshImageData = async () => {
    if (!selectedPieceLabel) return;
    
    try {
      console.log('Refreshing image data for piece:', selectedPieceLabel);
      
      const response = await api.get(`/api/annotation/annotations/get_images_of_piece/${selectedPieceLabel}`);
      const updatedImages = response.data;
      
      console.log('Updated images from backend:', updatedImages);
      
      setAllImages(updatedImages);
      
      // Update the is_annotated status for current image
      const currentImage = updatedImages.find(img => img.name === selectedImageId);
      if (currentImage && currentImage.is_annotated) {
        setAnnotatedImages(prev => {
          if (!prev.includes(selectedImageUrl)) {
            return [...prev, selectedImageUrl];
          }
          return prev;
        });
      }
      
      return updatedImages; // Return the updated data
    } catch (error) {
      console.error("Error refreshing image data:", error);
      return null;
    }
  };

  // ADDED: Function to force sidebar refresh
  const forceSidebarRefresh = () => {
    setRefreshTrigger(prev => prev + 1);
  };

  // Load initial piece
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

  // FIXED: Load existing annotations for the piece - only when piece changes
  useEffect(() => {
    const loadExistingAnnotations = async () => {
      if (!selectedPieceLabel) {
        setAnnotatedImages([]);
        return;
      }

      try {
        // Use the existing endpoint that returns all annotations for a piece
        const response = await api.get(`/api/annotation/annotations/piece/${selectedPieceLabel}/annotations`);
        
        if (response.data && response.data.images) {
          // Extract image paths that have annotations
          const annotatedImagePaths = [];
          
          Object.values(response.data.images).forEach(imageData => {
            if (imageData.annotations && imageData.annotations.length > 0) {
              annotatedImagePaths.push(imageData.image_path);
            }
          });
          
          setAnnotatedImages(annotatedImagePaths);
        } else {
          setAnnotatedImages([]);
        }
      } catch (error) {
        console.error("Error loading existing annotations:", error);
        // If endpoint doesn't exist or fails, continue with empty annotated images
        setAnnotatedImages([]);
      }
    };

    loadExistingAnnotations();
  }, [selectedPieceLabel]); // Only depend on selectedPieceLabel

  // Get annotation statistics from allImages (using backend data)
  const getAnnotationStats = () => {
    if (allImages.length === 0) return { annotatedCount: 0, totalCount: 0 };
    const annotatedCount = allImages.filter(img => img.is_annotated === true).length;
    const totalCount = allImages.length;
    return { annotatedCount, totalCount };
  };

  const { annotatedCount, totalCount } = getAnnotationStats();

  return (
    <Box sx={{ width: '100%', maxWidth: { sm: '100%', md: '1700px' }, margin: '0 auto' }}>
      {/* Header Section - Keep outside the main grid */}
      <HeaderBox>
        <HeaderTitle>
          {selectedPieceLabel ? `${selectedPieceLabel} Images` : "Select a Piece"}
        </HeaderTitle>
        {/* FIXED: Updated Stats Header to use backend data */}
        {totalImages > 0 && (
          <Box sx={{ 
            padding: '8px 16px', 
            backgroundColor: 'rgba(102, 126, 234, 0.05)',
            border: '1px solid rgba(102, 126, 234, 0.1)',
            borderRadius: '8px', 
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            mt: 1
          }}>
            <Typography variant="caption" sx={{ color: '#666', fontSize: '0.75rem' }}>
              Image {currentImageIndex + 1} of {totalCount} • {annotatedCount}/{totalCount} annotated
            </Typography>
            <Box sx={{ display: 'flex', gap: 1 }}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                <CheckCircle sx={{ fontSize: 12, color: '#4caf50' }} />
                <Typography variant="caption" sx={{ fontSize: '0.7rem', color: '#4caf50' }}>
                  {annotatedCount}
                </Typography>
              </Box>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                <RadioButtonUnchecked sx={{ fontSize: 12, color: '#ff9800' }} />
                <Typography variant="caption" sx={{ fontSize: '0.7rem', color: '#ff9800' }}>
                  {totalCount - annotatedCount}
                </Typography>
              </Box>
            </Box>
          </Box>
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
                imageId={selectedImageId}
                onAnnotationSaved={handleAnnotationSaved}
                onMoveToNextImage={moveToNextImage}
                onRefreshImages={refreshImageData} // FIXED: Pass refresh function
              />
            </AnnotationCard>
          </CenteredContainer>
        </Grid>
        
        {/* Sidebar Column */}
        <Grid size={{ xs: 12, md: 3 }}>
          <SidebarContainer>
            <Box sx={{ width: "100%", height: "480px" }}> {/* Match annotation card height */}
              <SidenavImageDisplay
                ref={sidenavRef}
                pieceLabel={selectedPieceLabel}
                onImageSelect={handleImageSelect}
                onFirstImageLoad={handleFirstImageLoad}
                onImageCountUpdate={handleImageCountUpdate}
                onImagesLoaded={handleImagesLoaded}
                currentImageIndex={currentImageIndex}
                refreshTrigger={refreshTrigger} // ADDED: Pass refresh trigger
              />
            </Box>
          </SidebarContainer>
        </Grid>
      </Grid>
    </Box>
  );
}