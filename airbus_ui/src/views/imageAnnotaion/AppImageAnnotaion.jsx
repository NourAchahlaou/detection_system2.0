import React, { useState, useEffect, useRef, useCallback } from "react";
import { Card, Grid, Box, styled, Stack, Typography } from "@mui/material";
import { CheckCircle, RadioButtonUnchecked } from "@mui/icons-material";
import SidenavImageDisplay from "./annotation/components/SidenavImageDisplay";
import Simple from "./Simple";
import api from "../../utils/UseAxios";
import { useNavigate, useSearchParams, useLocation } from "react-router-dom";

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
  const [isLoading, setIsLoading] = useState(true);
  const [pieceNotFound, setPieceNotFound] = useState(false);
    
  const [refreshTrigger, setRefreshTrigger] = useState(0);
  const sidenavRef = useRef(null);
  
  const [imageStatusUpdates, setImageStatusUpdates] = useState({});
  
  const [updateImageStatusCallback, setUpdateImageStatusCallback] = useState(null);
  
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const location = useLocation();

  // NEW: Extract piece from URL parameters
  const urlPiece = searchParams.get('piece');
  

  // FIXED: Handle image selection from sidebar - ensure index is always updated
  const handleImageSelect = useCallback((imageUrl, imageId, index, statusUpdateCallback) => {
    console.log('Image selected from sidebar:', { imageUrl, imageId, index });
    setSelectedImageUrl(imageUrl);
    setSelectedImageId(imageId);
    
    // 🔧 FIX: Always update currentImageIndex when provided
    if (typeof index === 'number') {
      console.log('🔧 Updating currentImageIndex from sidebar:', index);
      setCurrentImageIndex(index);
    }
    
    // Store the status update callback for use in Simple component
    setUpdateImageStatusCallback(() => statusUpdateCallback);
  }, []);
  // Handle first image load - don't call handleImageSelect to avoid loops
  const handleFirstImageLoad = useCallback((url, imageId) => {
    setSelectedImageUrl(url);
    setSelectedImageId(imageId);
    setCurrentImageIndex(0);
  }, []);

  // Handle image count updates
  const handleImageCountUpdate = useCallback((count) => {
    setTotalImages(count);
  }, []);

  // Handle when images are fetched from SidenavImageDisplay - only set allImages
  const handleImagesLoaded = useCallback((images) => {
    setAllImages(images);
  }, []);

  // UPDATED: Handle annotation saved - now uses the new status change system
  const handleAnnotationSaved = useCallback(async (imageUrl, imageId) => {
    console.log('Annotation saved for image:', imageId);
    
    // Mark current image as annotated locally first for immediate UI update
    setAnnotatedImages(prev => {
      if (!prev.includes(imageUrl)) {
        return [...prev, imageUrl];
      }
      return prev;
    });

    // Force SidenavImageDisplay to refresh its data
    await refreshImageData();
    
    // Trigger a refresh of the sidebar component
    setRefreshTrigger(prev => prev + 1);
  }, []);

  // Move to next image - simplified logic
  const moveToNextImage = useCallback(() => {
    if (allImages.length === 0) return;

    // Simple logic: just move to the next image in sequence
    let nextIndex = (currentImageIndex + 1) % allImages.length;
    
    const nextImage = allImages[nextIndex];
    if (nextImage) {
      setSelectedImageUrl(nextImage.url);
      setSelectedImageId(nextImage.name);
      setCurrentImageIndex(nextIndex);
    }
  }, [allImages, currentImageIndex]);

  // Enhanced function to refresh image data from backend
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
  // Handle refreshing images
  const handleRefreshImages = useCallback(() => {
    setRefreshTrigger(prev => prev + 1);
  }, []);

  // NEW: Handle image status changes from Simple component
  const handleImageStatusChange = useCallback((imageId, hasAnnotations) => {
    console.log(`Parent: Image ${imageId} status changed to ${hasAnnotations ? 'annotated' : 'not annotated'}`);
    
    // Update local status tracking
    setImageStatusUpdates(prev => ({
      ...prev,
      [imageId]: hasAnnotations
    }));
    
    // Also call the SidenavImageDisplay status update function if available
    if (updateImageStatusCallback) {
      updateImageStatusCallback(imageId, hasAnnotations);
    }
  }, [updateImageStatusCallback]);

  // NEW: Function to validate if a piece exists and has images
  const validatePiece = async (pieceLabel) => {
    try {
      console.log('Validating piece:', pieceLabel);
      
      // Try to get images for this piece
      const response = await api.get(`/api/annotation/annotations/get_images_of_piece/${pieceLabel}`);
      
      if (response.data && response.data.length > 0) {
        return true;
      } else {
        console.warn('Piece exists but has no images:', pieceLabel);
        return false;
      }
    } catch (error) {
      console.error('Error validating piece:', error);
      if (error.response?.status === 404) {
        console.warn('Piece not found:', pieceLabel);
        return false;
      }
      // For other errors, assume piece might exist
      return true;
    }
  };

  // NEW: Updated initial load effect to handle URL parameters
  useEffect(() => {
    async function fetchInitialPiece() {
      setIsLoading(true);
      setPieceNotFound(false);
      
      try {
        // If there's a piece in URL parameters, try to use it first
        if (urlPiece) {
          console.log('URL piece parameter found:', urlPiece);
          
          const decodedPiece = decodeURIComponent(urlPiece);
          const isValid = await validatePiece(decodedPiece);
          
          if (isValid) {
            console.log('URL piece is valid, using:', decodedPiece);
            setSelectedPieceLabel(decodedPiece);
            setInitialPiece(decodedPiece);
            setIsLoading(false);
            return;
          } else {
            console.log('URL piece is invalid, falling back to default');
            setPieceNotFound(true);
            // Don't return here, fall through to default behavior
          }
        }

        // Default behavior: get first non-annotated piece
        const response = await api.get("/api/annotation/annotations/get_Img_nonAnnotated");
        const pieces = response.data;
        
        if (pieces.length > 0) {
          const firstPiece = pieces[0];
          console.log('Using first available piece:', firstPiece.piece_label);
          setInitialPiece(firstPiece.piece_label);
          setSelectedPieceLabel(firstPiece.piece_label);
          
          // If we came from URL with invalid piece, update URL to reflect actual piece
          if (urlPiece && pieceNotFound) {
            const newUrl = new URL(window.location);
            newUrl.searchParams.set('piece', encodeURIComponent(firstPiece.piece_label));
            window.history.replaceState({}, '', newUrl);
          }
        } else {
          console.log('No pieces available, redirecting to 204');
          navigate("/204");
        }
      } catch (error) {
        console.error("Error fetching initial piece:", error.response?.data?.detail || error.message);
        navigate("/204");
      } finally {
        setIsLoading(false);
      }
    }

    fetchInitialPiece();
  }, [navigate, urlPiece]); // Include urlPiece in dependencies

  // NEW: Effect to handle URL parameter changes during runtime
  useEffect(() => {
    if (!urlPiece || !selectedPieceLabel || isLoading) return;
    
    const decodedUrlPiece = decodeURIComponent(urlPiece);
    
    // If URL piece is different from current piece, switch to it
    if (decodedUrlPiece !== selectedPieceLabel) {
      console.log('URL piece changed, switching from', selectedPieceLabel, 'to', decodedUrlPiece);
      
      validatePiece(decodedUrlPiece).then(isValid => {
        if (isValid) {
          setSelectedPieceLabel(decodedUrlPiece);
          // Reset image selection state
          setSelectedImageUrl('');
          setSelectedImageId(null);
          setCurrentImageIndex(0);
          setAllImages([]);
          setAnnotatedImages([]);
          // Force sidebar refresh
          setRefreshTrigger(prev => prev + 1);
        } else {
          console.warn('Invalid piece in URL, ignoring:', decodedUrlPiece);
          setPieceNotFound(true);
        }
      });
    }
  }, [urlPiece, selectedPieceLabel, isLoading]);

  // Load existing annotations for the piece - only when piece changes
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

  // NEW: Show loading state
  if (isLoading) {
    return (
      <Box sx={{ 
        width: '100%', 
        maxWidth: { sm: '100%', md: '1700px' }, 
        margin: '0 auto',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        minHeight: '400px'
      }}>
        <Typography variant="h6" sx={{ color: '#666' }}>
          Loading piece data...
        </Typography>
      </Box>
    );
  }

  return (
    <Box sx={{ width: '100%', maxWidth: { sm: '100%', md: '1700px' }, margin: '0 auto' }}>
      {/* Header Section - Keep outside the main grid */}
      <HeaderBox>
        <HeaderTitle>
          {selectedPieceLabel ? `${selectedPieceLabel} Images` : "Select a Piece"}
          {/* NEW: Show warning if piece from URL was not found */}
          {pieceNotFound && urlPiece && (
            <Typography 
              variant="caption" 
              sx={{ 
                display: 'block', 
                color: '#ff9800', 
                fontSize: '0.75rem', 
                mt: 0.5 
              }}
            >
              Note: Piece "{decodeURIComponent(urlPiece)}" not found, showing default piece
            </Typography>
          )}
        </HeaderTitle>
        {/* Updated Stats Header to use backend data */}
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
                onRefreshImages={handleRefreshImages}
                // NEW: Pass the status change callback
                onImageStatusChange={handleImageStatusChange}
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
                refreshTrigger={refreshTrigger}
                // NEW: Pass image status updates
                imageStatusUpdates={imageStatusUpdates}
                onImageStatusUpdate={(imageId, hasAnnotations) => {
                  console.log(`SidenavImageDisplay notified parent: Image ${imageId} -> ${hasAnnotations}`);
                }}
              />
            </Box>
          </SidebarContainer>
        </Grid>
      </Grid>
      
    </Box>
  );
}