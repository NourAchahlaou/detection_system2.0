import React, { useState, useEffect, useRef, useCallback } from "react";
import { Card, Grid, Box, styled, Typography, IconButton, Button } from "@mui/material";
import { 
  ArrowBack, 
  ArrowForward, 
  ZoomIn, 
  ZoomOut, 
  CenterFocusStrong,
  Fullscreen,
  FullscreenExit,
  PhotoLibrary
} from "@mui/icons-material";
import api from "../../utils/UseAxios";
import { useNavigate, useSearchParams } from "react-router-dom";

// Updated to match VideoCard styling exactly
const ViewerCard = styled(Card)(({ theme }) => ({
  width: "900px",
  height: "480px", 
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  backgroundColor: "#f5f5f5",
  border: "2px solid #667eea",
  borderRadius: "12px",
  position: "relative",
  overflow: "hidden",
  padding: 0,
  margin: 0,
  '& .MuiCard-root': {
    padding: 0,
  },
  [theme.breakpoints.down("md")]: {
    width: "100%",
    maxWidth: "700px",
  },
  [theme.breakpoints.down("sm")]: {
    height: "300px",
  },
}));

const ImageContainer = styled(Box)({
  position: "relative",
  width: "100%",
  height: "100%",
  overflow: "hidden",
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  backgroundColor: "#000",
});

const ViewerImage = styled("img")({
  maxWidth: "100%",
  maxHeight: "100%",
  objectFit: "contain",
  transition: "transform 0.3s ease",
  cursor: "grab",
  "&:active": {
    cursor: "grabbing",
  },
});

const ControlsOverlay = styled(Box)({
  position: "absolute",
  bottom: "16px",
  left: "50%",
  transform: "translateX(-50%)",
  display: "flex",
  gap: "8px",
  backgroundColor: "rgba(0, 0, 0, 0.7)",
  borderRadius: "24px",
  padding: "8px 12px",
  backdropFilter: "blur(10px)",
});

const NavigationButton = styled(IconButton)({
  backgroundColor: "rgba(255, 255, 255, 0.1)",
  color: "white",
  "&:hover": {
    backgroundColor: "rgba(255, 255, 255, 0.2)",
  },
  "&:disabled": {
    backgroundColor: "rgba(255, 255, 255, 0.05)",
    color: "rgba(255, 255, 255, 0.3)",
  },
});

const SidebarContainer = styled(Box)({
  display: "flex",
  alignItems: "flex-start",
  justifyContent: "center", 
  height: "100%",
  minHeight: "600px",
  padding: "0 16px",
});

const SidebarCard = styled(Card)({
  width: "100%",
  height: "480px",
  border: "1px solid rgba(102, 126, 234, 0.2)",
  borderRadius: "12px",
  display: "flex",
  flexDirection: "column",
  overflow: "hidden",
});

const SidebarHeader = styled(Box)({
  padding: "16px",
  borderBottom: "1px solid rgba(102, 126, 234, 0.1)",
  backgroundColor: "rgba(102, 126, 234, 0.05)",
});

const ImageGrid = styled(Box)({
  flex: 1,
  overflow: "auto",
  padding: "16px",
  display: "grid",
  gridTemplateColumns: "repeat(auto-fill, minmax(80px, 1fr))",
  gap: "8px",
  alignContent: "start",
});

const ThumbnailImage = styled("img")(({ active }) => ({
  width: "100%",
  aspectRatio: "1",
  objectFit: "cover",
  borderRadius: "6px",
  cursor: "pointer",
  border: active ? "2px solid #667eea" : "2px solid transparent",
  transition: "all 0.2s ease",
  "&:hover": {
    border: "2px solid #667eea",
    transform: "scale(1.05)",
  },
}));

const HeaderBox = styled(Box)({
  paddingTop: "16px",
  paddingBottom: "16px",
});

const HeaderTitle = styled(Typography)({
  fontSize: "1.1rem",
  fontWeight: "600",
  color: "#333",
  textAlign: "center",
});

const CenteredContainer = styled(Box)({
  display: "flex",
  flexDirection: "column",
  alignItems: "center",
  justifyContent: "center",
  height: "100%",
  minHeight: "600px",
});

const NoImageState = styled(Box)({
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  flexDirection: "column",
  height: "100%",
  color: "#666",
  gap: 2,
});

export default function PieceImageViewer() {
  const [selectedPieceLabel, setSelectedPieceLabel] = useState('');
  const [selectedImageUrl, setSelectedImageUrl] = useState('');
  const [selectedImageId, setSelectedImageId] = useState(null);
  const [currentImageIndex, setCurrentImageIndex] = useState(0);
  const [allImages, setAllImages] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [pieceNotFound, setPieceNotFound] = useState(false);
  const [zoom, setZoom] = useState(1);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [imagePosition, setImagePosition] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });
  
  const imageRef = useRef(null);
  const containerRef = useRef(null);
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();

  const urlPiece = searchParams.get('piece');

  useEffect(() => {
    async function fetchInitialPiece() {
      setIsLoading(true);
      setPieceNotFound(false);
      
      try {
        if (urlPiece) {
          console.log('URL piece parameter found:', urlPiece);
          
          const decodedPiece = decodeURIComponent(urlPiece);
          const isValid = await validatePiece(decodedPiece);
          
          if (isValid) {
            setSelectedPieceLabel(decodedPiece);
            setIsLoading(false);
            return;
          } else {
            setPieceNotFound(true);
          }
        }

        // Default behavior: get first piece
        const response = await api.get("/api/annotation/annotations/get_all_pieces");
        const pieces = response.data;
        
        if (pieces.length > 0) {
          const firstPiece = pieces[0];
          setSelectedPieceLabel(firstPiece.piece_label);
          
          if (urlPiece && pieceNotFound) {
            const newUrl = new URL(window.location);
            newUrl.searchParams.set('piece', encodeURIComponent(firstPiece.piece_label));
            window.history.replaceState({}, '', newUrl);
          }
        } else {
          navigate("/204");
        }
      } catch (error) {
        console.error("Error fetching initial piece:", error);
        navigate("/204");
      } finally {
        setIsLoading(false);
      }
    }

    fetchInitialPiece();
  }, [navigate, urlPiece]);

  const validatePiece = async (pieceLabel) => {
    try {
      const response = await api.get(`/api/annotation/annotations/get_images_of_piece/${pieceLabel}`);
      return response.data && response.data.length > 0;
    } catch (error) {
      console.error('Error validating piece:', error);
      return false;
    }
  };

  useEffect(() => {
    if (!selectedPieceLabel) return;

    const loadImages = async () => {
      try {
        const response = await api.get(`/api/annotation/annotations/get_images_of_piece/${selectedPieceLabel}`);
        const images = response.data || [];
        
        setAllImages(images);
        
        if (images.length > 0) {
          setSelectedImageUrl(images[0].url);
          setSelectedImageId(images[0].name);
          setCurrentImageIndex(0);
        }
      } catch (error) {
        console.error("Error loading images:", error);
        setAllImages([]);
      }
    };

    loadImages();
  }, [selectedPieceLabel]);

  const handleImageSelect = useCallback((image, index) => {
    setSelectedImageUrl(image.url);
    setSelectedImageId(image.name);
    setCurrentImageIndex(index);
    resetZoom();
  }, []);

  const handlePrevImage = () => {
    if (allImages.length === 0) return;
    const prevIndex = currentImageIndex === 0 ? allImages.length - 1 : currentImageIndex - 1;
    const prevImage = allImages[prevIndex];
    handleImageSelect(prevImage, prevIndex);
  };

  const handleNextImage = () => {
    if (allImages.length === 0) return;
    const nextIndex = currentImageIndex === allImages.length - 1 ? 0 : currentImageIndex + 1;
    const nextImage = allImages[nextIndex];
    handleImageSelect(nextImage, nextIndex);
  };

  const handleZoomIn = () => {
    setZoom(prev => Math.min(prev * 1.2, 5));
  };

  const handleZoomOut = () => {
    setZoom(prev => Math.max(prev / 1.2, 0.5));
  };

  const resetZoom = () => {
    setZoom(1);
    setImagePosition({ x: 0, y: 0 });
  };

  const toggleFullscreen = () => {
    if (!document.fullscreenElement) {
      containerRef.current?.requestFullscreen();
      setIsFullscreen(true);
    } else {
      document.exitFullscreen();
      setIsFullscreen(false);
    }
  };

  // Handle mouse drag for panning
  const handleMouseDown = (e) => {
    if (zoom > 1) {
      setIsDragging(true);
      setDragStart({
        x: e.clientX - imagePosition.x,
        y: e.clientY - imagePosition.y
      });
    }
  };

  const handleMouseMove = (e) => {
    if (isDragging && zoom > 1) {
      setImagePosition({
        x: e.clientX - dragStart.x,
        y: e.clientY - dragStart.y
      });
    }
  };

  const handleMouseUp = () => {
    setIsDragging(false);
  };

  // Handle keyboard navigation
  useEffect(() => {
    const handleKeyDown = (e) => {
      switch (e.key) {
        case 'ArrowLeft':
          handlePrevImage();
          break;
        case 'ArrowRight':
          handleNextImage();
          break;
        case '+':
        case '=':
          handleZoomIn();
          break;
        case '-':
          handleZoomOut();
          break;
        case 'r':
        case 'R':
          resetZoom();
          break;
        case 'f':
        case 'F':
          toggleFullscreen();
          break;
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [currentImageIndex, allImages.length]);

  // Handle fullscreen change
  useEffect(() => {
    const handleFullscreenChange = () => {
      setIsFullscreen(!!document.fullscreenElement);
    };

    document.addEventListener('fullscreenchange', handleFullscreenChange);
    return () => document.removeEventListener('fullscreenchange', handleFullscreenChange);
  }, []);

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
      {/* Header Section */}
      <HeaderBox>
        <HeaderTitle>
          {selectedPieceLabel ? `${selectedPieceLabel} Images` : "Select a Piece"}
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
        {allImages.length > 0 && (
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
              Image {currentImageIndex + 1} of {allImages.length} • Viewing Mode
            </Typography>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <PhotoLibrary sx={{ fontSize: 14, color: '#667eea' }} />
              <Typography variant="caption" sx={{ fontSize: '0.7rem', color: '#667eea' }}>
                {allImages.length} total
              </Typography>
            </Box>
          </Box>
        )}
      </HeaderBox>

      {/* Main Content Grid */}
      <Grid
        container
        spacing={2}
        columns={12}
        sx={{ 
          mb: 2,
          minHeight: "600px",
          alignItems: "flex-start"
        }}
      >
        {/* Main Viewer Column */}
        <Grid size={{ xs: 12, md: 9 }}>
          <CenteredContainer>
            <ViewerCard ref={containerRef}>
              {selectedImageUrl ? (
                <ImageContainer
                  onMouseDown={handleMouseDown}
                  onMouseMove={handleMouseMove}
                  onMouseUp={handleMouseUp}
                  onMouseLeave={handleMouseUp}
                >
                  <ViewerImage
                    ref={imageRef}
                    src={selectedImageUrl}
                    alt={selectedImageId}
                    style={{
                      transform: `scale(${zoom}) translate(${imagePosition.x / zoom}px, ${imagePosition.y / zoom}px)`,
                      cursor: zoom > 1 ? (isDragging ? 'grabbing' : 'grab') : 'default'
                    }}
                    onError={(e) => {
                      console.log(`Failed to load image: ${selectedImageUrl}`);
                    }}
                  />
                  
                  <ControlsOverlay>
                    <NavigationButton 
                      size="small" 
                      onClick={handlePrevImage}
                      disabled={allImages.length <= 1}
                    >
                      <ArrowBack />
                    </NavigationButton>
                    
                    <NavigationButton size="small" onClick={handleZoomOut}>
                      <ZoomOut />
                    </NavigationButton>
                    
                    <NavigationButton size="small" onClick={resetZoom}>
                      <CenterFocusStrong />
                    </NavigationButton>
                    
                    <NavigationButton size="small" onClick={handleZoomIn}>
                      <ZoomIn />
                    </NavigationButton>
                    
                    <NavigationButton size="small" onClick={toggleFullscreen}>
                      {isFullscreen ? <FullscreenExit /> : <Fullscreen />}
                    </NavigationButton>
                    
                    <NavigationButton 
                      size="small" 
                      onClick={handleNextImage}
                      disabled={allImages.length <= 1}
                    >
                      <ArrowForward />
                    </NavigationButton>
                  </ControlsOverlay>
                </ImageContainer>
              ) : (
                <NoImageState>
                  <PhotoLibrary sx={{ fontSize: 64, opacity: 0.4, mb: 2 }} />
                  <Typography variant="h6" sx={{ opacity: 0.9, mb: 1 }}>
                    No Image Selected
                  </Typography>
                  <Typography variant="body2" sx={{ opacity: 0.7 }}>
                    Select an image from the sidebar to view
                  </Typography>
                </NoImageState>
              )}
            </ViewerCard>
          </CenteredContainer>
        </Grid>
        
        {/* Sidebar Column */}
        <Grid size={{ xs: 12, md: 3 }}>
          <SidebarContainer>
            <SidebarCard>
              <SidebarHeader>
                <Typography variant="h6" sx={{ fontSize: '1rem', fontWeight: '600', color: '#333' }}>
                  Images ({allImages.length})
                </Typography>
                <Typography variant="caption" sx={{ color: '#666', display: 'block', mt: 0.5 }}>
                  Click to view • Use arrow keys to navigate
                </Typography>
              </SidebarHeader>
              
              <ImageGrid>
                {allImages.map((image, index) => (
                  <ThumbnailImage
                    key={image.name}
                    src={image.url}
                    alt={image.name}
                    active={index === currentImageIndex}
                    onClick={() => handleImageSelect(image, index)}
                    onError={(e) => {
                      e.target.style.display = 'none';
                    }}
                  />
                ))}
                
                {allImages.length === 0 && (
                  <Box sx={{ 
                    gridColumn: '1 / -1', 
                    display: 'flex', 
                    flexDirection: 'column', 
                    alignItems: 'center', 
                    justifyContent: 'center',
                    minHeight: '200px',
                    color: '#666'
                  }}>
                    <PhotoLibrary sx={{ fontSize: 48, opacity: 0.4, mb: 2 }} />
                    <Typography variant="body2" sx={{ opacity: 0.7, textAlign: 'center' }}>
                      No images found for this piece
                    </Typography>
                  </Box>
                )}
              </ImageGrid>
            </SidebarCard>
          </SidebarContainer>
        </Grid>
      </Grid>
      
      {/* Keyboard shortcuts info */}
      <Box sx={{ 
        mt: 2, 
        p: 2, 
        backgroundColor: 'rgba(102, 126, 234, 0.05)', 
        borderRadius: '8px',
        textAlign: 'center'
      }}>
        <Typography variant="caption" sx={{ color: '#666' }}>
          <strong>Keyboard shortcuts:</strong> ← → (navigate) • + - (zoom) • R (reset) • F (fullscreen)
        </Typography>
      </Box>
    </Box>
  );
}