import { useEffect, useState, useCallback, useRef } from "react";
import { 
  Box, 
  Typography, 
  styled, 
  Fade, 
  CircularProgress,
  IconButton,
  Card,
  Chip,
  Modal,
  Backdrop,
  Tooltip
} from "@mui/material";
import { 
  Photo, 
  KeyboardArrowLeft, 
  KeyboardArrowRight,
  CheckCircle,
  RadioButtonUnchecked,
  Fullscreen,
  FullscreenExit,
  Refresh
} from "@mui/icons-material";
import api from "../../../../utils/UseAxios";
import ImageWithAnnotations from "./ImageWithAnnotations";

// Keep all your existing styled components
const MaxCustomaizer = styled("div")(({ theme }) => ({
  width: "100%",
  height: "100%",
  display: "flex",
  flexDirection: "column",
  overflow: "hidden",
  backgroundColor: "transparent",
}));

const LoadingState = styled(Box)({
  flex: 1,
  display: "flex",
  flexDirection: "column",
  alignItems: "center",
  justifyContent: "center",
  color: "#666",
  gap: 2,
});

const EmptyState = styled(Box)({
  flex: 1,
  display: "flex",
  flexDirection: "column",
  alignItems: "center",
  justifyContent: "center",
  color: "#666",
  textAlign: "center",
  padding: "32px 16px",
});

export default function HorizontalAnnotationSlider({
  pieceLabel,
  onImageSelect,
  onFirstImageLoad,
  onImageCountUpdate,
  onImagesLoaded,
  currentImageIndex,
  refreshTrigger,
  onImageStatusUpdate,
  imageStatusUpdates = {},
  onImageDoubleClick
}) {
  const [images, setImages] = useState([]);
  const [localCurrentIndex, setLocalCurrentIndex] = useState(0);
  const [loading, setLoading] = useState(false);
  const [selectedImageUrl, setSelectedImageUrl] = useState('');
  const [originalOrderMap, setOriginalOrderMap] = useState(new Map());
  const [isTransitioning, setIsTransitioning] = useState(false);
  const [isFullscreen, setIsFullscreen] = useState(false);
  
  const mountedRef = useRef(true);
  const lastPieceLabelRef = useRef(null);

  // Handle double click function
  const handleImageDoubleClick = useCallback((imageIndex) => {
    if (onImageDoubleClick) {
      onImageDoubleClick(imageIndex);
    }
  }, [onImageDoubleClick]);

  // Effect to handle real-time image status updates
  useEffect(() => {
    if (Object.keys(imageStatusUpdates).length > 0) {
      setImages(prevImages => 
        prevImages.map(image => {
          const imageId = image.name || image.id;
          if (imageId in imageStatusUpdates) {
            const hasAnnotations = imageStatusUpdates[imageId];
            console.log(`HorizontalAnnotationSlider: Updating status for image ${imageId}: ${hasAnnotations ? 'annotated' : 'not annotated'}`);
            return {
              ...image,
              is_annotated: hasAnnotations
            };
          }
          return image;
        })
      );
    }
  }, [imageStatusUpdates]);

  // Helper function to assign stable display numbers
  const assignStableNumbers = (newImages, existingOrderMap = new Map()) => {
    const updatedOrderMap = new Map(existingOrderMap);
    const numberedImages = [];
    
    const usedNumbers = new Set(Array.from(updatedOrderMap.values()));
    
    newImages.forEach((image, index) => {
      const imageKey = image.name || image.id || image.url;
      
      if (updatedOrderMap.has(imageKey)) {
        numberedImages.push({
          ...image,
          originalIndex: index,
          displayNumber: updatedOrderMap.get(imageKey)
        });
      } else {
        let nextNumber = 1;
        while (usedNumbers.has(nextNumber)) {
          nextNumber++;
        }
        updatedOrderMap.set(imageKey, nextNumber);
        usedNumbers.add(nextNumber);
        
        numberedImages.push({
          ...image,
          originalIndex: index,
          displayNumber: nextNumber
        });
      }
    });
    
    numberedImages.sort((a, b) => a.displayNumber - b.displayNumber);
    
    return { numberedImages, updatedOrderMap };
  };

  // Function to update individual image annotation status
  const updateImageAnnotationStatus = useCallback(async (imageId, hasAnnotations) => {
    console.log(`HorizontalAnnotationSlider: Received status update for image ${imageId}: ${hasAnnotations}`);
    
    setImages(prevImages => 
      prevImages.map(image => {
        if ((image.name || image.id) === imageId) {
          console.log(`HorizontalAnnotationSlider: Updating image ${imageId} status from ${image.is_annotated} to ${hasAnnotations}`);
          return {
            ...image,
            is_annotated: hasAnnotations
          };
        }
        return image;
      })
    );

    if (onImageStatusUpdate) {
      onImageStatusUpdate(imageId, hasAnnotations);
    }
  }, [onImageStatusUpdate]);

  useEffect(() => {
    if (refreshTrigger > 0 && pieceLabel) {
      console.log('HorizontalAnnotationSlider: Refresh trigger activated, refetching images...');
      lastPieceLabelRef.current = null;
      fetchImages();
    }
  }, [refreshTrigger, pieceLabel]);

  useEffect(() => {
    if (typeof currentImageIndex === 'number' && currentImageIndex !== localCurrentIndex && images.length > 0) {
      setLocalCurrentIndex(currentImageIndex);
      if (images[currentImageIndex]) {
        setSelectedImageUrl(images[currentImageIndex].url);
      }
    }
  }, [currentImageIndex, images, localCurrentIndex]);

  const fetchImages = useCallback(async () => {
    if (!pieceLabel) {
      setImages([]);
      setLocalCurrentIndex(0);
      setSelectedImageUrl('');
      setOriginalOrderMap(new Map());
      if (onImageCountUpdate) {
        onImageCountUpdate(0);
      }
      return;
    }

    if (lastPieceLabelRef.current === pieceLabel && refreshTrigger === 0) {
      return;
    }

    try {
      setLoading(true);
      lastPieceLabelRef.current = pieceLabel;
      
      console.log('HorizontalAnnotationSlider: Fetching images for piece:', pieceLabel);
      
      const response = await api.get(`/api/annotation/annotations/get_images_of_piece/${pieceLabel}`);
      const rawData = response.data;
      
      console.log('HorizontalAnnotationSlider: Received images data:', rawData);
      
      if (mountedRef.current) {
        const { numberedImages, updatedOrderMap } = assignStableNumbers(
          rawData, 
          refreshTrigger > 0 ? originalOrderMap : new Map()
        );
        
        setImages(numberedImages);
        setOriginalOrderMap(updatedOrderMap);
        
        if (refreshTrigger === 0) {
          setLocalCurrentIndex(0);
          setSelectedImageUrl('');
        }

        if (onImageCountUpdate) {
          onImageCountUpdate(numberedImages.length);
        }

        if (onImagesLoaded) {
          onImagesLoaded(numberedImages);
        }

        if (numberedImages.length > 0 && refreshTrigger === 0) {
          const firstImage = numberedImages[0];
          setSelectedImageUrl(firstImage.url);
          
          if (onFirstImageLoad) {
            onFirstImageLoad(firstImage.url, firstImage.name);
          }
          
          if (onImageSelect) {
            onImageSelect(firstImage.url, firstImage.name, 0, updateImageAnnotationStatus);
          }
        } else if (numberedImages.length > 0 && refreshTrigger > 0) {
          const currentImage = numberedImages[localCurrentIndex];
          if (currentImage) {
            setSelectedImageUrl(currentImage.url);
          }
        }
      }
  
    } catch (error) {
      console.error("Error fetching images:", error.response?.data?.detail || error.message);
      if (mountedRef.current) {
        setImages([]);
        setLocalCurrentIndex(0);
        setSelectedImageUrl('');
        setOriginalOrderMap(new Map());
        if (onImageCountUpdate) {
          onImageCountUpdate(0);
        }
      }
    } finally {
      if (mountedRef.current) {
        setLoading(false);
      }
    }
  }, [pieceLabel, refreshTrigger, localCurrentIndex, originalOrderMap, onImageCountUpdate, onImagesLoaded, onFirstImageLoad, onImageSelect, updateImageAnnotationStatus]);

  useEffect(() => {
    mountedRef.current = true;
    
    if (lastPieceLabelRef.current !== pieceLabel) {
      lastPieceLabelRef.current = null;
      fetchImages();
    }
    
    return () => {
      mountedRef.current = false;
    };
  }, [pieceLabel, fetchImages]);

  const handleImageClick = (imageUrl, imageId, index) => {
    setSelectedImageUrl(imageUrl);
    setLocalCurrentIndex(index);
    if (onImageSelect) {
      onImageSelect(imageUrl, imageId, index, updateImageAnnotationStatus);
    }
  };

  // Navigate left with transition
  const handlePrevious = useCallback(() => {
    if (images.length > 0 && !isTransitioning) {
      setIsTransitioning(true);
      setTimeout(() => {
        const newIndex = localCurrentIndex > 0 ? localCurrentIndex - 1 : images.length - 1;
        const newImage = images[newIndex];
        
        setLocalCurrentIndex(newIndex);
        setSelectedImageUrl(newImage.url);
        
        if (onImageSelect) {
          onImageSelect(newImage.url, newImage.name, newIndex, updateImageAnnotationStatus);
        }
        setTimeout(() => setIsTransitioning(false), 100);
      }, 200);
    }
  }, [images, localCurrentIndex, onImageSelect, updateImageAnnotationStatus, isTransitioning]);

  // Navigate right with transition
  const handleNext = useCallback(() => {
    if (images.length > 0 && !isTransitioning) {
      setIsTransitioning(true);
      setTimeout(() => {
        const newIndex = localCurrentIndex < images.length - 1 ? localCurrentIndex + 1 : 0;
        const newImage = images[newIndex];
        
        setLocalCurrentIndex(newIndex);
        setSelectedImageUrl(newImage.url);
        
        if (onImageSelect) {
          onImageSelect(newImage.url, newImage.name, newIndex, updateImageAnnotationStatus);
        }
        setTimeout(() => setIsTransitioning(false), 100);
      }, 200);
    }
  }, [images, localCurrentIndex, onImageSelect, updateImageAnnotationStatus, isTransitioning]);

  // Get visible images for horizontal three-image display
  const getVisibleImages = () => {
    if (images.length === 0) return [];
    
    if (images.length === 1) {
      return [{ ...images[0], position: 'center', index: 0 }];
    }
    
    if (images.length === 2) {
      const otherIndex = localCurrentIndex === 0 ? 1 : 0;
      return [
        { ...images[otherIndex], position: 'side', index: otherIndex },
        { ...images[localCurrentIndex], position: 'center', index: localCurrentIndex }
      ];
    }
    
    // For 3 or more images
    const prevIndex = localCurrentIndex > 0 ? localCurrentIndex - 1 : images.length - 1;
    const nextIndex = localCurrentIndex < images.length - 1 ? localCurrentIndex + 1 : 0;
    
    return [
      { ...images[prevIndex], position: 'left', index: prevIndex },
      { ...images[localCurrentIndex], position: 'center', index: localCurrentIndex },
      { ...images[nextIndex], position: 'right', index: nextIndex }
    ];
  };

  const visibleImages = getVisibleImages();

  // Keyboard navigation
  useEffect(() => {
    const handleKeyPress = (event) => {
      if (isTransitioning) return;
      
      switch (event.key) {
        case 'ArrowLeft':
          event.preventDefault();
          handlePrevious();
          break;
        case 'ArrowRight':
          event.preventDefault();
          handleNext();
          break;
        case 'f':
        case 'F':
          event.preventDefault();
          setIsFullscreen(prev => !prev);
          break;
        default:
          break;
      }
    };

    window.addEventListener('keydown', handleKeyPress);
    return () => {
      window.removeEventListener('keydown', handleKeyPress);
    };
  }, [handlePrevious, handleNext, isTransitioning]);

  // Render fullscreen modal
  const renderFullscreenModal = () => (
    <Modal
      open={isFullscreen}
      onClose={() => setIsFullscreen(false)}
      closeAfterTransition
      BackdropComponent={Backdrop}
      BackdropProps={{
        timeout: 500,
        sx: { backgroundColor: 'rgba(0, 0, 0, 0.9)' }
      }}
    >
      <Fade in={isFullscreen}>
        <Box
          sx={{
            position: 'fixed',
            top: 0,
            left: 0,
            width: '100vw',
            height: '100vh',
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            outline: 'none',
            overflow: 'hidden'
          }}
        >
          {/* Header Controls */}
          <Box
            sx={{
              position: 'absolute',
              top: 0,
              left: 0,
              right: 0,
              height: 80,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              px: 3,
              background: 'linear-gradient(180deg, rgba(0,0,0,0.8) 0%, transparent 100%)',
              zIndex: 2000
            }}
          >
            {/* Title */}
            <Typography
              variant="h6"
              sx={{
                color: 'white',
                fontWeight: 500,
                display: 'flex',
                alignItems: 'center',
                gap: 2
              }}
            >
              <Photo sx={{ color: '#667eea' }} />
              {pieceLabel} Images
              {images.length > 0 && (
                <Chip
                  label={`${localCurrentIndex + 1} of ${images.length}`}
                  size="small"
                  sx={{
                    backgroundColor: 'rgba(102, 126, 234, 0.8)',
                    color: 'white',
                    fontSize: '0.75rem'
                  }}
                />
              )}
            </Typography>

            {/* Controls */}
            <Box sx={{ display: 'flex', gap: 1 }}>
              <Tooltip title="Exit Fullscreen">
                <IconButton 
                  onClick={() => setIsFullscreen(false)}
                  sx={{
                    color: 'white',
                    backgroundColor: 'rgba(255,255,255,0.1)',
                    '&:hover': { backgroundColor: 'rgba(255,255,255,0.2)' }
                  }}
                >
                  <FullscreenExit />
                </IconButton>
              </Tooltip>
            </Box>
          </Box>

          {/* Main Content Area */}
          <Box
            sx={{
              flex: 1,
              width: '100%',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              position: 'relative',
              mt: 10,
              mb: 8
            }}
          >
            {/* Navigation Arrows */}
            {images.length > 1 && (
              <>
                <IconButton
                  onClick={handlePrevious}
                  disabled={isTransitioning}
                  sx={{
                    position: 'absolute',
                    left: 20,
                    zIndex: 1500,
                    color: 'white',
                    backgroundColor: 'rgba(0,0,0,0.3)',
                    '&:hover': {
                      backgroundColor: 'rgba(0,0,0,0.5)',
                      transform: 'translateX(-3px)'
                    },
                    '&:disabled': {
                      opacity: 0.5
                    },
                    transition: 'all 0.3s ease',
                    width: 56,
                    height: 56
                  }}
                >
                  <KeyboardArrowLeft sx={{ fontSize: 32 }} />
                </IconButton>
                
                <IconButton
                  onClick={handleNext}
                  disabled={isTransitioning}
                  sx={{
                    position: 'absolute',
                    right: 20,
                    zIndex: 1500,
                    color: 'white',
                    backgroundColor: 'rgba(0,0,0,0.3)',
                    '&:hover': {
                      backgroundColor: 'rgba(0,0,0,0.5)',
                      transform: 'translateX(3px)'
                    },
                    '&:disabled': {
                      opacity: 0.5
                    },
                    transition: 'all 0.3s ease',
                    width: 56,
                    height: 56
                  }}
                >
                  <KeyboardArrowRight sx={{ fontSize: 32 }} />
                </IconButton>
              </>
            )}

            {/* Images Container */}
            <Box
              sx={{
                position: 'relative',
                width: '90vw',
                height: '75vh',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: 3
              }}
            >
              {visibleImages.map((image) => {
                const isCenter = image.position === 'center';
                const isLeft = image.position === 'left';
                const isRight = image.position === 'right';
                const isSide = image.position === 'side';
                const isAnnotated = image.is_annotated === true;

                // Position and style calculations - EXACT SAME AS PASTE 1
                let translateX = 0;
                let zIndex = 100;
                let scale = 0.75;
                let opacity = isTransitioning ? 0.3 : 0.6;
                let rotation = 0;
                let width = '675px';  // 900px * 0.75 for side images
                let height = '360px'; // 480px * 0.75 for side images

                if (isCenter) {
                  translateX = 0;
                  zIndex = 300;
                  scale = 1;
                  opacity = isTransitioning ? 0.7 : 1;
                  rotation = 0;
                  width = '1080px';  // Fullscreen size
                  height = '576px';  // Fullscreen size
                } else if (isLeft) {
                  translateX = -300;
                  zIndex = 200;
                  scale = 0.8;
                  opacity = isTransitioning ? 0.2 : 0.5;
                  rotation = -8;
                  width = '720px';  // 900px * 0.8
                  height = '384px'; // 480px * 0.8
                } else if (isRight) {
                  translateX = 300;
                  zIndex = 200;
                  scale = 0.8;
                  opacity = isTransitioning ? 0.2 : 0.5;
                  rotation = 8;
                  width = '720px';  // 900px * 0.8
                  height = '384px'; // 480px * 0.8
                } else if (isSide) {
                  // For 2-image case
                  translateX = 200;
                  zIndex = 200;
                  scale = 0.85;
                  opacity = isTransitioning ? 0.3 : 0.6;
                  rotation = 5;
                  width = '765px';  // 900px * 0.85
                  height = '408px'; // 480px * 0.85
                }

                return (
                  <Card
                    key={`${image.index}-${image.position}-${refreshTrigger}-${image.is_annotated ? 'annotated' : 'pending'}`}
                    sx={{
                      position: 'absolute',
                      width: width,
                      height: height,
                      borderRadius: 3,
                      overflow: 'hidden',
                      cursor: !isCenter && images.length > 1 ? 'pointer' : 'default',
                      transition: isTransitioning 
                        ? 'all 0.6s cubic-bezier(0.4, 0, 0.2, 1)' 
                        : 'all 0.5s cubic-bezier(0.4, 0, 0.2, 1)',
                      transform: `translateX(${translateX}px) scale(${scale}) rotate(${rotation}deg)`,
                      transformOrigin: 'center center',
                      zIndex: zIndex,
                      opacity: opacity,
                      boxShadow: isCenter 
                        ? '0 25px 50px rgba(0,0,0,0.5)' 
                        : '0 15px 30px rgba(0,0,0,0.3)',
                      border: isAnnotated 
                        ? '3px solid #4caf50' 
                        : '3px solid rgba(255,255,255,0.2)',
                      padding: 0,
                      margin: 0,
                      '&:hover': {
                        opacity: !isCenter && images.length > 1 && !isTransitioning ? 0.8 : opacity,
                        transform: !isCenter && images.length > 1 && !isTransitioning
                          ? `translateX(${translateX}px) scale(${Math.min(scale + 0.05, 0.9)}) rotate(${rotation}deg)` 
                          : `translateX(${translateX}px) scale(${scale}) rotate(${rotation}deg)`,
                        zIndex: !isCenter && images.length > 1 ? zIndex + 50 : zIndex,
                        boxShadow: !isCenter && images.length > 1 
                          ? '0 20px 40px rgba(0,0,0,0.4)' 
                          : (isCenter ? '0 25px 50px rgba(0,0,0,0.5)' : '0 15px 30px rgba(0,0,0,0.3)')
                      }
                    }}
                    onClick={() => {
                      if (images.length > 1 && !isCenter && !isTransitioning) {
                        if (isLeft) handlePrevious();
                        else if (isRight) handleNext();
                        else if (isSide) {
                          handleImageClick(image.url, image.name, image.index);
                        }
                      }
                    }}
                    onDoubleClick={() => handleImageDoubleClick(image.index)}
                  >
                    <ImageWithAnnotations
                      imageUrl={image.url}
                      imageId={image.name}
                      width="100%"
                      height="100%"
                      alt={`Image ${image.displayNumber}`}
                      sx={{
                        width: '100%',
                        height: '100%',
                        objectFit: 'cover',
                        display: 'block',
                        padding: 0,
                        margin: 0,
                        border: 'none',
                        position: 'absolute',
                        top: 0,
                        left: 0,
                        right: 0,
                        bottom: 0
                      }}
                    />
                    
                    {/* Image Number Badge */}
                    <Box
                      sx={{
                        position: 'absolute',
                        top: 12,
                        left: 12,
                        backgroundColor: isCenter ? 'rgba(0,0,0,0.8)' : 'rgba(0,0,0,0.7)',
                        color: 'white',
                        borderRadius: '50%',
                        width: 32,
                        height: 32,
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        fontSize: '0.8rem',
                        fontWeight: 'bold',
                        boxShadow: '0 2px 8px rgba(0,0,0,0.3)',
                        zIndex: 1001
                      }}
                    >
                      {image.displayNumber}
                    </Box>

                    {/* Annotation Status Badge */}
                    <Box
                      sx={{
                        position: 'absolute',
                        bottom: 12,
                        right: 12,
                        backgroundColor: isAnnotated ? '#4caf50' : '#ff9800',
                        color: 'white',
                        borderRadius: '12px',
                        padding: '4px 8px',
                        fontSize: '0.6rem',
                        fontWeight: '600',
                        textTransform: 'uppercase',
                        letterSpacing: '0.5px',
                        boxShadow: isAnnotated 
                          ? '0 2px 8px rgba(76, 175, 80, 0.4)' 
                          : '0 2px 8px rgba(255, 152, 0, 0.4)',
                        zIndex: 1001,
                        display: 'flex',
                        alignItems: 'center',
                        gap: '4px'
                      }}
                    >
                      {isAnnotated ? (
                        <>
                          <CheckCircle sx={{ fontSize: 10 }} />
                          Done
                        </>
                      ) : (
                        <>
                          <RadioButtonUnchecked sx={{ fontSize: 10 }} />
                          Pending
                        </>
                      )}
                    </Box>
                  </Card>
                );
              })}
            </Box>
          </Box>

          {/* Footer with Progress Indicators */}
          {images.length > 1 && (
            <Box
              sx={{
                position: 'absolute',
                bottom: 20,
                left: '50%',
                transform: 'translateX(-50%)',
                display: 'flex',
                alignItems: 'center',
                gap: 1,
                backgroundColor: 'rgba(0,0,0,0.5)',
                backdropFilter: 'blur(10px)',
                borderRadius: 3,
                padding: 2,
                zIndex: 1000
              }}
            >
              {images.map((_, index) => (
                <Box
                  key={index}
                  sx={{
                    width: index === localCurrentIndex ? 12 : 8,
                    height: index === localCurrentIndex ? 12 : 8,
                    borderRadius: '50%',
                    backgroundColor: index === localCurrentIndex ? '#667eea' : 'rgba(255,255,255,0.4)',
                    transition: 'all 0.3s ease',
                    cursor: 'pointer'
                  }}
                  onClick={() => {
                    if (!isTransitioning) {
                      setIsTransitioning(true);
                      setTimeout(() => {
                        handleImageClick(images[index].url, images[index].name, index);
                        setTimeout(() => setIsTransitioning(false), 100);
                      }, 200);
                    }
                  }}
                />
              ))}
            </Box>
          )}

          {/* Keyboard Shortcuts Info */}
          <Box
            sx={{
              position: 'absolute',
              bottom: 20,
              right: 20,
              color: 'rgba(255,255,255,0.6)',
              fontSize: '0.75rem',
              textAlign: 'right',
              zIndex: 500
            }}
          >
            <Typography variant="caption" sx={{ display: 'block' }}>
              ← → Navigate • F Fullscreen
            </Typography>
          </Box>
        </Box>
      </Fade>
    </Modal>
  );

  return (
    <MaxCustomaizer>
      <Box
        sx={{
          flex: 1,
          display: 'flex',
          flexDirection: 'column',
          position: 'relative',
          backgroundColor: 'transparent',
          overflow: 'hidden',
        }}
      >
        {loading && (
          <Fade in={loading}>
            <LoadingState>
              <CircularProgress sx={{ color: '#667eea' }} size={32} />
              <Typography variant="body2" sx={{ opacity: 0.8 }}>
                Loading images...
              </Typography>
            </LoadingState>
          </Fade>
        )}

        {!loading && images.length === 0 && (
          <EmptyState>
            <Photo sx={{ fontSize: 64, opacity: 0.4, mb: 3 }} />
            <Typography variant="h6" sx={{ mb: 1, opacity: 0.9 }}>
              No Images
            </Typography>
            <Typography variant="body2" sx={{ opacity: 0.7 }}>
              {pieceLabel ? `No images found for "${pieceLabel}"` : 'Select a piece to view images'}
            </Typography>
          </EmptyState>
        )}

        {!loading && images.length > 0 && (
          <Box
            sx={{
              position: 'absolute',
              top: 0,
              left: 0,
              right: 0,
              bottom: 0,
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              padding: 2
            }}
          >
            {/* Header with image count */}
            <Box
              sx={{
                position: 'absolute',
                top: 16,
                left: 16,
                right: 16,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                zIndex: 500
              }}
            >
<Typography
                variant="h6"
                sx={{
                  color: '#667eea',
                  fontWeight: 500,
                  display: 'flex',
                  alignItems: 'center',
                  gap: 1
                }}
              >
                <Photo sx={{ color: '#667eea' }} />
                {pieceLabel}
              </Typography>
              
              {images.length > 0 && (
                <Chip
                  label={`${localCurrentIndex + 1} of ${images.length}`}
                  size="small"
                  sx={{
                    backgroundColor: 'rgba(102, 126, 234, 0.8)',
                    color: 'white',
                    fontSize: '0.75rem'
                  }}
                />
              )}
            </Box>

            {/* Fullscreen Toggle Button */}
            <IconButton
              onClick={() => setIsFullscreen(true)}
              sx={{
                position: 'absolute',
                top: 16,
                right: 16,
                zIndex: 500,
                color: '#667eea',
                backgroundColor: 'rgba(102, 126, 234, 0.1)',
                '&:hover': {
                  backgroundColor: 'rgba(102, 126, 234, 0.2)',
                },
                width: 40,
                height: 40
              }}
            >
              <Tooltip title="Fullscreen (F)">
                <Fullscreen sx={{ fontSize: 20 }} />
              </Tooltip>
            </IconButton>

            {/* Navigation Arrows */}
            {images.length > 1 && (
              <>
                <IconButton
                  onClick={handlePrevious}
                  disabled={isTransitioning}
                  sx={{
                    position: 'absolute',
                    left: 16,
                    zIndex: 400,
                    color: '#667eea',
                    backgroundColor: 'rgba(102, 126, 234, 0.1)',
                    '&:hover': {
                      backgroundColor: 'rgba(102, 126, 234, 0.2)',
                      transform: 'translateX(-3px)'
                    },
                    '&:disabled': {
                      opacity: 0.5
                    },
                    transition: 'all 0.3s ease',
                    width: 48,
                    height: 48
                  }}
                >
                  <KeyboardArrowLeft sx={{ fontSize: 28 }} />
                </IconButton>
                
                <IconButton
                  onClick={handleNext}
                  disabled={isTransitioning}
                  sx={{
                    position: 'absolute',
                    right: 16,
                    zIndex: 400,
                    color: '#667eea',
                    backgroundColor: 'rgba(102, 126, 234, 0.1)',
                    '&:hover': {
                      backgroundColor: 'rgba(102, 126, 234, 0.2)',
                      transform: 'translateX(3px)'
                    },
                    '&:disabled': {
                      opacity: 0.5
                    },
                    transition: 'all 0.3s ease',
                    width: 48,
                    height: 48
                  }}
                >
                  <KeyboardArrowRight sx={{ fontSize: 28 }} />
                </IconButton>
              </>
            )}

            {/* Images Container */}
            <Box
              sx={{
                position: 'relative',
                width: '80vw',
                height:'65vh',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: 2,
                mt: 4,
                mb: 4
              }}
            >
              {visibleImages.map((image) => {
                const isCenter = image.position === 'center';
                const isLeft = image.position === 'left';
                const isRight = image.position === 'right';
                const isSide = image.position === 'side';
                const isAnnotated = image.is_annotated === true;

                // Position and style calculations
                let translateX = 0;
                let zIndex = 100;
                let scale = 0.75;
                let opacity = isTransitioning ? 0.3 : 0.6;
                let rotation = 0;
                let width = '675px';
                let height = '360px';

                if (isCenter) {
                  translateX = 0;
                  zIndex = 300;
                  scale = 1;
                  opacity = isTransitioning ? 0.7 : 1;
                  rotation = 0;
                  width = '900px';
                  height = '480px';
                } else if (isLeft) {
                  translateX = -300;
                  zIndex = 200;
                  scale = 0.8;
                  opacity = isTransitioning ? 0.2 : 0.5;
                  rotation = -5;
                  width = '720px';
                  height = '384px';
                } else if (isRight) {
                  translateX = 300;
                  zIndex = 200;
                  scale = 0.8;
                  opacity = isTransitioning ? 0.2 : 0.5;
                  rotation = 5;
                  width = '720px';
                  height = '384px';
                } else if (isSide) {
                  translateX = 200;
                  zIndex = 200;
                  scale = 0.85;
                  opacity = isTransitioning ? 0.3 : 0.6;
                  rotation = 3;
                  width = '765px';
                  height = '408px';
                }

                return (
                  <Card
                    key={`${image.index}-${image.position}-${refreshTrigger}-${image.is_annotated ? 'annotated' : 'pending'}`}
                    sx={{
                      position: 'absolute',
                      width: width,
                      height: height,
                      borderRadius: 3,
                      overflow: 'hidden',
                      cursor: !isCenter && images.length > 1 ? 'pointer' : 'default',
                      transition: isTransitioning 
                        ? 'all 0.6s cubic-bezier(0.4, 0, 0.2, 1)' 
                        : 'all 0.5s cubic-bezier(0.4, 0, 0.2, 1)',
                      transform: `translateX(${translateX}px) scale(${scale}) rotate(${rotation}deg)`,
                      transformOrigin: 'center center',
                      zIndex: zIndex,
                      opacity: opacity,
                      boxShadow: isCenter 
                        ? '0 25px 50px rgba(0,0,0,0.3)' 
                        : '0 15px 30px rgba(0,0,0,0.2)',
                      border: isAnnotated 
                        ? '3px solid #4caf50' 
                        : isCenter 
                          ? '3px solid #667eea' 
                          : '3px solid rgba(255,255,255,0.9)',
                      padding: 0,
                      margin: 0,
                      '&:hover': {
                        opacity: !isCenter && images.length > 1 && !isTransitioning ? 0.8 : opacity,
                        transform: !isCenter && images.length > 1 && !isTransitioning
                          ? `translateX(${translateX}px) scale(${Math.min(scale + 0.05, 0.9)}) rotate(${rotation}deg)` 
                          : `translateX(${translateX}px) scale(${scale}) rotate(${rotation}deg)`,
                        zIndex: !isCenter && images.length > 1 ? zIndex + 50 : zIndex,
                        boxShadow: !isCenter && images.length > 1 
                          ? '0 20px 40px rgba(0,0,0,0.4)' 
                          : (isCenter ? '0 25px 50px rgba(0,0,0,0.3)' : '0 15px 30px rgba(0,0,0,0.2)')
                      }
                    }}
                    onClick={() => {
                      if (images.length > 1 && !isCenter && !isTransitioning) {
                        if (isLeft) handlePrevious();
                        else if (isRight) handleNext();
                        else if (isSide) {
                          handleImageClick(image.url, image.name, image.index);
                        }
                      }
                    }}
                    onDoubleClick={() => handleImageDoubleClick(image.index)}
                  >
                    <ImageWithAnnotations
                      imageUrl={image.url}
                      imageId={image.name}
                      width="100%"
                      height="100%"
                      alt={`Image ${image.displayNumber}`}
                      sx={{
                        position: 'absolute',
                        top: 0,
                        left: 0,
                        right: 0,
                        bottom: 0
                      }}
                    />
                    
                    {/* Image Number Badge */}
                    <Box
                      sx={{
                        position: 'absolute',
                        top: 12,
                        right: 12,
                        backgroundColor: isAnnotated ? "#4caf50" : isCenter ? 'rgba(0,0,0,0.8)' : 'rgba(0,0,0,0.7)',
                        color: 'white',
                        borderRadius: '50%',
                        width: 28,
                        height: 28,
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        fontSize: '0.75rem',
                        fontWeight: 'bold',
                        boxShadow: '0 2px 8px rgba(0,0,0,0.3)',
                        zIndex: 399
                      }}
                    >
                      {image.displayNumber}
                    </Box>

                    {/* Annotation Status Badge */}
                    <Box
                      sx={{
                        position: 'absolute',
                        bottom: 12,
                        right: 12,
                        backgroundColor: isAnnotated ? '#4caf50' : '#ff9800',
                        color: 'white',
                        borderRadius: '12px',
                        padding: '4px 8px',
                        fontSize: '0.6rem',
                        fontWeight: '600',
                        textTransform: 'uppercase',
                        letterSpacing: '0.5px',
                        boxShadow: isAnnotated 
                          ? '0 2px 8px rgba(76, 175, 80, 0.4)' 
                          : '0 2px 8px rgba(255, 152, 0, 0.4)',
                        zIndex: 399,
                        display: 'flex',
                        alignItems: 'center',
                        gap: '4px'
                      }}
                    >
                      {isAnnotated ? (
                        <>
                          <CheckCircle sx={{ fontSize: 10 }} />
                          Done
                        </>
                      ) : (
                        <>
                          <RadioButtonUnchecked sx={{ fontSize: 10 }} />
                          Pending
                        </>
                      )}
                    </Box>
                  </Card>
                );
              })}
            </Box>

            {/* Footer with Progress Indicators */}
            {images.length > 1 && (
              <Box
                sx={{
                  position: 'absolute',
                  bottom: 20,
                  left: '50%',
                  transform: 'translateX(-50%)',
                  display: 'flex',
                  alignItems: 'center',
                  gap: 1,
                  backgroundColor: 'rgba(102, 126, 234, 0.1)',
                  backdropFilter: 'blur(10px)',
                  borderRadius: 3,
                  padding: 2,
                  zIndex: 500
                }}
              >
                {images.map((_, index) => (
                  <Box
                    key={index}
                    sx={{
                      width: index === localCurrentIndex ? 12 : 8,
                      height: index === localCurrentIndex ? 12 : 8,
                      borderRadius: '50%',
                      backgroundColor: index === localCurrentIndex ? '#667eea' : 'rgba(102, 126, 234, 0.4)',
                      transition: 'all 0.3s ease',
                      cursor: 'pointer'
                    }}
                    onClick={() => {
                      if (!isTransitioning) {
                        setIsTransitioning(true);
                        setTimeout(() => {
                          handleImageClick(images[index].url, images[index].name, index);
                          setTimeout(() => setIsTransitioning(false), 100);
                        }, 200);
                      }
                    }}
                  />
                ))}
              </Box>
            )}

            {/* Keyboard Shortcuts Info */}
            <Box
              sx={{
                position: 'absolute',
                bottom: 20,
                right: 20,
                color: 'rgba(102, 126, 234, 0.6)',
                fontSize: '0.7rem',
                textAlign: 'right',
                zIndex: 400
              }}
            >
              <Typography variant="caption" sx={{ display: 'block' }}>
                ← → Navigate • F Fullscreen
              </Typography>
            </Box>
          </Box>
        )}
      </Box>

      {/* Fullscreen Modal */}
      {renderFullscreenModal()}
    </MaxCustomaizer>
  );
}