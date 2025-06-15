import { useEffect, useState, useCallback, useRef } from "react";
import { 
  Box, 
  Typography, 
  styled, 
  Fade, 
  CircularProgress,
  IconButton,
  Card
} from "@mui/material";
import { 
  Photo, 
  KeyboardArrowUp, 
  KeyboardArrowDown,
  CheckCircle,
  RadioButtonUnchecked
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

export default function SidenavImageDisplay({
  pieceLabel,
  onImageSelect,
  onFirstImageLoad,
  onImageCountUpdate,
  onImagesLoaded,
  currentImageIndex,
  refreshTrigger,
  // NEW: Add these props to handle image status updates
  onImageStatusUpdate, // Callback to notify parent when image status changes
  imageStatusUpdates = {} // Object with imageId -> hasAnnotations mapping
}) {
  const [images, setImages] = useState([]);
  const [localCurrentIndex, setLocalCurrentIndex] = useState(0);
  const [loading, setLoading] = useState(false);
  const [selectedImageUrl, setSelectedImageUrl] = useState('');
  // NEW: Store original order mapping
  const [originalOrderMap, setOriginalOrderMap] = useState(new Map());
  
  const mountedRef = useRef(true);
  const lastPieceLabelRef = useRef(null);

  // NEW: Effect to handle real-time image status updates
  useEffect(() => {
    if (Object.keys(imageStatusUpdates).length > 0) {
      setImages(prevImages => 
        prevImages.map(image => {
          const imageId = image.name || image.id;
          if (imageId in imageStatusUpdates) {
            const hasAnnotations = imageStatusUpdates[imageId];
            console.log(`SidenavImageDisplay: Updating status for image ${imageId}: ${hasAnnotations ? 'annotated' : 'not annotated'}`);
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
    
    // First, handle images that already have assigned numbers
    const usedNumbers = new Set(Array.from(updatedOrderMap.values()));
    
    newImages.forEach((image, index) => {
      const imageKey = image.name || image.id || image.url; // Use a unique identifier
      
      if (updatedOrderMap.has(imageKey)) {
        // Image already has a number assigned
        numberedImages.push({
          ...image,
          originalIndex: index,
          displayNumber: updatedOrderMap.get(imageKey)
        });
      } else {
        // New image - assign next available number
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
    
    // Sort by display number to maintain consistent ordering in UI
    numberedImages.sort((a, b) => a.displayNumber - b.displayNumber);
    
    return { numberedImages, updatedOrderMap };
  };

  // NEW: Function to update individual image annotation status
  const updateImageAnnotationStatus = useCallback(async (imageId, hasAnnotations) => {
    console.log(`SidenavImageDisplay: Received status update for image ${imageId}: ${hasAnnotations}`);
    
    setImages(prevImages => 
      prevImages.map(image => {
        if ((image.name || image.id) === imageId) {
          console.log(`SidenavImageDisplay: Updating image ${imageId} status from ${image.is_annotated} to ${hasAnnotations}`);
          return {
            ...image,
            is_annotated: hasAnnotations
          };
        }
        return image;
      })
    );

    // Notify parent component if callback is provided
    if (onImageStatusUpdate) {
      onImageStatusUpdate(imageId, hasAnnotations);
    }
  }, [onImageStatusUpdate]);

  useEffect(() => {
    if (refreshTrigger > 0 && pieceLabel) {
      console.log('SidenavImageDisplay: Refresh trigger activated, refetching images...');
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
      setOriginalOrderMap(new Map()); // Reset order map
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
      
      console.log('SidenavImageDisplay: Fetching images for piece:', pieceLabel);
      
      const response = await api.get(`/api/annotation/annotations/get_images_of_piece/${pieceLabel}`);
      const rawData = response.data;
      
      console.log('SidenavImageDisplay: Received images data:', rawData);
      
      if (mountedRef.current) {
        // NEW: Assign stable numbers and maintain order
        const { numberedImages, updatedOrderMap } = assignStableNumbers(
          rawData, 
          refreshTrigger > 0 ? originalOrderMap : new Map()
        );
        
        setImages(numberedImages);
        setOriginalOrderMap(updatedOrderMap);
        
        // Don't reset current index if we're just refreshing
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

        // Only set first image if this is initial load, not a refresh
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
          // After refresh, maintain current selection if possible
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
      // UPDATED: Pass the updateImageAnnotationStatus callback
      onImageSelect(imageUrl, imageId, index, updateImageAnnotationStatus);
    }
  };

  const handlePrevious = useCallback(() => {
    if (images.length > 0) {
      const newIndex = localCurrentIndex > 0 ? localCurrentIndex - 1 : images.length - 1;
      const newImage = images[newIndex];
      
      setLocalCurrentIndex(newIndex);
      setSelectedImageUrl(newImage.url);
      
      if (onImageSelect) {
        // UPDATED: Pass the updateImageAnnotationStatus callback
        onImageSelect(newImage.url, newImage.name, newIndex, updateImageAnnotationStatus);
      }
    }
  }, [images, localCurrentIndex, onImageSelect, updateImageAnnotationStatus]);

  const handleNext = useCallback(() => {
    if (images.length > 0) {
      const newIndex = localCurrentIndex < images.length - 1 ? localCurrentIndex + 1 : 0;
      const newImage = images[newIndex];
      
      setLocalCurrentIndex(newIndex);
      setSelectedImageUrl(newImage.url);
      
      if (onImageSelect) {
        // UPDATED: Pass the updateImageAnnotationStatus callback
        onImageSelect(newImage.url, newImage.name, newIndex, updateImageAnnotationStatus);
      }
    }
  }, [images, localCurrentIndex, onImageSelect, updateImageAnnotationStatus]);

  const getVisibleImages = () => {
    if (images.length === 0) return [];
    
    if (images.length === 1) {
      return [{ ...images[0], position: 'center', index: 0 }];
    }
    
    if (images.length === 2) {
      const otherIndex = localCurrentIndex === 0 ? 1 : 0;
      return [
        { ...images[otherIndex], position: 'back', index: otherIndex },
        { ...images[localCurrentIndex], position: 'center', index: localCurrentIndex }
      ];
    }
    
    const prevIndex = localCurrentIndex > 0 ? localCurrentIndex - 1 : images.length - 1;
    const nextIndex = localCurrentIndex < images.length - 1 ? localCurrentIndex + 1 : 0;
    
    return [
      { ...images[prevIndex], position: 'back-top', index: prevIndex },
      { ...images[nextIndex], position: 'back-bottom', index: nextIndex },
      { ...images[localCurrentIndex], position: 'center', index: localCurrentIndex }
    ];
  };

  const visibleImages = getVisibleImages();

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
            {images.length > 1 && (
              <IconButton
                onClick={handlePrevious}
                sx={{
                  position: 'absolute',
                  top: 0,
                  zIndex: 400,
                  backgroundColor: 'transparent !important',
                  color: '#667eea',
                  border: 'none',
                  boxShadow: 'none !important',
                  '&:hover': {
                    backgroundColor: 'transparent !important',
                    color: 'white',
                    transform: 'translateY(-2px)',
                    boxShadow: 'none !important',
                  },
                  transition: 'all 0.3s ease'
                }}
                disableRipple
                disableFocusRipple
                disableTouchRipple
              >
                <KeyboardArrowUp />
              </IconButton>
            )}

            <Box
              sx={{
                position: 'relative',
                width: '300px',
                height: '420px',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center'
              }}
            >
              {visibleImages.map((image) => {
                const isCenter = image.position === 'center';
                const isBackTop = image.position === 'back-top';
                const isBackBottom = image.position === 'back-bottom';
                const isBack = image.position === 'back';
                const isAnnotated = image.is_annotated === true;

                let translateY = 0;
                let translateX = 0;
                let zIndex = 100;
                let scale = 0.85;
                let opacity = 0.85;
                let rotation = 0;

                if (isCenter) {
                  translateY = 0;
                  translateX = 0;
                  zIndex = 300;
                  scale = 1;
                  opacity = 1;
                  rotation = 0;
                } else if (isBackTop) {
                  translateY = -120;
                  translateX = 0;
                  zIndex = 200;
                  scale = 0.88;
                  opacity = 0.5;
                  rotation = 0;
                } else if (isBackBottom) {
                  translateY = 120;
                  translateX = 0;
                  zIndex = 100;
                  scale = 0.88;
                  opacity = 0.5;
                  rotation = 0;
                } else if (isBack) {
                  translateY = 80;
                  translateX = 0;
                  zIndex = 200;
                  scale = 0.9;
                  opacity = 0.5;
                  rotation = 0;
                }

                return (
                  <Card
                    key={`${image.index}-${image.position}-${refreshTrigger}-${image.is_annotated ? 'annotated' : 'pending'}`}
                    sx={{
                      position: 'absolute',
                      width: '280px',
                      height: '200px',
                      borderRadius: 3,
                      overflow: 'hidden',
                      cursor: !isCenter && images.length > 1 ? 'pointer' : 'default',
                      transition: 'all 0.5s cubic-bezier(0.4, 0, 0.2, 1)',
                      transform: `translateY(${translateY}px) translateX(${translateX}px) scale(${scale}) rotate(${rotation}deg)`,
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
                        opacity: !isCenter && images.length > 1 ? 1 : opacity,
                        transform: !isCenter && images.length > 1 
                          ? `translateY(${translateY}px) translateX(${translateX}px) scale(${Math.min(scale + 0.03, 0.95)}) rotate(${rotation}deg)` 
                          : `translateY(${translateY}px) translateX(${translateX}px) scale(${scale}) rotate(${rotation}deg)`,
                        zIndex: !isCenter && images.length > 1 ? zIndex + 50 : zIndex,
                        boxShadow: !isCenter && images.length > 1 
                          ? '0 20px 40px rgba(0,0,0,0.25)' 
                          : (isCenter ? '0 25px 50px rgba(0,0,0,0.3)' : '0 15px 30px rgba(0,0,0,0.2)')
                      }
                    }}
                    onClick={() => {
                      if (images.length > 1) {
                        if (isBackTop) {
                          handlePrevious();
                        } else if (isBackBottom) {
                          handleNext();
                        } else if (isBack && !isCenter) {
                          handleImageClick(image.url, image.name, image.index);
                        }
                      }
                    }}
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
                    
                    {/* CHANGED: Use displayNumber instead of index + 1 */}
                    <Box
                      sx={{
                        position: 'absolute',
                        top: 12,
                        right: 12,
                        backgroundColor: isAnnotated ? "#4caf50" : isCenter ? 'rgba(0,0,0,0.8)' : 'rgba(0,0,0,0.7)',
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
                        zIndex: 399
                      }}
                    >
                      {image.displayNumber}
                    </Box>

                    <Box
                      sx={{
                        position: 'absolute',
                        bottom: 12,
                        right: 12,
                        backgroundColor: isAnnotated ? '#4caf50' : '#ff9800',
                        color: 'white',
                        borderRadius: '12px',
                        padding: '4px 12px',
                        fontSize: '0.65rem',
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
                          <CheckCircle sx={{ fontSize: 12 }} />
                          Done
                        </>
                      ) : (
                        <>
                          <RadioButtonUnchecked sx={{ fontSize: 12 }} />
                          Pending
                        </>
                      )}
                    </Box>
                  </Card>
                );
              })}
            </Box>
            
            {images.length > 1 && (
              <IconButton
                onClick={handleNext}
                sx={{
                  position: 'absolute',
                  bottom: 0,
                  zIndex: 400,
                  color: '#667eea',
                  backgroundColor: 'transparent !important',
                  border: 'none',
                  boxShadow: 'none !important',
                  '&:hover': {
                    color: 'white',
                    transform: 'translateY(2px)',
                    backgroundColor: 'transparent !important',
                    boxShadow: 'none !important',
                  },
                  transition: 'all 0.3s ease'
                }}
                disableRipple
                disableFocusRipple
                disableTouchRipple
              >
                <KeyboardArrowDown />
              </IconButton>
            )}
          </Box>
        )}
      </Box>
    </MaxCustomaizer>
  );
}