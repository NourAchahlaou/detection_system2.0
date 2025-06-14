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
import ImageWithAnnotations from "./ImageWithAnnotations"; // Adjust path as needed

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
  currentImageIndex
}) {
  const [images, setImages] = useState([]);
  const [localCurrentIndex, setLocalCurrentIndex] = useState(0);
  const [loading, setLoading] = useState(false);
  const [selectedImageUrl, setSelectedImageUrl] = useState('');
  
  const mountedRef = useRef(true);
  const lastPieceLabelRef = useRef(null);

  // Keep all your existing useEffect and handler functions exactly the same
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
      if (onImageCountUpdate) {
        onImageCountUpdate(0);
      }
      return;
    }

    if (lastPieceLabelRef.current === pieceLabel) {
      return;
    }

    try {
      setLoading(true);
      lastPieceLabelRef.current = pieceLabel;
      
      const response = await api.get(`/api/annotation/annotations/get_images_of_piece/${pieceLabel}`);
      const data = response.data;
      
      if (mountedRef.current) {
        setImages(data);
        setLocalCurrentIndex(0);
        setSelectedImageUrl('');

        if (onImageCountUpdate) {
          onImageCountUpdate(data.length);
        }

        if (onImagesLoaded) {
          onImagesLoaded(data);
        }

        if (data.length > 0) {
          const firstImage = data[0];
          setSelectedImageUrl(firstImage.url);
          
          if (onFirstImageLoad) {
            onFirstImageLoad(firstImage.url, firstImage.name);
          }
          
          if (onImageSelect) {
            onImageSelect(firstImage.url, firstImage.name, 0);
          }
        }
      }
  
    } catch (error) {
      console.error("Error fetching images:", error.response?.data?.detail || error.message);
      if (mountedRef.current) {
        setImages([]);
        setLocalCurrentIndex(0);
        setSelectedImageUrl('');
        if (onImageCountUpdate) {
          onImageCountUpdate(0);
        }
      }
    } finally {
      if (mountedRef.current) {
        setLoading(false);
      }
    }
  }, [pieceLabel]);

  useEffect(() => {
    mountedRef.current = true;
    
    if (lastPieceLabelRef.current !== pieceLabel) {
      lastPieceLabelRef.current = null;
      fetchImages();
    }
    
    return () => {
      mountedRef.current = false;
    };
  }, [pieceLabel]);

  const handleImageClick = (imageUrl, imageId, index) => {
    setSelectedImageUrl(imageUrl);
    setLocalCurrentIndex(index);
    if (onImageSelect) {
      onImageSelect(imageUrl, imageId, index);
    }
  };

  const handlePrevious = useCallback(() => {
    if (images.length > 0) {
      const newIndex = localCurrentIndex > 0 ? localCurrentIndex - 1 : images.length - 1;
      const newImage = images[newIndex];
      
      setLocalCurrentIndex(newIndex);
      setSelectedImageUrl(newImage.url);
      
      if (onImageSelect) {
        onImageSelect(newImage.url, newImage.name, newIndex);
      }
    }
  }, [images, localCurrentIndex, onImageSelect]);

  const handleNext = useCallback(() => {
    if (images.length > 0) {
      const newIndex = localCurrentIndex < images.length - 1 ? localCurrentIndex + 1 : 0;
      const newImage = images[newIndex];
      
      setLocalCurrentIndex(newIndex);
      setSelectedImageUrl(newImage.url);
      
      if (onImageSelect) {
        onImageSelect(newImage.url, newImage.name, newIndex);
      }
    }
  }, [images, localCurrentIndex, onImageSelect]);

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
        {/* Keep existing loading and empty states */}
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

        {/* Updated Image Slider with annotations */}
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
            {/* Keep existing navigation arrows */}
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

            {/* Images Stack Container */}
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

                // Keep all your existing positioning calculations
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
                    key={`${image.index}-${image.position}`}
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
                    {/* Replace CardMedia with ImageWithAnnotations component */}
                    <ImageWithAnnotations
                      imageUrl={image.url}
                      imageId={image.name} // This should be the image ID
                      width="100%"
                      height="100%"
                      alt={`Image ${image.index + 1}`}
                      sx={{
                        position: 'absolute',
                        top: 0,
                        left: 0,
                        right: 0,
                        bottom: 0
                      }}
                    />
                    
                    {/* Keep existing badges */}
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
                      {image.index + 1}
                    </Box>

                    <Box
                      sx={{
                        position: 'absolute',
                        bottom: 12,
                        right: 12, // Changed from left to right to avoid overlap with annotation count
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
            {/* Navigation Arrow - Down */}
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