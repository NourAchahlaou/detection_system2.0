// components/ImageSlider.jsx
import React, { useState, useEffect, useCallback, useRef } from 'react';
import { 
  Box, 
  IconButton, 
  Typography, 
  Card,
  CardMedia,
  Tooltip,
  CircularProgress,
  Fade,
  Chip
} from '@mui/material';
import { 
  KeyboardArrowUp, 
  KeyboardArrowDown, 
  Photo,
  Refresh
} from '@mui/icons-material';
import { cameraService } from './CameraService';

const ImageSlider = ({ targetLabel, refreshTrigger }) => {
  const [images, setImages] = useState([]);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  
  const mountedRef = useRef(true);

  // Fetch images function
  const fetchImages = useCallback(async () => {
    if (!targetLabel || targetLabel.trim() === '') {
      setImages([]);
      setCurrentIndex(0);
      return;
    }

    try {
      setLoading(true);
      setError(null);

      const serverTempImages = await cameraService.getImagesByLabel(targetLabel);
      const validImages = serverTempImages.filter(img => img.url || img.src);

      if (mountedRef.current) {
        setImages(validImages);
        setCurrentIndex(prevIndex => {
          if (prevIndex >= validImages.length && validImages.length > 0) {
            return validImages.length - 1;
          }
          return prevIndex;
        });
      }
    } catch (err) {
      console.error('Error fetching images:', err);
      if (mountedRef.current) {
        setError('Failed to load images');
        setImages([]);
        setCurrentIndex(0);
      }
    } finally {
      if (mountedRef.current) {
        setLoading(false);
      }
    }
  }, [targetLabel]);

  // Initial fetch when targetLabel changes
  useEffect(() => {
    fetchImages();
  }, [targetLabel, fetchImages]);

  // React to refresh trigger (when new images are captured)
  useEffect(() => {
    if (refreshTrigger && targetLabel) {
      fetchImages();
    }
  }, [refreshTrigger, fetchImages, targetLabel]);

  // Component cleanup
  useEffect(() => {
    mountedRef.current = true;
    
    return () => {
      mountedRef.current = false;
      
      // Clean up blob URLs
      images.forEach(image => {
        if (image.url && image.url.startsWith('blob:')) {
          URL.revokeObjectURL(image.url);
        }
        if (image.src && image.src.startsWith('blob:')) {
          URL.revokeObjectURL(image.src);
        }
      });
    };
  }, []);

  // Manual refresh function
  const handleRefresh = useCallback(() => {
    fetchImages();
  }, [fetchImages]);

  // Navigate up in the slider
  const handlePrevious = useCallback(() => {
    if (images.length > 0) {
      setCurrentIndex((prev) => (prev > 0 ? prev - 1 : images.length - 1));
    }
  }, [images.length]);

  // Navigate down in the slider
  const handleNext = useCallback(() => {
    if (images.length > 0) {
      setCurrentIndex((prev) => (prev < images.length - 1 ? prev + 1 : 0));
    }
  }, [images.length]);

  // Get visible images (current, previous, next) - Memoized
  const visibleImages = React.useMemo(() => {
    if (images.length === 0) return [];
    
    const visibleImages = [];
    const totalImages = images.length;
    
    if (totalImages === 1) {
      visibleImages.push({ ...images[currentIndex], position: 'center', index: currentIndex });
    } else if (totalImages === 2) {
      const otherIndex = currentIndex === 0 ? 1 : 0;
      visibleImages.push({ ...images[otherIndex], position: 'back', index: otherIndex });
      visibleImages.push({ ...images[currentIndex], position: 'center', index: currentIndex });
    } else {
      const prevIndex = currentIndex > 0 ? currentIndex - 1 : totalImages - 1;
      const nextIndex = currentIndex < totalImages - 1 ? currentIndex + 1 : 0;
      
      visibleImages.push({ ...images[prevIndex], position: 'back-top', index: prevIndex });
      visibleImages.push({ ...images[nextIndex], position: 'back-bottom', index: nextIndex });
      visibleImages.push({ ...images[currentIndex], position: 'center', index: currentIndex });
    }
    
    return visibleImages;
  }, [images, currentIndex]);

  const imageSource = (image) => image.url || image.src;

  return (
    <Box
      sx={{
        height: '550px',
        display: 'flex',
        flexDirection: 'column',
        position: 'relative',
        backgroundColor: 'transparent',
        borderRadius: 3,
        overflow: 'hidden',
      }}
    >
      {/* Refresh Button - Top Right */}
      <Box
        sx={{
          position: 'absolute',
          top: 16,
          right: 16,
          zIndex: 2000,
        }}
      >
        <Tooltip title="Refresh Images">
          <IconButton 
            onClick={handleRefresh} 
            size="small" 
            disabled={loading}
            sx={{
              backgroundColor: 'rgba(255,255,255,0.9)',
              color: '#333',
              boxShadow: '0 4px 12px rgba(0,0,0,0.15)',
              '&:hover': {
                backgroundColor: 'white',
                transform: 'scale(1.05)',
              },
              '&:disabled': {
                backgroundColor: 'rgba(255,255,255,0.5)',
              }
            }}
          >
            <Refresh />
          </IconButton>
        </Tooltip>
      </Box>

      {/* Content Area */}
      <Box sx={{ flex: 1, position: 'relative', overflow: 'hidden' }}>

        {/* Loading State */}
        {loading && (
          <Fade in={loading}>
            <Box sx={{ 
              position: 'absolute',
              top: 0,
              left: 0,
              right: 0,
              bottom: 0,
              display: 'flex', 
              alignItems: 'center', 
              justifyContent: 'center',
              flexDirection: 'column',
              gap: 2,
              color: '#666'
            }}>
              <CircularProgress sx={{ color: '#667eea' }} size={32} />
              <Typography variant="body2" sx={{ opacity: 0.8 }}>
                Loading images...
              </Typography>
            </Box>
          </Fade>
        )}

        {/* Error State */}
        {error && !loading && (
          <Box sx={{ 
            position: 'absolute',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            color: '#666',
            textAlign: 'center'
          }}>
            <Photo sx={{ fontSize: 48, opacity: 0.6, mb: 2 }} />
            <Typography variant="body1" sx={{ mb: 2, opacity: 0.9 }}>
              {error}
            </Typography>
            <IconButton 
              onClick={handleRefresh}
              sx={{
                backgroundColor: 'rgba(255,255,255,0.9)',
                color: '#333',
                '&:hover': { backgroundColor: 'white' }
              }}
            >
              <Refresh />
            </IconButton>
          </Box>
        )}

        {/* No Images State */}
        {!loading && !error && images.length === 0 && (
          <Box sx={{ 
            position: 'absolute',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            color: '#666',
            textAlign: 'center'
          }}>
            <Photo sx={{ fontSize: 64, opacity: 0.4, mb: 3 }} />
            <Typography variant="h6" sx={{ mb: 1, opacity: 0.9 }}>
              No Images Yet
            </Typography>
            <Typography variant="body2" sx={{ opacity: 0.7, maxWidth: 200 }}>
              {targetLabel ? `No images captured for "${targetLabel}"` : 'Enter a piece label and start capturing'}
            </Typography>
          </Box>
        )}

        {/* Image Slider with Layered Effect */}
        {!loading && !error && images.length > 0 && (
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
              padding: 3
            }}
          >
            {/* Navigation Arrows */}
            {images.length > 1 && (
              <IconButton
                onClick={handlePrevious}
                sx={{
                  position: 'absolute',
                  top: 20,
                  zIndex: 1500,
                  backgroundColor: 'rgba(255,255,255,0.9)',
                  color: '#667eea',
                  boxShadow: '0 4px 12px rgba(0,0,0,0.15)',
                  '&:hover': {
                    backgroundColor: 'white',
                    transform: 'translateY(-2px)',
                    boxShadow: '0 6px 16px rgba(0,0,0,0.2)',
                  },
                  transition: 'all 0.3s ease'
                }}
              >
                <KeyboardArrowUp />
              </IconButton>
            )}

            {/* Images Container with Proper Layering */}
            <Box
              sx={{
                position: 'relative',
                width: '320px',
                height: '360px',
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

                return (
                  <Card
                    key={`${image.index}-${image.position}`}
                    sx={{
                      position: 'absolute',
                      width: isCenter ? '280px' : '240px',
                      height: isCenter ? '200px' : '160px',
                      borderRadius: 3,
                      overflow: 'hidden',
                      cursor: !isCenter && images.length > 1 ? 'pointer' : 'default',
                      transition: 'all 0.4s cubic-bezier(0.4, 0, 0.2, 1)',
                      transform: `
                        ${isCenter ? 'translateY(0px) translateX(0px) scale(1)' : 
                          isBackTop ? 'translateY(-60px) translateX(-30px) scale(0.85)' : 
                          isBackBottom ? 'translateY(60px) translateX(30px) scale(0.85)' :
                          isBack ? 'translateY(30px) translateX(-20px) scale(0.85)' : 'scale(0.85)'}
                      `,
                      zIndex: isCenter ? 200 : (isBackTop ? 100 : (isBackBottom ? 50 : 75)),
                      opacity: isCenter ? 1 : 0.8,
                      boxShadow: isCenter 
                        ? '0 20px 40px rgba(0,0,0,0.25)' 
                        : '0 10px 20px rgba(0,0,0,0.15)',
                      border: image.isTemporary ? '3px solid #4CAF50' : '3px solid rgba(255,255,255,0.8)',
                      '&:hover': {
                        opacity: !isCenter && images.length > 1 ? 1 : 1,
                        transform: !isCenter && images.length > 1 
                          ? `${isBackTop ? 'translateY(-55px) translateX(-25px)' : 
                               isBackBottom ? 'translateY(55px) translateX(25px)' : 
                               'translateY(25px) translateX(-15px)'} scale(0.9)`
                          : undefined,
                        zIndex: !isCenter && images.length > 1 ? 150 : undefined
                      }
                    }}
                    onClick={() => {
                      if (images.length > 1) {
                        if (isBackTop) handlePrevious();
                        else if (isBackBottom) handleNext();
                        else if (isBack) {
                          if (image.index !== currentIndex) {
                            setCurrentIndex(image.index);
                          }
                        }
                      }
                    }}
                  >
                    <CardMedia
                      component="img"
                      height="100%"
                      image={imageSource(image)}
                      alt={`Image ${image.index + 1}`}
                      sx={{
                        objectFit: 'cover',
                        width: '100%',
                        height: '100%'
                      }}
                      onError={(e) => {
                        console.error('Image failed to load:', imageSource(image));
                        e.target.src = 'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjgwIiBoZWlnaHQ9IjIwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMTAwJSIgaGVpZ2h0PSIxMDAlIiBmaWxsPSIjZjVmNWY1Ii8+PHRleHQgeD0iNTAlIiB5PSI1MCUiIGZvbnQtZmFtaWx5PSJBcmlhbCwgc2Fucy1zZXJpZiIgZm9udC1zaXplPSIxNCIgZmlsbD0iIzk5OSIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZHk9Ii4zZW0iPkltYWdlIG5vdCBmb3VuZDwvdGV4dD48L3N2Zz4=';
                      }}
                    />
                    
                    {/* Image Number Badge */}
                    <Box
                      sx={{
                        position: 'absolute',
                        top: 12,
                        right: 12,
                        backgroundColor: isCenter ? 'rgba(0,0,0,0.8)' : 'rgba(0,0,0,0.6)',
                        color: 'white',
                        borderRadius: '50%',
                        width: 28,
                        height: 28,
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        fontSize: '0.75rem',
                        fontWeight: 'bold',
                        boxShadow: '0 2px 8px rgba(0,0,0,0.3)'
                      }}
                    >
                      {image.index + 1}
                    </Box>

                    {/* Temporary Image Indicator */}
                    {image.isTemporary && (
                      <Box
                        sx={{
                          position: 'absolute',
                          top: 12,
                          left: 12,
                          backgroundColor: '#4CAF50',
                          color: 'white',
                          borderRadius: 1,
                          px: 1,
                          py: 0.5,
                          fontSize: '0.6rem',
                          fontWeight: 'bold',
                          textTransform: 'uppercase',
                          boxShadow: '0 2px 8px rgba(76, 175, 80, 0.3)'
                        }}
                      >
                        New
                      </Box>
                    )}
                  </Card>
                );
              })}
            </Box>

            {/* Navigation Arrows */}
            {images.length > 1 && (
              <IconButton
                onClick={handleNext}
                sx={{
                  position: 'absolute',
                  bottom: 20,
                  zIndex: 1500,
                  backgroundColor: 'rgba(255,255,255,0.9)',
                  color: '#667eea',
                  boxShadow: '0 4px 12px rgba(0,0,0,0.15)',
                  '&:hover': {
                    backgroundColor: 'white',
                    transform: 'translateY(2px)',
                    boxShadow: '0 6px 16px rgba(0,0,0,0.2)',
                  },
                  transition: 'all 0.3s ease'
                }}
              >
                <KeyboardArrowDown />
              </IconButton>
            )}
          </Box>
        )}
      </Box>

      {/* Footer with Image Counter */}
      {!loading && !error && images.length > 0 && (
        <Box
          sx={{
            position: 'absolute',
            bottom: 16,
            left: '50%',
            transform: 'translateX(-50%)',
            background: 'rgba(255,255,255,0.9)',
            backdropFilter: 'blur(10px)',
            borderRadius: 2,
            padding: 1.5,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            boxShadow: '0 4px 12px rgba(0,0,0,0.1)',
            zIndex: 1000
          }}
        >
          <Typography
            variant="body2"
            sx={{
              color: '#333',
              fontWeight: 500,
              display: 'flex',
              alignItems: 'center',
              gap: 1
            }}
          >
            <Box
              sx={{
                width: 8,
                height: 8,
                borderRadius: '50%',
                backgroundColor: '#667eea',
                opacity: 0.8
              }}
            />
            {currentIndex + 1} of {images.length}
            {images.some(img => img.isTemporary) && (
              <Chip
                label="New"
                size="small"
                sx={{
                  ml: 1,
                  backgroundColor: '#4CAF50',
                  color: 'white',
                  fontSize: '0.65rem',
                  height: 20
                }}
              />
            )}
          </Typography>
        </Box>
      )}
    </Box>
  );
};

export default ImageSlider;