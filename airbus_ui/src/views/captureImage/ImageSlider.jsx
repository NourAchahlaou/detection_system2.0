// components/ImageSlider.jsx
import React, { useState, useEffect, useCallback, useRef } from 'react';
import { 
  Box, 
  IconButton, 
  Typography, 
  Card,
  CardMedia,
  Tooltip,
  CircularProgress 
} from '@mui/material';
import { 
  KeyboardArrowUp, 
  KeyboardArrowDown, 
  Photo,
  Refresh 
} from '@mui/icons-material';
import { cameraService } from './CameraService';

const ImageSlider = ({ targetLabel }) => {
  const [images, setImages] = useState([]);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  
  // Use refs to track state without causing re-renders
  const intervalRef = useRef(null);
  const mountedRef = useRef(true);
  const lastFetchTimeRef = useRef(0);

  // Stable fetch function with proper error handling
  const fetchImages = useCallback(async (immediate = false) => {
    const now = Date.now();
    const timeSinceLastFetch = now - lastFetchTimeRef.current;
    
    // Rate limiting: don't fetch too frequently unless immediate
    if (!immediate && timeSinceLastFetch < 2000) {
      return;
    }

    if (!targetLabel || targetLabel.trim() === '') {
      setImages([]);
      setCurrentIndex(0);
      return;
    }

    try {
      setLoading(true);
      setError(null);
      lastFetchTimeRef.current = now;

      const serverTempImages = await cameraService.getImagesByLabel(targetLabel);
      const validImages = serverTempImages.filter(img => img.url || img.src);

      // Only update if component is still mounted
      if (mountedRef.current) {
        setImages(prevImages => {
          // Only update if the number of images changed or if we have no images
          if (prevImages.length !== validImages.length || prevImages.length === 0) {
            return validImages;
          }
          return prevImages;
        });
        
        setCurrentIndex(prevIndex => {
          if (prevIndex >= validImages.length && validImages.length > 0) {
            return 0;
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
    if (targetLabel && targetLabel.trim() !== '') {
      fetchImages(true); // Immediate fetch on label change
    } else {
      setImages([]);
      setCurrentIndex(0);
    }
  }, [targetLabel, fetchImages]);

  // Auto-refresh with polling - Fixed to prevent infinite loops
  useEffect(() => {
    // Clear existing interval
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }

    if (!targetLabel || targetLabel.trim() === '') {
      return;
    }

    // Set up new interval
    intervalRef.current = setInterval(() => {
      if (mountedRef.current) {
        fetchImages(false);
      }
    }, 5000);

    // Cleanup function
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    };
  }, [targetLabel, fetchImages]);

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
      
      // Clear intervals
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    };
  }, []); // Empty dependency array - only run on mount/unmount

  // Manual refresh function
  const handleRefresh = useCallback(() => {
    fetchImages(true); // Force immediate refresh
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

  // Get visible images (current, previous, next)
  const getVisibleImages = useCallback(() => {
    if (images.length === 0) return [];
    
    const visibleImages = [];
    const totalImages = images.length;
    
    if (totalImages === 1) {
      visibleImages.push({ ...images[currentIndex], position: 'middle', index: currentIndex });
    } else if (totalImages === 2) {
      const otherIndex = currentIndex === 0 ? 1 : 0;
      visibleImages.push({ ...images[otherIndex], position: 'top', index: otherIndex });
      visibleImages.push({ ...images[currentIndex], position: 'middle', index: currentIndex });
    } else {
      const prevIndex = currentIndex > 0 ? currentIndex - 1 : totalImages - 1;
      visibleImages.push({ ...images[prevIndex], position: 'top', index: prevIndex });
      
      visibleImages.push({ ...images[currentIndex], position: 'middle', index: currentIndex });
      
      const nextIndex = currentIndex < totalImages - 1 ? currentIndex + 1 : 0;
      visibleImages.push({ ...images[nextIndex], position: 'bottom', index: nextIndex });
    }
    
    return visibleImages;
  }, [images, currentIndex]);

  const visibleImages = getVisibleImages();
  const imageSource = (image) => image.url || image.src;

  return (
    <Box
      sx={{
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'flex-start',
        position: 'relative',
        backgroundColor: '#f5f5f5',
        borderRadius: 2,
        padding: 2,
        minHeight: '500px'
      }}
    >
      {/* Header */}
      <Box
        sx={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          width: '100%',
          mb: 2
        }}
      >
        <Typography variant="h6" sx={{ color: '#666' }}>
          Captured Images
        </Typography>
        <Tooltip title="Refresh Images">
          <IconButton onClick={handleRefresh} size="small" disabled={loading}>
            <Refresh />
          </IconButton>
        </Tooltip>
      </Box>

      {/* Loading State */}
      {loading && (
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, flex: 1, justifyContent: 'center' }}>
          <CircularProgress size={24} />
          <Typography variant="body2" color="textSecondary">
            Loading images...
          </Typography>
        </Box>
      )}

      {/* Error State */}
      {error && !loading && (
        <Box sx={{ textAlign: 'center', color: 'error.main', flex: 1, display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
          <Photo sx={{ fontSize: 48, opacity: 0.5, mb: 1 }} />
          <Typography variant="body2">{error}</Typography>
          <IconButton onClick={handleRefresh} sx={{ mt: 1 }}>
            <Refresh />
          </IconButton>
        </Box>
      )}

      {/* No Images State */}
      {!loading && !error && images.length === 0 && (
        <Box sx={{ textAlign: 'center', color: 'text.secondary', flex: 1, display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
          <Photo sx={{ fontSize: 48, opacity: 0.5, mb: 1 }} />
          <Typography variant="body2">
            {targetLabel ? `No images captured for "${targetLabel}"` : 'Enter a piece label and start capturing'}
          </Typography>
        </Box>
      )}

      {/* Image Slider */}
      {!loading && !error && images.length > 0 && (
        <Box
          sx={{
            flex: 1,
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            width: '100%',
            position: 'relative'
          }}
        >
          {/* Up Arrow - only show if more than 1 image */}
          {images.length > 1 && (
            <IconButton
              onClick={handlePrevious}
              sx={{
                mb: 1,
                backgroundColor: 'primary.main',
                color: 'white',
                '&:hover': {
                  backgroundColor: 'primary.dark'
                },
                boxShadow: 2
              }}
            >
              <KeyboardArrowUp />
            </IconButton>
          )}

          {/* Images Container */}
          <Box
            sx={{
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              gap: 2,
              width: '100%',
              maxWidth: '300px'
            }}
          >
            {visibleImages.map((image, index) => (
              <Card
                key={`${image.index}-${image.position}`}
                sx={{
                  width: image.position === 'middle' ? '100%' : '80%',
                  opacity: image.position === 'middle' ? 1 : 0.6,
                  transform: image.position === 'middle' ? 'scale(1)' : 'scale(0.9)',
                  transition: 'all 0.3s ease-in-out',
                  cursor: image.position !== 'middle' && images.length > 1 ? 'pointer' : 'default',
                  boxShadow: image.position === 'middle' ? 4 : 2,
                  border: image.isTemporary ? '2px solid #2196f3' : 'none',
                  position: 'relative',
                  '&:hover': {
                    opacity: image.position !== 'middle' && images.length > 1 ? 0.8 : 1
                  }
                }}
                onClick={() => {
                  if (images.length > 1) {
                    if (image.position === 'top') handlePrevious();
                    if (image.position === 'bottom') handleNext();
                  }
                }}
              >
                <CardMedia
                  component="img"
                  height={image.position === 'middle' ? '200' : '150'}
                  image={imageSource(image)}
                  alt={`Captured image ${image.index + 1}`}
                  sx={{
                    objectFit: 'cover'
                  }}
                  onError={(e) => {
                    console.error('Image failed to load:', imageSource(image));
                    e.target.src = 'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjAwIiBoZWlnaHQ9IjIwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMTAwJSIgaGVpZ2h0PSIxMDAlIiBmaWxsPSIjZGRkIi8+PHRleHQgeD0iNTAlIiB5PSI1MCUiIGZvbnQtZmFtaWx5PSJBcmlhbCwgc2Fucy1zZXJpZiIgZm9udC1zaXplPSIxNCIgZmlsbD0iIzk5OSIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZHk9Ii4zZW0iPkltYWdlIG5vdCBmb3VuZDwvdGV4dD48L3N2Zz4=';
                  }}
                />
                {/* Indicator for temporary images */}
                {image.isTemporary && (
                  <Box
                    sx={{
                      position: 'absolute',
                      top: 8,
                      right: 8,
                      backgroundColor: 'primary.main',
                      color: 'white',
                      borderRadius: '50%',
                      width: 20,
                      height: 20,
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      fontSize: '12px',
                      fontWeight: 'bold'
                    }}
                  >
                    T
                  </Box>
                )}
              </Card>
            ))}
          </Box>

          {/* Down Arrow - only show if more than 1 image */}
          {images.length > 1 && (
            <IconButton
              onClick={handleNext}
              sx={{
                mt: 1,
                backgroundColor: 'primary.main',
                color: 'white',
                '&:hover': {
                  backgroundColor: 'primary.dark'
                },
                boxShadow: 2
              }}
            >
              <KeyboardArrowDown />
            </IconButton>
          )}

          {/* Image Counter */}
          <Typography
            variant="caption"
            sx={{
              mt: 2,
              px: 2,
              py: 0.5,
              backgroundColor: 'rgba(0, 0, 0, 0.7)',
              color: 'white',
              borderRadius: 1,
              textAlign: 'center'
            }}
          >
            {currentIndex + 1} of {images.length}
            {images.some(img => img.isTemporary) && (
              <Typography variant="caption" sx={{ display: 'block', fontSize: '10px', opacity: 0.8 }}>
                Blue border = temporary
              </Typography>
            )}
          </Typography>
        </Box>
      )}
    </Box>
  );
};

export default ImageSlider;