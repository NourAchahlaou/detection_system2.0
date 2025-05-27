// components/ImageSlider.jsx
import React, { useState, useEffect } from 'react';
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
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Fetch images for the current piece label
  const fetchImages = async () => {
    if (!targetLabel || targetLabel.trim() === '') {
      setImages([]);
      setLoading(false);
      return;
    }

    try {
      setLoading(true);
      setError(null);
      // You'll need to implement this endpoint in your backend
      const response = await cameraService.getImagesByLabel(targetLabel);
      setImages(response || []);
      setCurrentIndex(0);
    } catch (err) {
      console.error('Error fetching images:', err);
      setError('Failed to load images');
      setImages([]);
    } finally {
      setLoading(false);
    }
  };

  // Refresh images
  const handleRefresh = () => {
    fetchImages();
  };

  // Navigate up in the slider
  const handlePrevious = () => {
    if (images.length > 0) {
      setCurrentIndex((prev) => (prev > 0 ? prev - 1 : images.length - 1));
    }
  };

  // Navigate down in the slider
  const handleNext = () => {
    if (images.length > 0) {
      setCurrentIndex((prev) => (prev < images.length - 1 ? prev + 1 : 0));
    }
  };

  // Fetch images when targetLabel changes
  useEffect(() => {
    const fetchImagesAsync = async () => {
      if (!targetLabel || targetLabel.trim() === '') {
        setImages([]);
        setLoading(false);
        return;
      }

      try {
        setLoading(true);
        setError(null);
        // You'll need to implement this endpoint in your backend
        const response = await cameraService.getImagesByLabel(targetLabel);
        setImages(response || []);
        setCurrentIndex(0);
      } catch (err) {
        console.error('Error fetching images:', err);
        setError('Failed to load images');
        setImages([]);
      } finally {
        setLoading(false);
      }
    };

    fetchImagesAsync();
  }, [targetLabel]);

  // Get visible images (current, previous, next)
  const getVisibleImages = () => {
    if (images.length === 0) return [];
    
    const visibleImages = [];
    const totalImages = images.length;
    
    // Previous image (top)
    const prevIndex = currentIndex > 0 ? currentIndex - 1 : totalImages - 1;
    visibleImages.push({ ...images[prevIndex], position: 'top', index: prevIndex });
    
    // Current image (middle)
    visibleImages.push({ ...images[currentIndex], position: 'middle', index: currentIndex });
    
    // Next image (bottom)
    const nextIndex = currentIndex < totalImages - 1 ? currentIndex + 1 : 0;
    visibleImages.push({ ...images[nextIndex], position: 'bottom', index: nextIndex });
    
    return visibleImages;
  };

  const visibleImages = getVisibleImages();

  return (
    <Box
      sx={{
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
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
          <IconButton onClick={handleRefresh} size="small">
            <Refresh />
          </IconButton>
        </Tooltip>
      </Box>

      {/* Loading State */}
      {loading && (
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
          <CircularProgress size={24} />
          <Typography variant="body2" color="textSecondary">
            Loading images...
          </Typography>
        </Box>
      )}

      {/* Error State */}
      {error && !loading && (
        <Box sx={{ textAlign: 'center', color: 'error.main' }}>
          <Photo sx={{ fontSize: 48, opacity: 0.5, mb: 1 }} />
          <Typography variant="body2">{error}</Typography>
        </Box>
      )}

      {/* No Images State */}
      {!loading && !error && images.length === 0 && (
        <Box sx={{ textAlign: 'center', color: 'text.secondary' }}>
          <Photo sx={{ fontSize: 48, opacity: 0.5, mb: 1 }} />
          <Typography variant="body2">
            {targetLabel ? `No images found for "${targetLabel}"` : 'Enter a piece label to view images'}
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
          {/* Up Arrow */}
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
                  cursor: image.position !== 'middle' ? 'pointer' : 'default',
                  boxShadow: image.position === 'middle' ? 4 : 2,
                  '&:hover': {
                    opacity: image.position !== 'middle' ? 0.8 : 1
                  }
                }}
                onClick={() => {
                  if (image.position === 'top') handlePrevious();
                  if (image.position === 'bottom') handleNext();
                }}
              >
                <CardMedia
                  component="img"
                  height={image.position === 'middle' ? '200' : '150'}
                  image={image.url || image.src}
                  alt={`Captured image ${image.index + 1}`}
                  sx={{
                    objectFit: 'cover'
                  }}
                />
              </Card>
            ))}
          </Box>

          {/* Down Arrow */}
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
          </Typography>
        </Box>
      )}
    </Box>
  );
};

export default ImageSlider;