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
  Chip,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button
} from '@mui/material';
import { 
  KeyboardArrowUp, 
  KeyboardArrowDown, 
  Photo,
  Refresh,
  Delete
} from '@mui/icons-material';
import { cameraService } from './CameraService';

const ImageSlider = ({ targetLabel, refreshTrigger }) => {
  const [images, setImages] = useState([]);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [imageToDelete, setImageToDelete] = useState(null);
  const [deleting, setDeleting] = useState(false);
  
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

  // Handle delete image
  const handleDeleteImage = useCallback(async (imageToDelete) => {
    if (!imageToDelete || !targetLabel) return;

    try {
      setDeleting(true);
      const success = await cameraService.deleteTempImage(targetLabel, imageToDelete.image_name);
      
      if (success) {
        // Remove the image from local state
        setImages(prevImages => {
          const newImages = prevImages.filter(img => img.image_name !== imageToDelete.image_name);
          
          // Adjust current index if necessary
          setCurrentIndex(prevIndex => {
            if (newImages.length === 0) return 0;
            if (prevIndex >= newImages.length) return newImages.length - 1;
            return prevIndex;
          });
          
          return newImages;
        });
        
        // Clean up blob URL if it exists
        if (imageToDelete.url && imageToDelete.url.startsWith('blob:')) {
          URL.revokeObjectURL(imageToDelete.url);
        }
        
        console.log('Image deleted successfully');
      } else {
        setError('Failed to delete image');
      }
    } catch (err) {
      console.error('Error deleting image:', err);
      setError('Failed to delete image');
    } finally {
      setDeleting(false);
      setDeleteDialogOpen(false);
      setImageToDelete(null);
    }
  }, [targetLabel]);

  // Open delete confirmation dialog
  const openDeleteDialog = useCallback((image) => {
    setImageToDelete(image);
    setDeleteDialogOpen(true);
  }, []);

  // Close delete dialog
  const closeDeleteDialog = useCallback(() => {
    if (!deleting) {
      setDeleteDialogOpen(false);
      setImageToDelete(null);
    }
  }, [deleting]);

  // Get image source
  const imageSource = (image) => image.url || image.src;

  // Get visible images for the stack effect
  const getVisibleImages = () => {
    if (images.length === 0) return [];
    
    if (images.length === 1) {
      return [{ ...images[0], position: 'center', index: 0 }];
    }
    
    if (images.length === 2) {
      const otherIndex = currentIndex === 0 ? 1 : 0;
      return [
        { ...images[otherIndex], position: 'back', index: otherIndex },
        { ...images[currentIndex], position: 'center', index: currentIndex }
      ];
    }
    
    // For 3 or more images
    const prevIndex = currentIndex > 0 ? currentIndex - 1 : images.length - 1;
    const nextIndex = currentIndex < images.length - 1 ? currentIndex + 1 : 0;
    
    return [
      { ...images[prevIndex], position: 'back-top', index: prevIndex },
      { ...images[nextIndex], position: 'back-bottom', index: nextIndex },
      { ...images[currentIndex], position: 'center', index: currentIndex }
    ];
  };

  const visibleImages = getVisibleImages();

  return (
    <Box
      sx={{
        height: '550px',
        display: 'flex',
        flexDirection: 'column',
        position: 'relative',
        backgroundColor: 'transparent',
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

        {/* Image Slider with Improved Vertical Stack */}
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
              padding: 2
            }}
          >
            {/* Navigation Arrow - Up */}
            {images.length > 1 && (
              <IconButton
                onClick={handlePrevious}
                sx={{
                  position: 'absolute',
                  top: 30,
                  zIndex: 1500,
                  backgroundColor: 'rgba(255,255,255,0.95)',
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

                // Improved positioning calculations
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
                  // For 2-image case
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
                      border: image.isTemporary ? '3px solid #4CAF50' : '3px solid rgba(255,255,255,0.9)',
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
                      if (images.length > 1 && !isCenter) {
                        if (isBackTop) handlePrevious();
                        else if (isBackBottom) handleNext();
                        else if (isBack) {
                          setCurrentIndex(image.index);
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
                    
                    {/* Delete Button - Only show on hover and for temporary images */}
                    {image.isTemporary && (
                      <Box
                        sx={{
                          position: 'absolute',
                          top: '50%',
                          left: '50%',
                          transform: 'translate(-50%, -50%)',
                          opacity: 0,
                          transition: 'opacity 0.3s ease',
                          zIndex: 1000,
                          '.MuiCard-root:hover &': {
                            opacity: 1,
                          }
                        }}
                      >
                        <Tooltip title="Delete Image">
                          <IconButton
                            onClick={(e) => {
                              e.stopPropagation();
                              openDeleteDialog(image);
                            }}
                            sx={{
                              backgroundColor: 'rgba(244, 63, 94, 0.9)',
                              color: 'white',
                              boxShadow: '0 4px 12px rgba(244, 63, 94, 0.4)',
                              '&:hover': {
                                backgroundColor: 'rgba(244, 63, 94, 1)',
                                transform: 'scale(1.1)',
                                boxShadow: '0 6px 16px rgba(244, 63, 94, 0.5)',
                              },
                              transition: 'all 0.2s ease'
                            }}
                          >
                            <Delete />
                          </IconButton>
                        </Tooltip>
                      </Box>
                    )}
                    
                    {/* Image Number Badge */}
                    <Box
                      sx={{
                        position: 'absolute',
                        top: 12,
                        right: 12,
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
                          fontSize: '0.65rem',
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

            {/* Navigation Arrow - Down */}
            {images.length > 1 && (
              <IconButton
                onClick={handleNext}
                sx={{
                  position: 'absolute',
                  bottom: 30,
                  zIndex: 1500,
                  backgroundColor: 'rgba(255,255,255,0.95)',
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
            bottom: 10,
            left: '50%',
            transform: 'translateX(-50%)',
            background: 'rgba(255,255,255,0.95)',
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

      {/* Delete Confirmation Dialog */}
      <Dialog
        open={deleteDialogOpen}
        onClose={closeDeleteDialog}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle sx={{ color: '#333', pb: 1 }}>
          Delete Image
        </DialogTitle>
        <DialogContent>
          <Typography variant="body1" sx={{ mb: 2 }}>
            Are you sure you want to delete this image? This action cannot be undone.
          </Typography>
          {imageToDelete && (
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
              <Photo sx={{ color: '#666' }} />
              <Typography variant="body2" sx={{ color: '#666' }}>
                {imageToDelete.image_name}
              </Typography>
            </Box>
          )}
        </DialogContent>
        <DialogActions sx={{ px: 3, pb: 2 }}>
          <Button 
            onClick={closeDeleteDialog} 
            disabled={deleting}
            sx={{ color: '#666' }}
          >
            Cancel
          </Button>
          <Button
            onClick={() => handleDeleteImage(imageToDelete)}
            variant="contained"
            disabled={deleting}
            sx={{
              backgroundColor: '#f43f5e',
              '&:hover': { backgroundColor: '#e11d48' },
              '&:disabled': { backgroundColor: '#fca5a5' }
            }}
            startIcon={deleting ? <CircularProgress size={16} color="inherit" /> : <Delete />}
          >
            {deleting ? 'Deleting...' : 'Delete'}
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default ImageSlider;