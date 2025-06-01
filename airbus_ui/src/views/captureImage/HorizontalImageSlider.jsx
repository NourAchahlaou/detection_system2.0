// components/HorizontalImageSlider.jsx
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
  Button,
  Modal,
  Backdrop
} from '@mui/material';
import { 
  KeyboardArrowLeft, 
  KeyboardArrowRight, 
  Photo,
  Refresh,
  Delete,
  Close,
  Fullscreen,
  FullscreenExit
} from '@mui/icons-material';
import { cameraService } from './CameraService';

const HorizontalImageSlider = ({ 
  open, 
  onClose, 
  targetLabel, 
  initialIndex = 0, 
  onImageCountChange 
}) => {
  const [images, setImages] = useState([]);
  const [currentIndex, setCurrentIndex] = useState(initialIndex);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [imageToDelete, setImageToDelete] = useState(null);
  const [deleting, setDeleting] = useState(false);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [isTransitioning, setIsTransitioning] = useState(false);
  
  const mountedRef = useRef(true);

  // Fetch images function
  const fetchImages = useCallback(async () => {
    if (!targetLabel || targetLabel.trim() === '' || !open) {
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
        
        if (onImageCountChange) {
          onImageCountChange(validImages.length);
        }
      }
    } catch (err) {
      console.error('Error fetching images:', err);
      if (mountedRef.current) {
        setError('Failed to load images');
        setImages([]);
        setCurrentIndex(0);
        if (onImageCountChange) {
          onImageCountChange(0);
        }
      }
    } finally {
      if (mountedRef.current) {
        setLoading(false);
      }
    }
  }, [targetLabel, open, onImageCountChange]);

  // Set initial index when modal opens
  useEffect(() => {
    if (open) {
      setCurrentIndex(initialIndex);
      fetchImages();
    }
  }, [open, initialIndex, fetchImages]);

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

  // Keyboard navigation
  useEffect(() => {
    const handleKeyPress = (event) => {
      if (!open || isTransitioning) return;
      
      switch (event.key) {
        case 'ArrowLeft':
          event.preventDefault();
          handlePrevious();
          break;
        case 'ArrowRight':
          event.preventDefault();
          handleNext();
          break;
        case 'Escape':
          event.preventDefault();
          handleClose();
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

    if (open) {
      window.addEventListener('keydown', handleKeyPress);
    }

    return () => {
      window.removeEventListener('keydown', handleKeyPress);
    };
  }, [open, images.length, isTransitioning]);

  // Handle close
  const handleClose = () => {
    setIsFullscreen(false);
    onClose();
  };

  // Manual refresh function
  const handleRefresh = useCallback(() => {
    fetchImages();
  }, [fetchImages]);

  // Navigate left with transition
  const handlePrevious = useCallback(() => {
    if (images.length > 0 && !isTransitioning) {
      setIsTransitioning(true);
      setTimeout(() => {
        setCurrentIndex((prev) => (prev > 0 ? prev - 1 : images.length - 1));
        setTimeout(() => setIsTransitioning(false), 100);
      }, 200);
    }
  }, [images.length, isTransitioning]);

  // Navigate right with transition
  const handleNext = useCallback(() => {
    if (images.length > 0 && !isTransitioning) {
      setIsTransitioning(true);
      setTimeout(() => {
        setCurrentIndex((prev) => (prev < images.length - 1 ? prev + 1 : 0));
        setTimeout(() => setIsTransitioning(false), 100);
      }, 200);
    }
  }, [images.length, isTransitioning]);

  // Handle delete image
  const handleDeleteImage = useCallback(async (imageToDelete) => {
    if (!imageToDelete || !targetLabel) return;

    try {
      setDeleting(true);
      const success = await cameraService.deleteTempImage(targetLabel, imageToDelete.image_name);
      
      if (success) {
        setImages(prevImages => {
          const newImages = prevImages.filter(img => img.image_name !== imageToDelete.image_name);
          
          setCurrentIndex(prevIndex => {
            if (newImages.length === 0) return 0;
            if (prevIndex >= newImages.length) return newImages.length - 1;
            return prevIndex;
          });
          
          if (onImageCountChange) {
            onImageCountChange(newImages.length);
          }
          
          return newImages;
        });
        
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
  }, [targetLabel, onImageCountChange]);

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

  // Get visible images for horizontal three-image display
  const getVisibleImages = () => {
    if (images.length === 0) return [];
    
    if (images.length === 1) {
      return [{ ...images[0], position: 'center', index: 0 }];
    }
    
    if (images.length === 2) {
      const otherIndex = currentIndex === 0 ? 1 : 0;
      return [
        { ...images[otherIndex], position: 'side', index: otherIndex },
        { ...images[currentIndex], position: 'center', index: currentIndex }
      ];
    }
    
    // For 3 or more images
    const prevIndex = currentIndex > 0 ? currentIndex - 1 : images.length - 1;
    const nextIndex = currentIndex < images.length - 1 ? currentIndex + 1 : 0;
    
    return [
      { ...images[prevIndex], position: 'left', index: prevIndex },
      { ...images[currentIndex], position: 'center', index: currentIndex },
      { ...images[nextIndex], position: 'right', index: nextIndex }
    ];
  };

  const visibleImages = getVisibleImages();

  return (
    <Modal
      open={open}
      onClose={handleClose}
      closeAfterTransition
      BackdropComponent={Backdrop}
      BackdropProps={{
        timeout: 500,
        sx: { backgroundColor: 'rgba(0, 0, 0, 0.9)' }
      }}
    >
      <Fade in={open}>
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
              {targetLabel} Images
              {images.length > 0 && (
                <Chip
                  label={`${currentIndex + 1} of ${images.length}`}
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
              <Tooltip title="Refresh Images">
                <IconButton 
                  onClick={handleRefresh} 
                  disabled={loading}
                  sx={{
                    color: 'white',
                    backgroundColor: 'rgba(255,255,255,0.1)',
                    '&:hover': { backgroundColor: 'rgba(255,255,255,0.2)' }
                  }}
                >
                  <Refresh />
                </IconButton>
              </Tooltip>
              
              <Tooltip title={isFullscreen ? 'Exit Fullscreen' : 'Fullscreen'}>
                <IconButton 
                  onClick={() => setIsFullscreen(!isFullscreen)}
                  sx={{
                    color: 'white',
                    backgroundColor: 'rgba(255,255,255,0.1)',
                    '&:hover': { backgroundColor: 'rgba(255,255,255,0.2)' }
                  }}
                >
                  {isFullscreen ? <FullscreenExit /> : <Fullscreen />}
                </IconButton>
              </Tooltip>
              
              <Tooltip title="Close (Esc)">
                <IconButton 
                  onClick={handleClose}
                  sx={{
                    color: 'white',
                    backgroundColor: 'rgba(255,255,255,0.1)',
                    '&:hover': { backgroundColor: 'rgba(255,255,255,0.2)' }
                  }}
                >
                  <Close />
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
            {/* Loading State */}
            {loading && (
              <Box sx={{ 
                display: 'flex', 
                alignItems: 'center', 
                justifyContent: 'center',
                flexDirection: 'column',
                gap: 2,
                color: 'white'
              }}>
                <CircularProgress sx={{ color: '#667eea' }} size={48} />
                <Typography variant="h6">Loading images...</Typography>
              </Box>
            )}

            {/* Error State */}
            {error && !loading && (
              <Box sx={{ 
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                color: 'white',
                textAlign: 'center'
              }}>
                <Photo sx={{ fontSize: 64, opacity: 0.6, mb: 2 }} />
                <Typography variant="h6" sx={{ mb: 2 }}>
                  {error}
                </Typography>
                <Button 
                  onClick={handleRefresh}
                  variant="contained"
                  sx={{ backgroundColor: '#667eea' }}
                  startIcon={<Refresh />}
                >
                  Try Again
                </Button>
              </Box>
            )}

            {/* No Images State */}
            {!loading && !error && images.length === 0 && (
              <Box sx={{ 
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                color: 'white',
                textAlign: 'center'
              }}>
                <Photo sx={{ fontSize: 80, opacity: 0.4, mb: 3 }} />
                <Typography variant="h5" sx={{ mb: 1 }}>
                  No Images Found
                </Typography>
                <Typography variant="body1" sx={{ opacity: 0.7 }}>
                  {targetLabel ? `No images captured for "${targetLabel}"` : 'No target label specified'}
                </Typography>
              </Box>
            )}

            {/* Three-Image Horizontal Display */}
            {!loading && !error && images.length > 0 && (
              <>
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
                    width: isFullscreen ? '90vw' : '80vw',
                    height: isFullscreen ? '75vh' : '65vh',
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

                    // Position and style calculations
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
                      width = isFullscreen ? '1080px' : '900px';  // Your requested size
                      height = isFullscreen ? '576px' : '480px';  // Your requested size
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
                        key={`${image.index}-${image.position}`}
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
                          border: image.isTemporary 
                            ? '3px solid #667eea' 
                            : '3px solid rgba(255,255,255,0.2)',
                          // Remove any default padding from the Card
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
                              setCurrentIndex(image.index);
                            }
                          }
                        }}
                      >
                        <CardMedia
                          component="img"
                          image={imageSource(image)}
                          alt={`Image ${image.index + 1}`}
                          sx={{
                            width: '100%',
                            height: '100%',
                            objectFit: 'cover',
                            display: 'block',
                            // Ensure no padding/margin/border on the image itself
                            padding: 0,
                            margin: 0,
                            border: 'none',
                            // Ensure the image fills the entire card
                            position: 'absolute',
                            top: 0,
                            left: 0,
                            right: 0,
                            bottom: 0
                          }}
                          onError={(e) => {
                            console.error('Image failed to load:', imageSource(image));
                            e.target.src = 'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNDAwIiBoZWlnaHQ9IjMwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMTAwJSIgaGVpZ2h0PSIxMDAlIiBmaWxsPSIjMzMzIi8+PHRleHQgeD0iNTAlIiB5PSI1MCUiIGZvbnQtZmFtaWx5PSJBcmlhbCwgc2Fucy1zZXJpZiIgZm9udC1zaXplPSIxOCIgZmlsbD0iIzk5OSIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZHk9Ii4zZW0iPkltYWdlIG5vdCBmb3VuZDwvdGV4dD48L3N2Zz4=';
                          }}
                        />
                        
                        {/* Delete Button - Only show on center image and for temporary images */}
                        {image.isTemporary && isCenter && (
                          <Tooltip title="Delete Image">
                            <IconButton
                              onClick={() => openDeleteDialog(image)}
                              sx={{
                                position: 'absolute',
                                top: 16,
                                right: 16,
                                backgroundColor: 'rgba(244, 63, 94, 0.9)',
                                color: 'white',
                                zIndex: 1002,
                                '&:hover': {
                                  backgroundColor: 'rgba(244, 63, 94, 1)',
                                  transform: 'scale(1.1)'
                                },
                                transition: 'all 0.2s ease'
                              }}
                            >
                              <Delete />
                            </IconButton>
                          </Tooltip>
                        )}

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
                          {image.index + 1}
                        </Box>
                      </Card>
                    );
                  })}
                </Box>
              </>
            )}
          </Box>

          {/* Footer with Progress Indicators */}
          {!loading && !error && images.length > 1 && (
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
                    width: index === currentIndex ? 12 : 8,
                    height: index === currentIndex ? 12 : 8,
                    borderRadius: '50%',
                    backgroundColor: index === currentIndex ? '#667eea' : 'rgba(255,255,255,0.4)',
                    transition: 'all 0.3s ease',
                    cursor: 'pointer'
                  }}
                  onClick={() => {
                    if (!isTransitioning) {
                      setIsTransitioning(true);
                      setTimeout(() => {
                        setCurrentIndex(index);
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
              ← → Navigate • F Fullscreen • Esc Close
            </Typography>
          </Box>

          {/* Delete Confirmation Dialog */}
          <Dialog
            open={deleteDialogOpen}
            onClose={closeDeleteDialog}
            maxWidth="sm"
            fullWidth
            sx={{
              '& .MuiDialog-paper': {
                backgroundColor: '#1a1a1a',
                color: 'white'
              }
            }}
          >
            <DialogTitle sx={{ color: 'white', pb: 1 }}>
              Delete Image
            </DialogTitle>
            <DialogContent>
              <Typography variant="body1" sx={{ mb: 2, color: 'rgba(255,255,255,0.9)' }}>
                Are you sure you want to delete this image? This action cannot be undone.
              </Typography>
              {imageToDelete && (
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                  <Photo sx={{ color: '#667eea' }} />
                  <Typography variant="body2" sx={{ color: 'rgba(255,255,255,0.7)' }}>
                    {imageToDelete.image_name}
                  </Typography>
                </Box>
              )}
            </DialogContent>
            <DialogActions sx={{ px: 3, pb: 2 }}>
              <Button 
                onClick={closeDeleteDialog} 
                disabled={deleting}
                sx={{ color: 'rgba(255,255,255,0.7)' }}
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
                  '&:disabled': { backgroundColor: 'rgba(244, 63, 94, 0.5)' }
                }}
                startIcon={deleting ? <CircularProgress size={16} color="inherit" /> : <Delete />}
              >
                {deleting ? 'Deleting...' : 'Delete'}
              </Button>
            </DialogActions>
          </Dialog>
        </Box>
      </Fade>
    </Modal>
  );
};

export default HorizontalImageSlider;