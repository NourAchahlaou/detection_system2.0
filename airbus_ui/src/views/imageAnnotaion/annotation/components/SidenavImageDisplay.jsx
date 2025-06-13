import { useEffect, useState, useCallback, useRef } from "react";
import { 
  Box, 
  Typography, 
  styled, 
  Fade, 
  CircularProgress,
  IconButton,
  Card,
  CardMedia,
  Chip
} from "@mui/material";
import { 
  Photo, 
  KeyboardArrowUp, 
  KeyboardArrowDown, 
} from "@mui/icons-material";
import api from "../../../../utils/UseAxios";

// Updated styled components to match capture theme exactly
const MaxCustomaizer = styled("div")(({ theme }) => ({
  width: "100%",
  height: "100%",
  display: "flex",
  flexDirection: "column",
  overflow: "hidden",
  backgroundColor: "transparent", // Match the capture component background
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
  annotatedImages,
  onImageCountUpdate
}) {
  const [images, setImages] = useState([]);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [loading, setLoading] = useState(false);
  const [selectedImageUrl, setSelectedImageUrl] = useState('');
  
  const mountedRef = useRef(true);

  const fetchImages = useCallback(async () => {
    if (!pieceLabel) {
      setImages([]);
      setCurrentIndex(0);
      if (onImageCountUpdate) {
        onImageCountUpdate(0);
      }
      return;
    }

    try {
      setLoading(true);
      const response = await api.get(`/api/annotation/annotations/get_images_of_piece/${pieceLabel}`);
      const data = response.data;
      
      if (mountedRef.current) {
        setImages(data);
        setCurrentIndex(0);

        if (onImageCountUpdate) {
          onImageCountUpdate(data.length);
        }

        if (data.length > 0 && onFirstImageLoad) {
          const firstImage = data[0];
          onFirstImageLoad(firstImage.url, firstImage.name); // Pass both URL and ID
          setSelectedImageUrl(firstImage.url);
        }
      }
  
    } catch (error) {
      console.error("Error fetching images:", error.response?.data?.detail || error.message);
      if (mountedRef.current) {
        setImages([]);
        setCurrentIndex(0);
        if (onImageCountUpdate) {
          onImageCountUpdate(0);
        }
      }
    } finally {
      if (mountedRef.current) {
        setLoading(false);
      }
    }
  }, [pieceLabel, onFirstImageLoad, onImageCountUpdate]);

  useEffect(() => {
    mountedRef.current = true;
    fetchImages();
    
    return () => {
      mountedRef.current = false;
    };
  }, [pieceLabel, fetchImages]);

  const handleImageClick = (imageUrl, imageId, index) => {
    setSelectedImageUrl(imageUrl);
    setCurrentIndex(index);
    onImageSelect(imageUrl, imageId); // Pass both URL and ID
  };
  // Navigate up in the slider
  const handlePrevious = useCallback(() => {
    if (images.length > 0) {
      const newIndex = currentIndex > 0 ? currentIndex - 1 : images.length - 1;
      setCurrentIndex(newIndex);
      const newImage = images[newIndex];
      const newImageUrl = newImage.url;
      const newImageId = newImage.name; // Use 'name' as the imageId
      setSelectedImageUrl(newImageUrl);
      onImageSelect(newImageUrl, newImageId); // Pass both URL and ID
    }
  }, [images, currentIndex, onImageSelect]);

  // Navigate down in the slider
  const handleNext = useCallback(() => {
    if (images.length > 0) {
      const newIndex = currentIndex < images.length - 1 ? currentIndex + 1 : 0;
      setCurrentIndex(newIndex);
      const newImage = images[newIndex];
      const newImageUrl = newImage.url;
      const newImageId = newImage.name; // Use 'name' as the imageId
      setSelectedImageUrl(newImageUrl);
      onImageSelect(newImageUrl, newImageId); // Pass both URL and ID
    }
  }, [images, currentIndex, onImageSelect]);


  // Get visible images for the stack effect - matching capture component proportions
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
    <MaxCustomaizer>
      {/* Content Area with ImageSlider styling */}
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
        {/* Loading State */}
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

        {/* No Images State */}
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

        {/* Image Slider - matching capture component exact proportions */}
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
            {/* Navigation Arrow - Up */}
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

            {/* Images Stack Container - matching capture component dimensions exactly */}
            <Box
              sx={{
                position: 'relative',
                width: '300px', // Match capture component width
                height: '420px', // Match capture component height
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
                const isAnnotated = annotatedImages.includes(image.url);

                // Positioning calculations - matching capture component exactly
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
                  translateY = -120; // Match capture component spacing
                  translateX = 0;
                  zIndex = 200;
                  scale = 0.88;
                  opacity = 0.5;
                  rotation = 0;
                } else if (isBackBottom) {
                  translateY = 120; // Match capture component spacing
                  translateX = 0;
                  zIndex = 100;
                  scale = 0.88;
                  opacity = 0.5;
                  rotation = 0;
                } else if (isBack) {
                  translateY = 80; // Match capture component spacing
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
                      width: '280px', // Match capture component card width
                      height: '200px', // Match capture component card height
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
                      if (images.length > 1 && !isCenter) {
                        if (isBackTop) handlePrevious();
                        else if (isBackBottom) handleNext();
                        else if (isBack) {
                          handleImageClick(image.url, image.name, image.index); // Pass image.name as imageId
                        }
                      }
                    }}
                  >
                    <CardMedia
                      component="img"
                      image={image.url}
                      alt={`Image ${image.index + 1}`}
                      sx={{
                        objectFit: 'cover',
                        width: '100%',
                        height: '100%',
                        display: 'block',
                        margin: 0,
                        padding: 0,
                        border: 'none',
                        outline: 'none',
                        verticalAlign: 'top',
                        position: 'absolute',
                        top: 0,
                        left: 0,
                        right: 0,
                        bottom: 0
                      }}
                      onError={(e) => {
                        e.target.src = 'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjgwIiBoZWlnaHQ9IjIwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMTAwJSIgaGVpZ2h0PSIxMDAlIiBmaWxsPSIjZjVmNWY1Ii8+PHRleHQgeD0iNTAlIiB5PSI1MCUiIGZvbnQtZmFtaWx5PSJBcmlhbCwgc2Fucy1zZXJpZiIgZm9udC1zaXplPSIxNCIgZmlsbD0iIzk5OSIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZHk9Ii4zZW0iPkltYWdlIG5vdCBmb3VuZDwvdGV4dD48L3N2Zz4=';
                      }}
                    />
                    
                    {/* Image Number Badge - matching capture component exactly */}
                    <Box
                      sx={{
                        position: 'absolute',
                        top: 12, // Match capture component positioning
                        right: 12, // Match capture component positioning
                        backgroundColor: isAnnotated ? "#4caf50" : isCenter ? 'rgba(0,0,0,0.8)' : 'rgba(0,0,0,0.7)',
                        color: 'white',
                        borderRadius: '50%',
                        width: 32, // Match capture component size
                        height: 32, // Match capture component size
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        fontSize: '0.8rem', // Match capture component font size
                        fontWeight: 'bold',
                        boxShadow: '0 2px 8px rgba(0,0,0,0.3)',
                        zIndex: 399
                      }}
                    >
                      {image.index + 1}
                    </Box>

                    {/* Annotated Badge */}
                    {isAnnotated && (
                      <Box
                        sx={{
                          position: 'absolute',
                          bottom: 12, // Match capture component positioning
                          left: 12, // Match capture component positioning
                          backgroundColor: '#4caf50',
                          color: 'white',
                          borderRadius: '12px',
                          padding: '4px 12px', // Slightly larger padding
                          fontSize: '0.65rem',
                          fontWeight: '600',
                          textTransform: 'uppercase',
                          letterSpacing: '0.5px',
                          boxShadow: '0 2px 8px rgba(76, 175, 80, 0.4)',
                          zIndex: 399
                        }}
                      >
                        Done
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