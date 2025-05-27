import React, { useState, useEffect, useRef } from "react";
import { 
  Stack,
  Box, 
  styled, 
  Button, 
  TextField, 
  Dialog, 
  DialogActions, 
  DialogContent, 
  DialogTitle, 
  Typography,
  Select, 
  MenuItem,
  Card,
  CardContent,
  Grid,
  IconButton,
  CardMedia,
  Tooltip
} from "@mui/material";
import { 
  Videocam, 
  VideocamOff, 
  KeyboardArrowUp, 
  KeyboardArrowDown, 
  Photo,
  Refresh 
} from "@mui/icons-material";

import api from "../../utils/UseAxios"

const Container = styled("div")(({ theme }) => ({
  margin: "30px",
  [theme.breakpoints.down("sm")]: { margin: "16px" },
  "& .breadcrumb": {
    marginBottom: "30px",
    [theme.breakpoints.down("sm")]: { marginBottom: "16px" },
  },
}));

const VideoCard = styled(Card)(({ theme }) => ({
  width: "100%",
  height: "480px",
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  backgroundColor: "#f5f5f5",
  border: "2px dashed #ccc",
  borderRadius: "12px",
  position: "relative",
  overflow: "hidden",
  [theme.breakpoints.down("sm")]: {
    height: "300px",
  },
}));

const ImageSliderCard = styled(Card)(({ theme }) => ({
  height: "480px",
  display: "flex",
  flexDirection: "column",
  backgroundColor: "#f5f5f5",
  border: "1px solid #ddd",
  borderRadius: "12px",
  [theme.breakpoints.down("sm")]: {
    height: "300px",
  },
}));

const PlaceholderContent = styled(Box)({
  display: "flex",
  flexDirection: "column",
  alignItems: "center",
  justifyContent: "center",
  textAlign: "center",
  color: "#666",
});

const VideoImage = styled("img")({
  width: "100%",
  height: "100%",
  objectFit: "cover",
});

// API functions
const startCamera = async (cameraId) => {
  try {
    const numericCameraId = parseInt(cameraId);
    if (isNaN(numericCameraId)) {
      throw new Error("Invalid camera ID: must be a number");
    }

    const response = await api.post("/api/artifact_keeper/camera/start", { 
      camera_id: numericCameraId
    });
    
    console.log(response.data.message || "Camera started successfully");
    return true;
  } catch (error) {
    console.error("Error starting camera:", error.response?.data?.detail || error.message);
    return false;
  }
};

const stopCamera = async () => {
  try {
    await api.post("/api/artifact_keeper/camera/cleanup-temp-photos");
    const response = await api.post("/api/artifact_keeper/camera/stop");
    
    console.log("Camera stopped and temporary photos cleaned up.");
    window.location.reload();
  } catch (error) {
    console.error("Error stopping camera:", error.response?.data?.detail || error.message);
  }
};

const captureImages = async (pieceLabel) => {
  try {
    const response = await api.get(`/api/artifact_keeper/camera/capture_images/${pieceLabel}`, {
      responseType: 'blob'
    });
    
    if (response.data instanceof Blob) {
      const imageUrl = URL.createObjectURL(response.data);
      return imageUrl;
    } else {
      console.error("Response is not a blob:", response.data);
      return null;
    }
  } catch (error) {
    console.error("Error capturing images:", error.response?.data?.detail || error.message);
    return null;
  }
};

const getTempImages = async (pieceLabel) => {
  try {
    const response = await api.get(`/api/artifact_keeper/camera/temp-photos/${pieceLabel}`);
    return response.data || [];
  } catch (error) {
    console.error("Error fetching temp images:", error.response?.data?.detail || error.message);
    return [];
  }
};

const saveImagesToDatabase = async (pieceLabel) => {
  try {
    const response = await api.post("/api/artifact_keeper/camera/save-images", {
      piece_label: pieceLabel
    });

    console.log(response.data.message || "Images saved successfully");
    await stopCamera();
    window.location.reload();
  } catch (error) {
    const errorMessage = error.response?.data?.detail || "Error saving images";
    console.error(errorMessage);
  }
};

// Image Slider Component
const ImageSlider = ({ capturedImages, targetLabel }) => {
  const [currentIndex, setCurrentIndex] = useState(0);

  const handlePrevious = () => {
    if (capturedImages.length > 0) {
      setCurrentIndex((prev) => (prev > 0 ? prev - 1 : capturedImages.length - 1));
    }
  };

  const handleNext = () => {
    if (capturedImages.length > 0) {
      setCurrentIndex((prev) => (prev < capturedImages.length - 1 ? prev + 1 : 0));
    }
  };

  const getVisibleImages = () => {
    if (capturedImages.length === 0) return [];
    
    const visibleImages = [];
    const totalImages = capturedImages.length;
    
    if (totalImages === 1) {
      return [{ ...capturedImages[0], position: 'middle', index: 0 }];
    }
    
    // Previous image (top)
    const prevIndex = currentIndex > 0 ? currentIndex - 1 : totalImages - 1;
    visibleImages.push({ ...capturedImages[prevIndex], position: 'top', index: prevIndex });
    
    // Current image (middle)
    visibleImages.push({ ...capturedImages[currentIndex], position: 'middle', index: currentIndex });
    
    // Next image (bottom)
    const nextIndex = currentIndex < totalImages - 1 ? currentIndex + 1 : 0;
    visibleImages.push({ ...capturedImages[nextIndex], position: 'bottom', index: nextIndex });
    
    return visibleImages;
  };

  const visibleImages = getVisibleImages();

  return (
    <ImageSliderCard>
      <Box sx={{ p: 2, borderBottom: 1, borderColor: 'divider' }}>
        <Typography variant="h6" sx={{ color: '#666', textAlign: 'center' }}>
          Captured Images
        </Typography>
      </Box>

      <Box sx={{ flex: 1, display: 'flex', flexDirection: 'column', p: 2 }}>
        {capturedImages.length === 0 ? (
          <Box 
            sx={{ 
              flex: 1, 
              display: 'flex', 
              flexDirection: 'column', 
              alignItems: 'center', 
              justifyContent: 'center',
              textAlign: 'center',
              color: 'text.secondary'
            }}
          >
            <Photo sx={{ fontSize: 48, opacity: 0.5, mb: 1 }} />
            <Typography variant="body2">
              {targetLabel ? `No images captured yet for "${targetLabel}"` : 'Enter a piece label and start capturing'}
            </Typography>
          </Box>
        ) : (
          <>
            {/* Navigation and Images */}
            <Box sx={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center' }}>
              {/* Up Arrow */}
              {capturedImages.length > 1 && (
                <IconButton
                  onClick={handlePrevious}
                  sx={{
                    mb: 1,
                    backgroundColor: 'primary.main',
                    color: 'white',
                    '&:hover': { backgroundColor: 'primary.dark' },
                    boxShadow: 2
                  }}
                >
                  <KeyboardArrowUp />
                </IconButton>
              )}

              {/* Images Container */}
              <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 1, flex: 1, justifyContent: 'center' }}>
                {visibleImages.map((image, index) => (
                  <Card
                    key={`${image.index}-${image.position}`}
                    sx={{
                      width: image.position === 'middle' ? '200px' : '160px',
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
                      height={image.position === 'middle' ? '150' : '120'}
                      image={image.url}
                      alt={`Captured image ${image.index + 1}`}
                      sx={{ objectFit: 'cover' }}
                    />
                    <Box sx={{ position: 'absolute', top: 4, left: 4, backgroundColor: 'rgba(0,0,0,0.7)', color: 'white', px: 1, py: 0.5, borderRadius: 1, fontSize: '0.75rem' }}>
                      {image.index + 1}
                    </Box>
                  </Card>
                ))}
              </Box>

              {/* Down Arrow */}
              {capturedImages.length > 1 && (
                <IconButton
                  onClick={handleNext}
                  sx={{
                    mt: 1,
                    backgroundColor: 'primary.main',
                    color: 'white',
                    '&:hover': { backgroundColor: 'primary.dark' },
                    boxShadow: 2
                  }}
                >
                  <KeyboardArrowDown />
                </IconButton>
              )}
            </Box>

            {/* Image Counter */}
            <Typography
              variant="caption"
              sx={{
                mt: 1,
                px: 2,
                py: 0.5,
                backgroundColor: 'rgba(0, 0, 0, 0.7)',
                color: 'white',
                borderRadius: 1,
                textAlign: 'center',
                alignSelf: 'center'
              }}
            >
              {currentIndex + 1} of {capturedImages.length}
            </Typography>
          </>
        )}
      </Box>
    </ImageSliderCard>
  );
};

// Main Video Feed Component
const VideoFeed = ({ isCameraStarted, onStartCamera, onStopCamera, onCaptureImages, cameraId, targetLabel, capturedImages, onImageCaptured }) => {
  const [videoUrl, setVideoUrl] = useState("");
  const [dialogOpen, setDialogOpen] = useState(false);
  const [snapshotEffect, setSnapshotEffect] = useState(false);
  const [isCapturing, setIsCapturing] = useState(false);
  const requiredCaptures = 10;
  const videoRef = useRef(null);

  useEffect(() => {
    if (isCameraStarted) {
      setVideoUrl("/api/artifact_keeper/camera/video_feed");
    } else {
      setVideoUrl("");
    }
  }, [isCameraStarted]);

  const handleCaptureImages = async () => {
    if (isCapturing) return;
    
    if (!targetLabel || targetLabel.trim() === "") {
      alert("Please enter a piece label before capturing images.");
      return;
    }

    setIsCapturing(true);
    setSnapshotEffect(true);
    
    try {
      const imageUrl = await onCaptureImages(targetLabel);
      if (imageUrl) {
        // Notify parent component about the new image
        onImageCaptured(imageUrl);
        
        if (capturedImages.length + 1 >= requiredCaptures) {
          setDialogOpen(true);
        }
      } else {
        alert("Failed to capture image. Please try again.");
      }
    } catch (error) {
      console.error("Error during image capture:", error);
      alert("Error capturing image. Please try again.");
    } finally {
      setIsCapturing(false);
      setTimeout(() => setSnapshotEffect(false), 1000);
    }
  };

  const handleSaveImages = async () => {
    if (!targetLabel) {
      alert("Piece label is required.");
      return;
    }
    await saveImagesToDatabase(targetLabel);
    setDialogOpen(false);
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", width: "100%" }}>
      <VideoCard>
        {isCameraStarted ? (
          <>
            <VideoImage
              ref={videoRef}
              src={videoUrl}
              alt="Video Feed"
            />
            {snapshotEffect && (
              <div
                style={{
                  position: "absolute",
                  top: 0,
                  left: 0,
                  width: "100%",
                  height: "100%",
                  backgroundColor: "rgba(0, 0, 0, 0.5)",
                  display: "flex",
                  justifyContent: "center",
                  alignItems: "center",
                  color: "white",
                  fontSize: "2rem",
                  fontWeight: "bold",
                }}
              >
                {isCapturing ? "Capturing..." : "Snapshot"}
              </div>
            )}
          </>
        ) : (
          <PlaceholderContent>
            <VideocamOff 
              sx={{ 
                fontSize: 80, 
                color: "#bbb", 
                marginBottom: 2 
              }} 
            />
            <Typography 
              variant="h5" 
              sx={{ 
                color: "#888", 
                marginBottom: 1,
                fontWeight: 500 
              }}
            >
              Camera Not Started
            </Typography>
            <Typography 
              variant="body1" 
              sx={{ 
                color: "#aaa",
                maxWidth: "300px",
                lineHeight: 1.5
              }}
            >
              Select a camera and click "Start Camera" to begin video feed
            </Typography>
          </PlaceholderContent>
        )}
      </VideoCard>

      <div style={{ marginTop: "20px" }}>
        {isCameraStarted ? (
          <>
            <Button 
              variant="contained" 
              onClick={handleCaptureImages} 
              style={{ margin: "0 10px" }}
              disabled={isCapturing || capturedImages.length >= requiredCaptures}
              startIcon={<Videocam />}
            >
              {isCapturing ? "Capturing..." : `Capture Images (${capturedImages.length}/${requiredCaptures})`}
            </Button>
            <Button 
              variant="outlined" 
              color="error"
              onClick={onStopCamera} 
              style={{ margin: "0 10px" }}
            >
              Stop Camera
            </Button>
          </>
        ) : (
          <Button
            variant="contained"
            onClick={() => onStartCamera(cameraId)}
            style={{ margin: "10px" }}
            disabled={!cameraId}
            startIcon={<Videocam />}
            size="large"
          >
            Start Camera
          </Button>
        )}
      </div>

      <Dialog
        open={dialogOpen}
        onClose={() => setDialogOpen(false)}
        aria-labelledby="alert-dialog-title"
        aria-describedby="alert-dialog-description"
      >
        <DialogTitle id="alert-dialog-title">{"Capture Complete"}</DialogTitle>
        <DialogContent>
          <Typography variant="body1">
            You have captured {capturedImages.length} images. Please save them to the database.
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={handleSaveImages} color="primary" variant="contained">
            Save
          </Button>
          <Button onClick={() => setDialogOpen(false)} color="secondary">
            Close
          </Button>
        </DialogActions>
      </Dialog>
    </div>
  );
};

// Main Component
export default function AppPartLibrary() {
  const [targetLabel, setTargetLabel] = useState("");
  const [cameraId, setCameraId] = useState("");
  const [cameras, setCameras] = useState([]);
  const [isCameraStarted, setCameraStarted] = useState(false);
  const [selectedCameraId, setSelectedCameraId] = useState('');
  const [capturedImages, setCapturedImages] = useState([]);
  
  useEffect(() => {
    const handleBeforeUnload = async () => {
      try {
        await api.post("/api/artifact_keeper/camera/stop");
        await api.post("/api/artifact_keeper/camera/cleanup-temp-photos");
      } catch (error) {
        console.error("Error during cleanup:", error);
      }
    };
  
    window.addEventListener("beforeunload", handleBeforeUnload);
  
    return () => {
      window.removeEventListener("beforeunload", handleBeforeUnload);
      handleBeforeUnload();
    };
  }, []);
  
  useEffect(() => {
    api.get('/api/artifact_keeper/camera/get_allcameras')
      .then(response => {
        console.log("Cameras received:", response.data);
        setCameras(response.data);
      })
      .catch(error => {
        console.error('There were an error fetching the camera data!', error);
      });
  }, []);

  // Clear captured images when target label changes
  useEffect(() => {
    setCapturedImages([]);
  }, [targetLabel]);

  const handleCameraChange = (event) => {
    const selectedCameraId = event.target.value;
    console.log("Selected camera ID:", selectedCameraId, "Type:", typeof selectedCameraId);
    setSelectedCameraId(selectedCameraId);
    setCameraId(selectedCameraId);
  };
  
  const handleStartCamera = async (cameraId) => {
    console.log("Starting camera with ID:", cameraId);
    if (cameraId && cameraId !== '') {
      const success = await startCamera(cameraId);
      setCameraStarted(success);
    } else {
      alert("Please select a camera first.");
    }
  };
  
  const handleStopCamera = async () => {
    await api.post("/api/artifact_keeper/camera/cleanup-temp-photos");
    await stopCamera();
    setCameraStarted(false);
    setCapturedImages([]); // Clear captured images when stopping camera
  };

  const handleTargetLabelChange = (event) => {
    setTargetLabel(event.target.value);
  };

  const handleImageCaptured = (imageUrl) => {
    setCapturedImages(prev => [...prev, { url: imageUrl, timestamp: new Date() }]);
  };

  // Clean up object URLs when component unmounts or images change
  useEffect(() => {
    return () => {
      capturedImages.forEach(image => {
        if (image.url && image.url.startsWith('blob:')) {
          URL.revokeObjectURL(image.url);
        }
      });
    };
  }, [capturedImages]);

  return (
    <Container>
      <Stack spacing={3}>
        <Box 
          className="controls"
          sx={{
            display: "flex",
            gap: 2,
            flexWrap: "wrap",
            alignItems: "center",
            justifyContent: "center"
          }}
        >
          <TextField 
            label="Target Label" 
            value={targetLabel} 
            onChange={handleTargetLabelChange}
            placeholder="e.g., G123.12345.123.12"
            required
            sx={{ minWidth: 250 }}
          />
          <Select
            labelId="camera-select-label"
            value={selectedCameraId}
            onChange={handleCameraChange}
            displayEmpty
            sx={{ minWidth: 200 }}
          >
            <MenuItem value="" disabled>Select a Camera</MenuItem>
            {cameras.map((camera) => (
              <MenuItem key={camera.id} value={camera.id}>
                {camera.model}
              </MenuItem>
            ))}
          </Select>
        </Box>

        {/* Main Content - Video Feed and Image Slider */}
        <Grid container spacing={3}>
          <Grid item xs={12} md={8}>
            <VideoFeed
              isCameraStarted={isCameraStarted}
              onStartCamera={handleStartCamera}
              onStopCamera={handleStopCamera}
              onCaptureImages={captureImages}
              cameraId={cameraId}
              targetLabel={targetLabel}
              capturedImages={capturedImages}
              onImageCaptured={handleImageCaptured}
            />
          </Grid>
          <Grid item xs={12} md={4}>
            <ImageSlider 
              capturedImages={capturedImages}
              targetLabel={targetLabel}
            />
          </Grid>
        </Grid>
      </Stack>
    </Container>
  );
}