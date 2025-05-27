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
  CardContent } from "@mui/material";
import { Videocam, VideocamOff } from "@mui/icons-material";

import api from "../../utils/UseAxios" // Import the API module instead of axios directly

const Container = styled("div")(({ theme }) => ({
  margin: "30px",
  [theme.breakpoints.down("sm")]: { margin: "16px" },
  "& .breadcrumb": {
    marginBottom: "30px",
    [theme.breakpoints.down("sm")]: { marginBottom: "16px" },
  },
}));

const VideoCard = styled(Card)(({ theme }) => ({
  width: "900px",
  height: "480px",
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  backgroundColor: "#f5f5f5",
  border: "2px dashed #ccc",
  borderRadius: "12px",
  position: "relative",
  overflow: "hidden",
  [theme.breakpoints.down("md")]: {
    width: "100%",
    maxWidth: "700px",
  },
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

const startCamera = async (cameraId) => {
  try {
    // Convert to integer and use correct field name
    const numericCameraId = parseInt(cameraId);
    if (isNaN(numericCameraId)) {
      throw new Error("Invalid camera ID: must be a number");
    }

    const response = await api.post("/api/artifact_keeper/camera/start", { 
      camera_id: numericCameraId  // ✅ Correct field name and type
    });
    
    console.log(response.data.message || "Camera started successfully");
    return true; // Indicate success
  } catch (error) {
    console.error("Error starting camera:", error.response?.data?.detail || error.message);
    return false; // Indicate failure
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
    // Fixed: Ensure we get the response as a blob
    const response = await api.get(`/api/artifact_keeper/camera/capture_images/${pieceLabel}`, {
      responseType: 'blob'
    });
    
    // Check if the response is actually a blob
    if (response.data instanceof Blob) {
      const imageUrl = URL.createObjectURL(response.data);
      return imageUrl; // Return the image URL
    } else {
      console.error("Response is not a blob:", response.data);
      return null;
    }
  } catch (error) {
    console.error("Error capturing images:", error.response?.data?.detail || error.message);
    return null;
  }
};

const saveImagesToDatabase = async (pieceLabel) => {
  try {
    const response = await api.post("/api/artifact_keeper/camera/save-images", {
      piece_label: pieceLabel
    });

    console.log(response.data.message || "Images saved successfully");

    // Stop the camera after saving images
    await stopCamera(); // Wait for the camera to stop

    // Reload the window after the camera is stopped
    window.location.reload();
  } catch (error) {
    const errorMessage = error.response?.data?.detail || "Error saving images";
    console.error(errorMessage);
  }
};

const VideoFeed = ({ isCameraStarted, onStartCamera, onStopCamera, onCaptureImages, cameraId, targetLabel }) => {
  const [videoUrl, setVideoUrl] = useState("");
  const [dialogOpen, setDialogOpen] = useState(false);
  const [capturedImages, setCapturedImages] = useState([]);
  const [capturedImagesCount, setCapturedImagesCount] = useState(0);
  const [snapshotEffect, setSnapshotEffect] = useState(false);
  const [isCapturing, setIsCapturing] = useState(false); // Add loading state
  const requiredCaptures = 10;
  const videoRef = useRef(null);

  useEffect(() => {
    if (isCameraStarted) {
      // Note: for video streaming, we might still need a direct URL
      // Since the API interceptor can't handle streaming data properly
      setVideoUrl("/api/artifact_keeper/camera/video_feed");
    } else {
      setVideoUrl("");
    }
  }, [isCameraStarted]);

  const handleCaptureImages = async () => {
    if (isCapturing) return; // Prevent multiple captures
    
    if (!targetLabel || targetLabel.trim() === "") {
      alert("Please enter a piece label before capturing images.");
      return;
    }

    setIsCapturing(true);
    setSnapshotEffect(true);
    
    try {
      const imageUrl = await onCaptureImages(targetLabel);
      if (imageUrl) {
        setCapturedImages((prevImages) => [...prevImages, imageUrl]);
        setCapturedImagesCount((prevCount) => prevCount + 1);
        if (capturedImagesCount + 1 >= requiredCaptures) {
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
      setTimeout(() => setSnapshotEffect(false), 1000); // Hide effect after 1 second
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

  // Clean up object URLs when component unmounts or images change
  useEffect(() => {
    return () => {
      capturedImages.forEach(imageUrl => {
        if (imageUrl.startsWith('blob:')) {
          URL.revokeObjectURL(imageUrl);
        }
      });
    };
  }, [capturedImages]);

  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center" }}>
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
            {/* Captured Images Stack */}
            <div
              style={{
                position: "absolute",
                bottom: "20px",
                right: "20px",
                width: "120px",
              }}
            >
              {capturedImages.map((imageUrl, index) => (
                <div
                  key={index}
                  style={{
                    position: "absolute",
                    bottom: index * 5 + "px",
                    right: index * 5 + "px",
                    transform: `rotate(${index % 2 === 0 ? "0deg" : "25deg"})`,
                    zIndex: index,
                  }}
                >
                  <img
                    src={imageUrl}
                    alt={`Captured ${index + 1}`}
                    style={{
                      width: "100px",
                      height: "auto",
                      border: "2px solid white",
                      borderRadius: "8px",
                      boxShadow: "0 4px 8px rgba(0, 0, 0, 0.2)",
                    }}
                  />
                  <div
                    style={{
                      position: "absolute",
                      top: "5px",
                      left: "5px",
                      fontSize: "12px",
                      color: "white",
                      backgroundColor: "rgba(0, 0, 0, 0.6)",
                      padding: "2px 5px",
                      borderRadius: "5px",
                      fontWeight: "bold",
                    }}
                  >
                    {index + 1}
                  </div>
                </div>
              ))}
            </div>
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
              disabled={isCapturing || capturedImagesCount >= requiredCaptures}
              startIcon={<Videocam />}
            >
              {isCapturing ? "Capturing..." : `Capture Images (${capturedImagesCount}/${requiredCaptures})`}
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
            You have captured {capturedImagesCount} images. Please save them to the database.
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

export default function AppPartLibrary() {
  const [targetLabel, setTargetLabel] = useState("");
  const [cameraId, setCameraId] = useState("");
  const [cameras, setCameras] = useState([]);
  const [isCameraStarted, setCameraStarted] = useState(false);
  const [selectedCameraId, setSelectedCameraId] = useState('');
  
  useEffect(() => {
    // Function to clean up temp photos and stop the camera
    const handleBeforeUnload = async () => {
      try {
        // Stop the camera
        await api.post("/api/artifact_keeper/camera/stop");
        
        // Cleanup temporary photos
        await api.post("/api/artifact_keeper/camera/cleanup-temp-photos");
      } catch (error) {
        console.error("Error during cleanup:", error);
      }
    };
  
    // Add event listener for page refresh/unload
    window.addEventListener("beforeunload", handleBeforeUnload);
  
    // Add cleanup function for component unmount or effect re-run
    return () => {
      window.removeEventListener("beforeunload", handleBeforeUnload);
  
      // Optionally stop the camera on component unmount
      handleBeforeUnload();
    };
  }, []);
  
  useEffect(() => {
    // Fetch the list of cameras from the backend using the API
    api.get('/api/artifact_keeper/camera/get_allcameras')
      .then(response => {
        console.log("Cameras received:", response.data); // Debug log
        setCameras(response.data);
      })
      .catch(error => {
        console.error('There were an error fetching the camera data!', error);
      });
  }, []);

  const handleCameraChange = (event) => {
    const selectedCameraId = event.target.value;
    console.log("Selected camera ID:", selectedCameraId, "Type:", typeof selectedCameraId); // Debug log
    setSelectedCameraId(selectedCameraId);
    setCameraId(selectedCameraId); // Update cameraId state when a camera is selected
  };
  
  const handleStartCamera = async (cameraId) => {
    console.log("Starting camera with ID:", cameraId); // Debug log
    if (cameraId && cameraId !== '') {
      const success = await startCamera(cameraId);
      setCameraStarted(success);
    } else {
      alert("Please select a camera first.");
    }
  };
  
  const handleStopCamera = async () => {
    await api.post("/api/artifact_keeper/camera/cleanup-temp-photos");
    await stopCamera(); // Stop camera functionality
    setCameraStarted(false);
  };

  const handleTargetLabelChange = (event) => {
    setTargetLabel(event.target.value);
  };

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
        <VideoFeed
          isCameraStarted={isCameraStarted}
          onStartCamera={handleStartCamera}
          onStopCamera={handleStopCamera}
          onCaptureImages={captureImages}
          cameraId={cameraId}
          targetLabel={targetLabel}
        />
      </Stack>
    </Container>
  );
}