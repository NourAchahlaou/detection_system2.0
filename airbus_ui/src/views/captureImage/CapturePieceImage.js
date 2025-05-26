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
  MenuItem } from "@mui/material";

import api from "../../utils/UseAxios" // Import the API module instead of axios directly

const Container = styled("div")(({ theme }) => ({
  margin: "30px",
  [theme.breakpoints.down("sm")]: { margin: "16px" },
  "& .breadcrumb": {
    marginBottom: "30px",
    [theme.breakpoints.down("sm")]: { marginBottom: "16px" },
  },
}));

const startCamera = async (cameraId) => {
  try {
    const response = await api.post(" /api/artifact_keeper/camera/start-camera", { 
      camera_id: cameraId 
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
    await api.post("/api/artifact_keeper/camera/cleanup_temp_photos");
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
    
    const imageUrl = URL.createObjectURL(response.data);
    return imageUrl; // Return the image URL
  } catch (error) {
    console.error("Error capturing images:", error.response?.data?.detail || error.message);
    return null;
  }
};

const saveImagesToDatabase = async (pieceLabel) => {
  try {
    const response = await api.post(" /api/artifact_keeper/camera/save-images", null, {
      params: { piece_label: pieceLabel }
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
    setSnapshotEffect(true);
    const imageUrl = await onCaptureImages(targetLabel);
    if (imageUrl) {
      setCapturedImages((prevImages) => [...prevImages, imageUrl]);
      setCapturedImagesCount((prevCount) => prevCount + 1);
      if (capturedImagesCount + 1 >= requiredCaptures) {
        setDialogOpen(true);
      }
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

  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center" }}>
      <div style={{ position: "relative" }}>
        {isCameraStarted ? (
          <>
            <img
              ref={videoRef}
              src={videoUrl}
              style={{ width: "900px", maxHeight: "480px", position: "relative" }}
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
                Snapshot
              </div>
            )}
  <div
    style={{
      position: "fixed", // Fixes the stack's position on the screen
      bottom: "50px", // Adjust distance from the bottom of the screen
      right: "50px", // Adjust distance from the right of the screen
      width: "120px", // Controls stack width
    }}
  >
    {capturedImages.map((imageUrl, index) => (
      <div
        key={index}
        style={{
          position: "absolute", // Ensures images stack on top of each other
          bottom: index * 5 + "px", // Offset each image slightly for stacking effect
          right: index * 5 + "px",
          transform: `rotate(${index % 2 === 0 ? "0deg" : "25deg"})`, // Alternate rotation
          zIndex: index, // Layer each image on top of the previous one
        }}
      >
        {/* Image element */}
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

        {/* Small number in the top-left corner */}
        <div
          style={{
            position: "absolute",
            top: "5px", // Fixed position within the image
            left: "5px", // Fixed position within the image
            fontSize: "12px", // Small font size for the count
            color: "white", // Text color
            backgroundColor: "rgba(0, 0, 0, 0.6)", // Background for visibility
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
          <p>No Video Feed</p>
        )}
      </div>

      <div style={{ marginTop: "20px" }}>
        {isCameraStarted ? (
          <>
            <Button variant="contained" onClick={handleCaptureImages} style={{ margin: "0 10px" }}>
              Capture Images
            </Button>
            <Button variant="contained" onClick={onStopCamera} style={{ margin: "0 10px" }}>
              Cancel
            </Button>
          </>
        ) : (
          <Button
            variant="contained"
            onClick={() => onStartCamera(cameraId)}
            style={{ margin: "10px" }}
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
          <Button onClick={handleSaveImages} color="primary">
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
        await api.post("/api/camera/stop");
        
        // Cleanup temporary photos
        await api.post(" /api/artifact_keeper/camera/cleanup-temp-photos");
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
    api.get('/api/artifact_keeper/camera/get_allcameras/')
      .then(response => {
        setCameras(response.data);
      })
      .catch(error => {
        console.error('There was an error fetching the camera data!', error);
      });
  }, []);

  const handleCameraChange = (event) => {
    const selectedCameraId = event.target.value;
    setSelectedCameraId(selectedCameraId);
    setCameraId(selectedCameraId); // Update cameraId state when a camera is selected
  };
  
  const handleStartCamera = async (cameraId) => {
    if (cameraId) {
      const success = await startCamera(cameraId);
      setCameraStarted(success);
    } else {
      alert("Please enter a valid camera ID.");
    }
  };
  
  const handleStopCamera = async () => {
    await api.post(" /api/artifact_keeper/camera/cleanup-temp-photos");
    await stopCamera(); // Stop camera functionality
    setCameraStarted(false);
  };

  const handleTargetLabelChange = (event) => {
    setTargetLabel(event.target.value);
  };

  return (
    <Container>
      <Stack spacing={3}>
        <div className="controls">
          <TextField label="Target Label" value={targetLabel} onChange={handleTargetLabelChange} />
          <Select
            labelId="camera-select-label"
            value={selectedCameraId}
            onChange={handleCameraChange}
            displayEmpty
          >
            <MenuItem value="" disabled>Select a Camera</MenuItem>
            {cameras.map((camera) => (
              <MenuItem key={camera.camera_id} value={camera.camera_id}>
                {camera.model}
              </MenuItem>
            ))}
          </Select>
        </div>
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