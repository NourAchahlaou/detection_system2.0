// components/camera/styledComponents.js
import { styled, Card, Box } from "@mui/material";

export const Container = styled("div")(({ theme }) => ({
  margin: "30px",
  [theme.breakpoints.down("sm")]: { margin: "16px" },
  "& .breadcrumb": {
    marginBottom: "30px",
    [theme.breakpoints.down("sm")]: { marginBottom: "16px" },
  },
}));

export const VideoCard = styled(Card)(({ theme, cameraActive }) => ({
  width: "900px",
  height: "480px",
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  backgroundColor: "#f5f5f5",
  border: cameraActive ? "2px solid #667eea" : "2px dashed #ccc",
  borderRadius: "12px",
  position: "relative",
  overflow: "hidden",
  padding: 0, 
  margin: 0,  
  // Override any Material-UI Card default styles
  '& .MuiCard-root': {
    padding: 0,
  },
  [theme.breakpoints.down("md")]: {
    width: "100%",
    maxWidth: "700px",
  },
  [theme.breakpoints.down("sm")]: {
    height: "300px",
  },
}));


export const PlaceholderContent = styled(Box)({
  display: "flex",
  flexDirection: "column",
  alignItems: "center",
  justifyContent: "center",
  textAlign: "center",
  color: "#666",
});

export const VideoImage = styled("img")({
  width: "100%",
  height: "100%",
  objectFit: "cover",
  display: "block", // Removes any inline spacing
  border: "none",   // Ensure no border
  outline: "none",  // Ensure no outline
});

export const FloatingControls = styled(Box)(({ theme }) => ({
  position: "absolute",
  bottom: "20px",
  left: "50%",
  transform: "translateX(-50%)",
  display: "flex",
  gap: "12px",
  alignItems: "center",
  padding: "12px 20px",
  background: "rgba(0, 0, 0, 0.8)",
  backdropFilter: "blur(10px)",
  borderRadius: "50px",
  border: "1px solid rgba(255, 255, 255, 0.1)",
  zIndex: 10,
  transition: "opacity 0.3s ease",
}));

export const StatusIndicator = styled(Box)(({ theme, active }) => ({
  position: "absolute",
  top: "20px",
  right: "20px",
  display: "flex",
  alignItems: "center",
  gap: "8px",
  padding: "8px 16px",
  background: active ? "rgba(76, 175, 80, 0.9)" : "rgba(244, 67, 54, 0.9)",
  color: "white",
  borderRadius: "20px",
  fontSize: "12px",
  fontWeight: "600",
  textTransform: "uppercase",
  letterSpacing: "0.5px",
  zIndex: 10,
}));

export const CaptureCounter = styled(Box)({
  position: "absolute",
  top: "20px",
  left: "20px",
  padding: "8px 16px",
  background: "rgba(0, 0, 0, 0.8)",
  color: "white",
  borderRadius: "20px",
  fontSize: "14px",
  fontWeight: "600",
  zIndex: 10,
});

export const SnapshotEffect = styled(Box)({
  position: "absolute",
  top: 0,
  left: 0,
  width: "100%",
  height: "100%",
  backgroundColor: "rgba(255, 255, 255, 0.27)",
  display: "flex",
  justifyContent: "center",
  alignItems: "center",
  color: "#333",
  fontSize: "2rem",
  fontWeight: "bold",
  zIndex: 15,
});

export const CapturedImagesStack = styled(Box)({
  position: "absolute",
  bottom: "80px",
  right: "20px",
  width: "120px",
});