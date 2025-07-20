// components/camera/CaptureDialog.jsx
import React from "react";
import {
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  Typography,
  Button
} from "@mui/material";
import { CameraAlt } from "@mui/icons-material";

const CaptureDialog = ({ 
  open, 
  onClose, 
  onSaveImages, 
  capturedImagesCount, 
  targetLabel 
}) => {
  return (
    <Dialog
      open={open}
      onClose={onClose}
      PaperProps={{
        sx: {
          borderRadius: "16px",
          padding: "8px",
        }
      }}
    >
      <DialogTitle sx={{ textAlign: "center", pb: 1 }}>
        <CameraAlt sx={{ fontSize: 40, color: "primary.main", mb: 1 }} />
        <Typography variant="h5" component="div">
          Capture Complete
        </Typography>
      </DialogTitle>
      <DialogContent sx={{ textAlign: "center", pt: 1 }}>
        <Typography variant="body1" sx={{ mb: 2 }}>
          Successfully captured {capturedImagesCount} images for <strong>{targetLabel}</strong>
        </Typography>
        <Typography variant="body2" color="text.secondary">
          Save these images to the database to continue.
        </Typography>
      </DialogContent>
      <DialogActions sx={{ justifyContent: "center", gap: 1, pb: 2 }}>
        <Button 
          onClick={onSaveImages} 
          variant="contained" 
          size="large"
          sx={{ minWidth: 120 }}
        >
          Save Images
        </Button>
        <Button 
          onClick={onClose} 
          variant="outlined"
          size="large"
          sx={{ minWidth: 120 }}
        >
          Cancel
        </Button>
      </DialogActions>
    </Dialog>
  );
};

export default CaptureDialog;