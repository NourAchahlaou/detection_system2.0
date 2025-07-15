import React from 'react';
import {
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  Button,
  Typography,
} from "@mui/material";

export default function ConfirmationDialog({ 
  open, 
  onClose, 
  selectedCount,
  onConfirm
}) {
  return (
    <Dialog
      open={open}
      onClose={() => onClose(false)}
      PaperProps={{
        sx: { borderRadius: "16px", padding: "8px" }
      }}
    >
      <DialogTitle sx={{ fontWeight: "600", color: "#333" }}>
        Delete Selected Items
      </DialogTitle>
      <DialogContent>
        <Typography>
          Are you sure you want to delete {selectedCount} selected item{selectedCount !== 1 ? 's' : ''}? 
          This action cannot be undone.
        </Typography>
      </DialogContent>
      <DialogActions sx={{ gap: 1, padding: "16px 24px" }}>
        <Button 
          onClick={() => onClose(false)}
          sx={{ textTransform: "none" }}
        >
          Cancel
        </Button>
        <Button 
          onClick={() => onConfirm(true)} 
          color="error"
          variant="contained"
          sx={{ textTransform: "none" }}
        >
          Delete
        </Button>
      </DialogActions>
    </Dialog>
  );
}