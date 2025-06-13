import React, { useState, useRef, useEffect } from 'react';
import { Box, Button, styled } from '@mui/material';
import Annotation from '../annotation'; // Make sure Annotation component is correctly implemented
import api from '../../../utils/UseAxios'; // Keep using your API

// Styled components to match VideoFeed styling (keeping your theme)
const AnnotationContainer = styled(Box)({
  position: 'relative',
  width: '100%',
  height: '100%',
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  overflow: 'hidden',
});

const AnnotationImage = styled('img')({
  width: '100%',
  height: '100%',
  objectFit: 'cover',
  display: 'block',
  border: 'none',
  outline: 'none',
});

// Floating controls matching VideoFeed style
const FloatingControls = styled(Box)(({ theme }) => ({
  position: 'absolute',
  bottom: '20px',
  left: '50%',
  transform: 'translateX(-50%)',
  display: 'flex',
  gap: '12px',
  alignItems: 'center',
  backdropFilter: 'blur(10px)',
  zIndex: 10,
  transition: 'opacity 0.3s ease',
}));

const SaveButton = styled(Button)({
  backgroundColor: '#667eea',
  color: 'white',
  fontWeight: '600',
  fontSize: '14px',
  textTransform: 'none',
  borderRadius: '25px',
  padding: '8px 20px',
  border: 'none',
  '&:hover': {
    backgroundColor: '#5a6fd8',
    transform: 'translateY(-1px)',
  },
  '&:active': {
    transform: 'translateY(0)',
  },
  transition: 'all 0.2s ease',
});

const ActionButton = styled(Button)({
  backgroundColor: 'rgba(102, 126, 234, 0.1)',
  color: '#667eea',
  fontWeight: '500',
  fontSize: '12px',
  textTransform: 'none',
  borderRadius: '20px',
  padding: '6px 16px',
  border: '1px solid rgba(102, 126, 234, 0.3)',
  '&:hover': {
    backgroundColor: 'rgba(102, 126, 234, 0.2)',
    borderColor: '#667eea',
  },
  '&:disabled': {
    opacity: 0.5,
    cursor: 'not-allowed',
  },
  transition: 'all 0.2s ease',
});

const PlaceholderContent = styled(Box)({
  display: 'flex',
  flexDirection: 'column',
  alignItems: 'center',
  justifyContent: 'center',
  textAlign: 'center',
  color: '#666',
  height: '100%',
  width: '100%',
});

export default function Simple({ imageUrl, annotated, pieceLabel,imageId  }) {
  // State management like paste 1 & 2
  const [annotations, setAnnotations] = useState([]);
  const [annotation, setAnnotation] = useState({});
  const [undoStack, setUndoStack] = useState([]);
  const [redoStack, setRedoStack] = useState([]);
  const [saving, setSaving] = useState(false);
  const containerRef = useRef(null);

  useEffect(() => {
    if (containerRef.current) {
      console.log('Container ref:', containerRef.current);
    }
  }, [containerRef]);

  // Load existing annotations if available
  useEffect(() => {
    if (annotated && Array.isArray(annotated)) {
      setAnnotations(annotated);
    }
  }, [annotated]);

  const onChange = (newAnnotation) => {
    setAnnotation(newAnnotation);
  };

// onSubmit functionality - NOW ACTUALLY SENDS TO BACKEND
  const onSubmit = async (newAnnotation) => {
    const { geometry, data } = newAnnotation;

    // Save current state to undo stack
    setUndoStack((prevUndoStack) => [...prevUndoStack, [...annotations]]);
    setRedoStack([]); // Clear redo stack after new action

    // Add new annotation to local state
    const newAnnotationWithId = {
      geometry,
      data: {
        ...data,
        id: Math.random(), // Use a unique identifier for each annotation
      },
    };

    setAnnotations((prevAnnotations) => [
      ...prevAnnotations,
      newAnnotationWithId,
    ]);
    setAnnotation({});

    // ADDED: Send annotation to backend virtual storage
    if (pieceLabel && imageId) {
      try {
        const annotationData = {
          image_id: parseInt(imageId),
          type: data.text || 'object', // Use text as type, or default to 'object'
          x: geometry.x,
          y: geometry.y,
          width: geometry.width,
          height: geometry.height
        };

        await api.post(`/api/annotation/annotations/${pieceLabel}`, annotationData);
        console.log('Annotation sent to backend:', annotationData);
      } catch (error) {
        console.error('Failed to send annotation to backend:', error);
        // Note: We don't revert the local state here, as the user can still save later
      }
    }
  };

  // Undo functionality from paste 1 & 2
  const undo = () => {
    if (undoStack.length === 0) return;

    const previousAnnotations = undoStack[undoStack.length - 1];
    setRedoStack((prevRedoStack) => [...prevRedoStack, [...annotations]]);
    setAnnotations(previousAnnotations);
    setUndoStack((prevUndoStack) => prevUndoStack.slice(0, -1));
  };

  // Redo functionality from paste 1 & 2
  const redo = () => {
    if (redoStack.length === 0) return;

    const nextAnnotations = redoStack[redoStack.length - 1];
    setUndoStack((prevUndoStack) => [...prevUndoStack, [...annotations]]);
    setAnnotations(nextAnnotations);
    setRedoStack((prevRedoStack) => prevRedoStack.slice(0, -1));
  };

  // Save functionality like paste 1 & 2 (save all annotations at once, then reload)
  const saveAnnotations = async () => {
    if (!pieceLabel) {
      console.error('No piece label provided');
      return;
    }
    
    if (annotation.length === 0) {
      console.error('No annotations to save');
      return;
    }

    try {
      setSaving(true);
      
      const response = await api.post(`/api/annotation/annotations/saveAnnotation/${pieceLabel}`, {
        piece_label: pieceLabel
      });


      if (response.status === 200) {
        console.log("Annotations saved successfully:", response.data);
        // Refresh the page after successful save (like paste 1 & 2)
        window.location.reload(); 
      } else {
        console.error("Error saving annotations:", response.data?.detail);
      }
    } catch (error) {
      console.error("An error occurred while saving annotations:", error.response?.data?.detail || error.message);
    } finally {
      setSaving(false);
    }
  };

  // Show placeholder when no image is selected
  if (!imageUrl) {
    return (
      <AnnotationContainer ref={containerRef}>
        <PlaceholderContent>
          <Box sx={{ fontSize: '48px', mb: 2, opacity: 0.6 }}>🖼️</Box>
          <Box sx={{ fontSize: '18px', fontWeight: 500, mb: 1 }}>
            No Image Selected
          </Box>
          <Box sx={{ fontSize: '14px', opacity: 0.7 }}>
            Select an image from the sidebar to start annotating
          </Box>
        </PlaceholderContent>
      </AnnotationContainer>
    );
  }

  return (
    <AnnotationContainer ref={containerRef}>
      <Annotation
        src={imageUrl}
        alt="Annotatable image"
        annotations={annotations}
        value={annotation}
        onChange={onChange}
        onSubmit={onSubmit}
        pieceLabel={pieceLabel}
        renderHighlight={({ annotation, active }) => (
          <div
            key={annotation.data.id}
            style={{
              position: 'absolute',
              left: `${annotation.geometry.x}%`,
              top: `${annotation.geometry.y}%`,
              width: `${annotation.geometry.width}%`,
              height: `${annotation.geometry.height}%`,
              border: active ? '2px solid #667eea' : '1px dashed #667eea',
              backgroundColor: active ? 'rgba(102, 126, 234, 0.2)' : 'rgba(102, 126, 234, 0.1)',
              pointerEvents: 'none',
            }}
          />
        )}
        renderContent={({ annotation }) => (
          <div style={{
            position: 'absolute',
            left: `${annotation.geometry.x}px`,
            top: `${annotation.geometry.y}px`,
            color: 'white',
            backgroundColor: 'rgba(102, 126, 234, 0.1)',
            padding: '4px 8px',
            borderRadius: '4px',
            fontSize: '12px',
            fontWeight: '500',
            pointerEvents: 'none',
          }}>
            {annotation.data.text}
          </div>
        )}
        style={{
          width: '100%',
          height: '100%',
          position: 'relative',
        }}
      />
      
      {/* Floating Controls - Always show like paste 1 & 2 */}
      <FloatingControls>
        <ActionButton 
          onClick={undo}
          disabled={undoStack.length === 0}
          size="small"
        >
          Undo
        </ActionButton>
        <ActionButton 
          onClick={redo}
          disabled={redoStack.length === 0}
          size="small"
        >
          Redo
        </ActionButton>
        <SaveButton 
          onClick={saveAnnotations}
          disabled={saving}
        >
          {saving ? 'Saving...' : 'Save'}
        </SaveButton>
      </FloatingControls>
    </AnnotationContainer>
  );
}