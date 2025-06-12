import React, { useState, useRef, useEffect } from 'react';
import { Box, Button, styled } from '@mui/material';
import Annotation from '../annotation'; // Make sure Annotation component is correctly implemented
import api from '../../../utils/UseAxios'; // Add this import

// Styled components to match VideoFeed styling
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
  display: 'block', // Removes any inline spacing
  border: 'none',   // Ensure no border
  outline: 'none',  // Ensure no outline
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

export default function Simple({ imageUrl, annotated, pieceLabel }) {
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

  const onChange = (newAnnotation) => {
    setAnnotation(newAnnotation);
  };

  const onSubmit = (newAnnotation) => {
    const { geometry, data } = newAnnotation;

    setUndoStack((prevUndoStack) => [...prevUndoStack, [...annotations]]);
    setRedoStack([]); // Clear redo stack after new action

    setAnnotations((prevAnnotations) => [
      ...prevAnnotations,
      {
        geometry,
        data: {
          ...data,
          id: Math.random(), // Use a unique identifier for each annotation
        },
      },
    ]);
    setAnnotation({});
  };

  const undo = () => {
    if (undoStack.length === 0) return;

    const previousAnnotations = undoStack.pop();
    setRedoStack((prevRedoStack) => [...prevRedoStack, [...annotations]]);
    setAnnotations(previousAnnotations);
    setUndoStack((prevUndoStack) => [...prevUndoStack]);
  };

  const redo = () => {
    if (redoStack.length === 0) return;

    const nextAnnotations = redoStack.pop();
    setUndoStack((prevUndoStack) => [...prevUndoStack, [...annotations]]);
    setAnnotations(nextAnnotations);
    setRedoStack((prevRedoStack) => [...prevRedoStack]);
  };

  const saveAnnotations = async () => {
    if (!pieceLabel) {
      console.error('No piece label provided');
      return;
    }

    try {
      setSaving(true);
      const response = await api.post(`/api/annotation/annotations/saveAnnotation/${pieceLabel}`, {
        piece_label: pieceLabel,
        annotations: annotations // Include the annotations data
      });

      console.log("Annotations saved successfully:", response.data);
      // Refresh the page after successful save
      window.location.reload(); 
    } catch (error) {
      console.error("Error saving annotations:", error.response?.data?.detail || error.message);
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
      
      {/* Floating Controls - Match VideoFeed style */}
      <FloatingControls>
        <SaveButton 
          onClick={saveAnnotations}
          disabled={saving || annotations.length === 0}
        >
          {saving ? 'Saving...' : `Save ${annotations.length} Annotation${annotations.length !== 1 ? 's' : ''}`}
        </SaveButton>
      </FloatingControls>
    </AnnotationContainer>
  );
}