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

const ExistingAnnotationOverlay = styled(Box)({
  position: 'absolute',
  border: '2px dashed #4caf50',
  backgroundColor: 'rgba(76, 175, 80, 0.1)',
  boxShadow: '0 0 8px 1px rgba(76, 175, 80, 0.3)',
  pointerEvents: 'none',
  zIndex: 5,
  borderRadius: '2px',
});

export default function Simple({ 
  imageUrl, 
  annotated, 
  pieceLabel, 
  imageId, 
  onAnnotationSaved, 
  onMoveToNextImage,
  onRefreshImages // ADD: New prop to refresh image data
}) {
  // State management
  const [annotations, setAnnotations] = useState([]);
  const [annotation, setAnnotation] = useState({});
  const [undoStack, setUndoStack] = useState([]);
  const [redoStack, setRedoStack] = useState([]);
  const [saving, setSaving] = useState(false);
  
  // NEW: State for existing annotations from backend
  const [existingAnnotations, setExistingAnnotations] = useState([]);
  const [loadingExistingAnnotations, setLoadingExistingAnnotations] = useState(false);
  
  const containerRef = useRef(null);

  useEffect(() => {
    if (containerRef.current) {
      console.log('Container ref:', containerRef.current);
    }
  }, [containerRef]);

  // FIXED: Load existing annotations when image changes with better error handling
  useEffect(() => {
    const fetchExistingAnnotations = async () => {
      if (!imageId) {
        setExistingAnnotations([]);
        return;
      }
      
      try {
        setLoadingExistingAnnotations(true);
        
        // Try multiple possible API endpoints
        let response;
        let backendAnnotations = [];
        
        try {
          // First try the original endpoint
          response = await api.get(`/api/annotation/annotations/image/${imageId}/annotations`);
          backendAnnotations = response.data.annotations || [];
        } catch (error) {
          if (error.response?.status === 404) {
            // Try alternative endpoint structure
            try {
              response = await api.get(`/api/annotation/annotations/${imageId}`);
              backendAnnotations = response.data || [];
            } catch (secondError) {
              // Try another possible endpoint
              try {
                response = await api.get(`/api/annotation/annotations/existing/${imageId}`);
                backendAnnotations = response.data.annotations || [];
              } catch (thirdError) {
                console.log('No existing annotations endpoint available, continuing without existing annotations');
                backendAnnotations = [];
              }
            }
          } else {
            throw error;
          }
        }
        
        // Convert backend annotations to the format expected by Rectangle component
        const formattedAnnotations = backendAnnotations.map((ann, index) => ({
          id: ann.id || `existing-${index}`,
          x: ann.x || 0,
          y: ann.y || 0,
          width: ann.width || 0,
          height: ann.height || 0,
          type: ann.type || 'annotation',
          isExisting: true
        }));
        
        setExistingAnnotations(formattedAnnotations);
      } catch (error) {
        console.log(`Could not fetch existing annotations for image ${imageId}:`, error.message);
        setExistingAnnotations([]);
      } finally {
        setLoadingExistingAnnotations(false);
      }
    };

    fetchExistingAnnotations();
  }, [imageId]);

  // Load existing annotations when image changes
  useEffect(() => {
    if (annotated && Array.isArray(annotated)) {
      setAnnotations(annotated);
    } else {
      // Clear annotations when switching to new image
      setAnnotations([]);
    }
    // Clear undo/redo stacks when switching images
    setUndoStack([]);
    setRedoStack([]);
    setAnnotation({});
  }, [imageUrl, imageId]);

  const onChange = (newAnnotation) => {
    setAnnotation(newAnnotation);
  };

  // onSubmit functionality - sends to backend
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

    // Send annotation to backend virtual storage
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

  // Undo functionality
  const undo = () => {
    if (undoStack.length === 0) return;

    const previousAnnotations = undoStack[undoStack.length - 1];
    setRedoStack((prevRedoStack) => [...prevRedoStack, [...annotations]]);
    setAnnotations(previousAnnotations);
    setUndoStack((prevUndoStack) => prevUndoStack.slice(0, -1));
  };

  // Redo functionality
  const redo = () => {
    if (redoStack.length === 0) return;

    const nextAnnotations = redoStack[redoStack.length - 1];
    setUndoStack((prevUndoStack) => [...prevUndoStack, [...annotations]]);
    setAnnotations(nextAnnotations);
    setRedoStack((prevRedoStack) => prevRedoStack.slice(0, -1));
  };

  // FIXED: Updated save functionality - properly notify parent and refresh
  const saveAnnotations = async () => {
    if (!pieceLabel) {
      console.error('No piece label provided');
      return;
    }
    
    if (annotations.length === 0) {
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
        
        // FIXED: Notify parent component that annotation was saved for THIS specific image
        if (onAnnotationSaved) {
          onAnnotationSaved(imageUrl, imageId);
        }

        // FIXED: Refresh image data to update is_annotated status
        if (onRefreshImages) {
          await onRefreshImages();
        }

        // Move to next image smoothly
        if (onMoveToNextImage) {
          onMoveToNextImage();
        }
        
        // Clear current annotations and reset state for next image
        setAnnotations([]);
        setUndoStack([]);
        setRedoStack([]);
        setAnnotation({});
        
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
      
      {/* FIXED: Render existing annotations as simple overlays */}
      {existingAnnotations.map((existingAnnotation, index) => (
        <ExistingAnnotationOverlay
          key={`existing-${existingAnnotation.id || index}`}
          sx={{
            left: `${existingAnnotation.x}%`,
            top: `${existingAnnotation.y}%`,
            width: `${existingAnnotation.width}%`,
            height: `${existingAnnotation.height}%`,
          }}
        />
      ))}
      
      {/* Loading indicator for existing annotations */}
      {loadingExistingAnnotations && (
        <Box
          sx={{
            position: 'absolute',
            top: 20,
            right: 20,
            backgroundColor: 'rgba(0, 0, 0, 0.8)',
            color: 'white',
            padding: '8px 12px',
            borderRadius: '4px',
            fontSize: '12px',
            zIndex: 15
          }}
        >
          Loading existing annotations...
        </Box>
      )}
      
      {/* Show count of existing annotations */}
      {!loadingExistingAnnotations && existingAnnotations.length > 0 && (
        <Box
          sx={{
            position: 'absolute',
            top: 20,
            right: 20,
            backgroundColor: 'rgba(76, 175, 80, 0.9)',
            color: 'white',
            padding: '4px 8px',
            borderRadius: '12px',
            fontSize: '11px',
            fontWeight: 'bold',
            zIndex: 15
          }}
        >
          {existingAnnotations.length} existing annotation{existingAnnotations.length !== 1 ? 's' : ''}
        </Box>
      )}
      
      {/* Floating Controls */}
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
          disabled={saving || annotations.length === 0}
        >
          {saving ? 'Saving...' : 'Save & Next'}
        </SaveButton>
      </FloatingControls>
    </AnnotationContainer>
  );
}