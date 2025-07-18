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

// NEW: Guide lines for precise annotation
const GuideLineVertical = styled('div')({
  position: 'absolute',
  width: '1px',
  height: '100%',
  backgroundColor: '#ff4444',
  opacity: 0.7,
  pointerEvents: 'none',
  zIndex: 20,
  transition: 'opacity 0.2s ease',
  boxShadow: '0 0 2px rgba(255, 68, 68, 0.5)',
});

const GuideLineHorizontal = styled('div')({
  position: 'absolute',
  height: '1px',
  width: '100%',
  backgroundColor: '#ff4444',
  opacity: 0.7,
  pointerEvents: 'none',
  zIndex: 20,
  transition: 'opacity 0.2s ease',
  boxShadow: '0 0 2px rgba(255, 68, 68, 0.5)',
});

// Grid overlay for additional precision
const GridOverlay = styled('div')({
  position: 'absolute',
  top: 0,
  left: 0,
  width: '100%',
  height: '100%',
  pointerEvents: 'none',
  zIndex: 15,
  opacity: 0.3,
  backgroundImage: `
    linear-gradient(rgba(255, 255, 255, 0.1) 1px, transparent 1px),
    linear-gradient(90deg, rgba(255, 255, 255, 0.1) 1px, transparent 1px)
  `,
  backgroundSize: '20px 20px',
  transition: 'opacity 0.3s ease',
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

// Piece label display component
const PieceLabelDisplay = styled(Box)({
  position: 'absolute',
  top: 20,
  left: 20,
  backgroundColor: 'rgba(102, 126, 234, 0.9)',
  color: 'white',
  padding: '8px 12px',
  borderRadius: '12px',
  fontSize: '12px',
  fontWeight: 'bold',
  zIndex: 15,
  border: '2px solid #667eea',
});

export default function Simple({ 
  imageUrl, 
  annotated, 
  pieceLabel, // This is the known piece label for the current image
  imageId, 
  onAnnotationSaved, 
  onMoveToNextImage,
  onRefreshImages, // Function to refresh image data
  onImageStatusChange // Function to notify when image annotation status changes
}) {
  // State management
  const [annotations, setAnnotations] = useState([]);
  const [annotation, setAnnotation] = useState({});
  const [undoStack, setUndoStack] = useState([]);
  const [saving, setSaving] = useState(false);
  
  // State for existing annotations from backend
  const [existingAnnotations, setExistingAnnotations] = useState([]);
  const [loadingExistingAnnotations, setLoadingExistingAnnotations] = useState(false);
  
  // Track which annotations are saved to backend vs virtual storage
  const [savedAnnotations, setSavedAnnotations] = useState([]);
  const [virtualAnnotations, setVirtualAnnotations] = useState([]);
  
  // NEW: Guide line state
  const [showGuides, setShowGuides] = useState(true);
  const [showGrid, setShowGrid] = useState(false);
  const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 });
  const [isAnnotating, setIsAnnotating] = useState(false);
  
  const containerRef = useRef(null);

  useEffect(() => {
    if (containerRef.current) {
      console.log('Container ref:', containerRef.current);
    }
  }, [containerRef]);

  // NEW: Mouse tracking for guide lines
  useEffect(() => {
    const handleMouseMove = (e) => {
      if (containerRef.current && showGuides) {
        const rect = containerRef.current.getBoundingClientRect();
        const x = ((e.clientX - rect.left) / rect.width) * 100;
        const y = ((e.clientY - rect.top) / rect.height) * 100;
        setMousePosition({ x: Math.max(0, Math.min(100, x)), y: Math.max(0, Math.min(100, y)) });
      }
    };

    const handleMouseEnter = () => {
      setIsAnnotating(true);
    };

    const handleMouseLeave = () => {
      setIsAnnotating(false);
    };

    const container = containerRef.current;
    if (container) {
      container.addEventListener('mousemove', handleMouseMove);
      container.addEventListener('mouseenter', handleMouseEnter);
      container.addEventListener('mouseleave', handleMouseLeave);
      
      return () => {
        container.removeEventListener('mousemove', handleMouseMove);
        container.removeEventListener('mouseenter', handleMouseEnter);
        container.removeEventListener('mouseleave', handleMouseLeave);
      };
    }
  }, [showGuides]);

  // Helper function to notify parent about image status changes
  const notifyImageStatusChange = (imageId, hasAnnotations) => {
    if (onImageStatusChange) {
      onImageStatusChange(imageId, hasAnnotations);
    }
  };

  // Load existing annotations when image changes with better error handling
  useEffect(() => {
    const fetchExistingAnnotations = async () => {
      if (!imageId) {
        setExistingAnnotations([]);
        setSavedAnnotations([]);
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
          type: ann.type || pieceLabel || 'annotation', // Use pieceLabel as fallback type
          isExisting: true,
          dbId: ann.id // Store the database ID for deletion
        }));
        
        setExistingAnnotations(formattedAnnotations);
        // Track which annotations are already saved in the database
        setSavedAnnotations(formattedAnnotations.map(ann => ann.dbId).filter(id => id));
        
      } catch (error) {
        console.log(`Could not fetch existing annotations for image ${imageId}:`, error.message);
        setExistingAnnotations([]);
        setSavedAnnotations([]);
      } finally {
        setLoadingExistingAnnotations(false);
      }
    };

    fetchExistingAnnotations();
  }, [imageId, pieceLabel]);

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
    setAnnotation({});
    setVirtualAnnotations([]);
  }, [imageUrl, imageId]);

  const onChange = (newAnnotation) => {
    setAnnotation(newAnnotation);
  };

  const onSubmit = async (newAnnotation) => {
    const { geometry } = newAnnotation;

    // Save current state to undo stack
    setUndoStack((prevUndoStack) => [...prevUndoStack, {
      annotations: [...annotations],
      savedAnnotations: [...savedAnnotations],
      virtualAnnotations: [...virtualAnnotations],
      action: 'add_annotation'
    }]);

    // FIXED: Automatically assign the piece label as the annotation data
    const annotationId = Math.random();
    const newAnnotationWithId = {
      geometry,
      data: {
        // FIXED: Safely handle undefined newAnnotation.data
        ...(newAnnotation.data || {}),
        id: annotationId,
        text: pieceLabel,
        label: pieceLabel,
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
          type: pieceLabel,
          x: geometry.x,
          y: geometry.y,
          width: geometry.width,
          height: geometry.height,
          label: pieceLabel
        };

        await api.post(`/api/annotation/annotations/${pieceLabel}`, annotationData);
        console.log('Annotation automatically saved with piece label:', annotationData);
        
        setVirtualAnnotations(prev => [...prev, annotationId]);
        
      } catch (error) {
        console.error('Failed to send annotation to backend:', error);
      }
    }
  };

  // Check if undo functionality is available
  const isUndoAvailable = () => {
    return (
      undoStack.length > 0 || 
      savedAnnotations.length > 0 || 
      virtualAnnotations.length > 0 || 
      annotations.length > 0 ||
      existingAnnotations.length > 0
    );
  };

  // Get undo button tooltip based on available actions
  const getUndoTooltip = () => {
    if (undoStack.length > 0) {
      return "Undo last action";
    } else if (annotations.length > 0) {
      return "Remove most recent annotation";
    } else if (virtualAnnotations.length > 0) {
      return "Remove last virtual annotation";
    } else if (savedAnnotations.length > 0) {
      return "Delete most recent saved annotation";
    } else if (existingAnnotations.length > 0) {
      return "Delete most recent existing annotation";
    } else {
      return "No actions to undo";
    }
  };

  // Undo functionality with immediate UI status updates
  const undo = async () => {
    // Priority 1: Use undo stack if available (most recent actions)
    if (undoStack.length > 0) {
      const previousState = undoStack[undoStack.length - 1];
      
      // Check if we're undoing an annotation that was added
      if (previousState.action === 'add_annotation') {
        const currentAnnotations = annotations;
        const previousAnnotations = previousState.annotations;
        
        // Find the annotation that was added (difference between current and previous)
        const addedAnnotation = currentAnnotations.find(curr => 
          !previousAnnotations.find(prev => prev.data.id === curr.data.id)
        );
        
        if (addedAnnotation) {
          const annotationId = addedAnnotation.data.id;
          
          // Check if this annotation is in virtual storage or database
          const isInVirtualStorage = virtualAnnotations.includes(annotationId);
          const isInDatabase = savedAnnotations.includes(annotationId);
          
          if (isInDatabase) {
            // Annotation is saved in database - delete it from database
            try {
              console.log('Deleting annotation from database:', annotationId);
              await api.delete(`/api/annotation/annotations/${annotationId}`);
              console.log('Successfully deleted annotation from database');
              
              // Remove from saved annotations tracking
              setSavedAnnotations(prev => prev.filter(id => id !== annotationId));
              
              // Refresh existing annotations immediately after deletion
              await refreshExistingAnnotations();
              
              // Check if image still has annotations and notify parent
              const remainingAnnotations = await checkImageAnnotationStatus();
              notifyImageStatusChange(imageId, remainingAnnotations > 0);
              
              // Notify parent to refresh image data and update status
              if (onRefreshImages) {
                await onRefreshImages();
              }
              
            } catch (error) {
              console.error('Failed to delete annotation from database:', error);
              // Don't revert UI state if database deletion fails
              return;
            }
          } else if (isInVirtualStorage) {
            // Annotation is in virtual storage - remove it from virtual storage
            try {
              console.log('Removing annotation from virtual storage:', annotationId);
              // Call backend to remove from virtual storage
              await api.delete(`/api/annotation/annotations/virtual/${pieceLabel}/${imageId}/${annotationId}`);
              console.log('Successfully removed annotation from virtual storage');
              
              // Remove from virtual annotations tracking
              setVirtualAnnotations(prev => prev.filter(id => id !== annotationId));
              
            } catch (error) {
              console.error('Failed to remove annotation from virtual storage:', error);
              // Continue with UI update even if virtual storage cleanup fails
            }
          }
        }
      }
      
      // Restore previous state
      setAnnotations(previousState.annotations);
      setSavedAnnotations(previousState.savedAnnotations);
      setVirtualAnnotations(previousState.virtualAnnotations);
      setUndoStack((prevUndoStack) => prevUndoStack.slice(0, -1));
      return;
    }

    // Priority 2: Remove most recent local annotation
    if (annotations.length > 0) {
      const lastAnnotation = annotations[annotations.length - 1];
      const annotationId = lastAnnotation.data.id;
      
      // Check if it's in virtual storage and remove it
      if (virtualAnnotations.includes(annotationId)) {
        try {
          await api.delete(`/api/annotation/annotations/virtual/${pieceLabel}/${imageId}/${annotationId}`);
          setVirtualAnnotations(prev => prev.filter(id => id !== annotationId));
        } catch (error) {
          console.error('Failed to remove annotation from virtual storage:', error);
        }
      }
      
      // Remove from local annotations
      setAnnotations(prev => prev.slice(0, -1));
      return;
    }

    // Priority 3: Remove from virtual storage (if any virtual annotations exist)
    if (virtualAnnotations.length > 0) {
      const lastVirtualId = virtualAnnotations[virtualAnnotations.length - 1];
      try {
        await api.delete(`/api/annotation/annotations/virtual/${pieceLabel}/${imageId}/${lastVirtualId}`);
        setVirtualAnnotations(prev => prev.slice(0, -1));
        console.log('Removed virtual annotation:', lastVirtualId);
      } catch (error) {
        console.error('Failed to remove virtual annotation:', error);
      }
      return;
    }

    // Priority 4: Delete most recent saved annotation from database
    if (savedAnnotations.length > 0) {
      const lastSavedId = savedAnnotations[savedAnnotations.length - 1];
      try {
        await api.delete(`/api/annotation/annotations/${lastSavedId}`);
        setSavedAnnotations(prev => prev.filter(id => id !== lastSavedId));
        
        // Refresh existing annotations immediately after deletion
        await refreshExistingAnnotations();
        
        // Check if image still has annotations and notify parent
        const remainingAnnotations = await checkImageAnnotationStatus();
        notifyImageStatusChange(imageId, remainingAnnotations > 0);
        
        // Notify parent to refresh image data and update status
        if (onRefreshImages) {
          await onRefreshImages();
        }
        
        console.log('Deleted saved annotation:', lastSavedId);
      } catch (error) {
        console.error('Failed to delete saved annotation:', error);
      }
      return;
    }

    // Priority 5: Delete most recent existing annotation from database
    if (existingAnnotations.length > 0) {
      const lastExisting = existingAnnotations[existingAnnotations.length - 1];
      if (lastExisting.dbId) {
        try {
          await api.delete(`/api/annotation/annotations/${lastExisting.dbId}`);
          
          // Remove from existing annotations immediately
          setExistingAnnotations(prev => prev.filter(ann => ann.dbId !== lastExisting.dbId));
          
          // Also remove from saved annotations tracking if it exists there
          setSavedAnnotations(prev => prev.filter(id => id !== lastExisting.dbId));
          
          // Refresh existing annotations to ensure consistency
          await refreshExistingAnnotations();
          
          // Check if image still has annotations and notify parent
          const remainingAnnotations = await checkImageAnnotationStatus();
          notifyImageStatusChange(imageId, remainingAnnotations > 0);
          
          // Notify parent to refresh image data and update status
          if (onRefreshImages) {
            await onRefreshImages();
          }
          
          console.log('Deleted existing annotation:', lastExisting.dbId);
        } catch (error) {
          console.error('Failed to delete existing annotation:', error);
        }
      }
      return;
    }

    console.log('No annotations to undo');
  };

  // Helper function to check current annotation status of image
  const checkImageAnnotationStatus = async () => {
    if (!imageId) return 0;
    
    try {
      let response;
      let backendAnnotations = [];
      
      try {
        response = await api.get(`/api/annotation/annotations/image/${imageId}/annotations`);
        backendAnnotations = response.data.annotations || [];
      } catch (error) {
        if (error.response?.status === 404) {
          try {
            response = await api.get(`/api/annotation/annotations/${imageId}`);
            backendAnnotations = response.data || [];
          } catch (secondError) {
            try {
              response = await api.get(`/api/annotation/annotations/existing/${imageId}`);
              backendAnnotations = response.data.annotations || [];
            } catch (thirdError) {
              backendAnnotations = [];
            }
          }
        } else {
          throw error;
        }
      }
      
      return backendAnnotations.length;
      
    } catch (error) {
      console.log('Error checking image annotation status:', error.message);
      return 0;
    }
  };

  // Helper function to refresh existing annotations
  const refreshExistingAnnotations = async () => {
    if (!imageId) return;
    
    try {
      let response;
      let backendAnnotations = [];
      
      try {
        response = await api.get(`/api/annotation/annotations/image/${imageId}/annotations`);
        backendAnnotations = response.data.annotations || [];
      } catch (error) {
        if (error.response?.status === 404) {
          try {
            response = await api.get(`/api/annotation/annotations/${imageId}`);
            backendAnnotations = response.data || [];
          } catch (secondError) {
            try {
              response = await api.get(`/api/annotation/annotations/existing/${imageId}`);
              backendAnnotations = response.data.annotations || [];
            } catch (thirdError) {
              console.log('No existing annotations found after deletion');
              backendAnnotations = [];
            }
          }
        } else {
          throw error;
        }
      }
      
      const formattedAnnotations = backendAnnotations.map((ann, index) => ({
        id: ann.id || `existing-${index}`,
        x: ann.x || 0,
        y: ann.y || 0,
        width: ann.width || 0,
        height: ann.height || 0,
        type: ann.type || pieceLabel || 'annotation',
        isExisting: true,
        dbId: ann.id
      }));
      
      setExistingAnnotations(formattedAnnotations);
      console.log('Refreshed existing annotations:', formattedAnnotations);
      
    } catch (error) {
      console.log('Error refreshing existing annotations:', error.message);
      setExistingAnnotations([]);
    }
  };

  // Save functionality - properly track saved annotations with immediate status update
  const saveAnnotations = async () => {
    if (!pieceLabel) {
      console.error('No piece label provided');
      return;
    }
    
    if (annotations.length === 0 && virtualAnnotations.length === 0) {
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
        
        // Track all current annotations as saved to database
        const currentAnnotationIds = annotations.map(ann => ann.data.id);
        setSavedAnnotations(prev => [...prev, ...currentAnnotationIds]);
        setVirtualAnnotations([]); // Clear virtual storage tracking since everything is now saved
        
        // Notify parent immediately that this image now has annotations
        notifyImageStatusChange(imageId, true);
        
        // Notify parent component that annotation was saved for THIS specific image
        if (onAnnotationSaved) {
          onAnnotationSaved(imageUrl, imageId);
        }

        // Refresh image data to update is_annotated status
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
        setAnnotation({});
        setSavedAnnotations([]);
        setVirtualAnnotations([]);
        
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
            backgroundColor: 'rgba(102, 126, 234, 0.9)',
            padding: '4px 8px',
            borderRadius: '4px',
            fontSize: '12px',
            fontWeight: '500',
            pointerEvents: 'none',
            border: '1px solid #667eea',
          }}>
            {annotation.data.text || annotation.data.label || pieceLabel}
          </div>
        )}
        style={{
          width: '100%',
          height: '100%',
          position: 'relative',
        }}
      />
      
      {/* NEW: Guide lines - only show when annotating and guides are enabled */}
      {showGuides && isAnnotating && imageUrl && (
        <>
          <GuideLineVertical
            style={{
              left: `${mousePosition.x}%`,
              opacity: isAnnotating ? 0.7 : 0,
            }}
          />
          <GuideLineHorizontal
            style={{
              top: `${mousePosition.y}%`,
              opacity: isAnnotating ? 0.7 : 0,
            }}
          />
        </>
      )}
      
      {/* Display the current piece label */}
      {pieceLabel && (
        <PieceLabelDisplay>
          Current Label: {pieceLabel}
        </PieceLabelDisplay>
      )}
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
            top: 60,
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
      
      {/* Status indicator for annotation types */}
      {(virtualAnnotations.length > 0 || savedAnnotations.length > 0) && (
        <Box
          sx={{
            position: 'absolute',
            top: 90,
            right: 20,
            backgroundColor: 'rgba(0, 0, 0, 0.8)',
            color: 'white',
            padding: '4px 8px',
            borderRadius: '12px',
            fontSize: '10px',
            zIndex: 15
          }}
        >
          Virtual: {virtualAnnotations.length} | Saved: {savedAnnotations.length}
        </Box>
      )}
      
      {/* Floating Controls */}
      <FloatingControls>
        <ActionButton 
          onClick={undo}
          disabled={!isUndoAvailable()}
          size="small"
          title={getUndoTooltip()}
        >
          Smart Undo
        </ActionButton>

        <SaveButton 
          onClick={saveAnnotations}
          disabled={saving || (annotations.length === 0 && virtualAnnotations.length === 0)}
        >
          {saving ? 'Saving...' : 'Save & Next'}
        </SaveButton>
      </FloatingControls>
    </AnnotationContainer>
  );
}