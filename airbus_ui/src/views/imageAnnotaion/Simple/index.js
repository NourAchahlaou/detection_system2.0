import React, { useState, useRef, useEffect } from 'react';
import Annotation from '../annotation'; // Make sure Annotation component is correctly implemented
import './Simple.css'; // Import your custom CSS file
import api from '../../../utils/UseAxios'; // Add this import

export default function Simple({ imageUrl, annotated, pieceLabel }) { // Accept pieceLabel as a prop
  const [annotations, setAnnotations] = useState([]);
  const [annotation, setAnnotation] = useState({});
  const [undoStack, setUndoStack] = useState([]);
  const [redoStack, setRedoStack] = useState([]);
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
    try {
      const response = await api.post(`/api/annotation/annotations/saveAnnotation/${pieceLabel}`, {
        piece_label: pieceLabel // Send the piece_label in the body
      });

      console.log("Annotations saved successfully:", response.data);
      // Refresh the page after successful save
      window.location.reload(); 
    } catch (error) {
      console.error("Error saving annotations:", error.response?.data?.detail || error.message);
    }
  };

  return (
    <div className="simple-container" ref={containerRef} style={{ position: 'relative' }}>
      <Annotation
        src={imageUrl} // Pass the image URL to Annotation component
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
              border: active ? '2px solid red' : '1px dashed yellow',
              backgroundColor: active ? 'rgba(255, 0, 0, 0.2)' : 'rgba(255, 255, 0, 0.2)',
              pointerEvents: 'none', // Ensure it doesn't interfere with mouse events
            }}
          />
        )}
        renderContent={({ annotation }) => (
          <div style={{
            position: 'absolute',
            left: `${annotation.geometry.x}px`,
            top: `${annotation.geometry.y}px`,
            color: 'black',
            backgroundColor: 'white',
            padding: '2px 5px',
            borderRadius: '3px',
            pointerEvents: 'none', // Ensure it doesn't interfere with mouse events
          }}>
            {annotation.data.text}
          </div>
        )}
      />
      <div className="controls" style={{ position: 'absolute', bottom: '10px', left: '10px' }}>
        <button onClick={saveAnnotations}>Save</button> {/* Save button triggers saveAnnotations function */}
      </div>
    </div>
  );
}