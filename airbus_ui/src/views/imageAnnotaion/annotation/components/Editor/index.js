import React, { useState, useEffect } from 'react';
import TextEditor from '../TextEditor';

const Editor = ({ annotation, onChange, onSubmit, pieceLabel, style }) => {
  const [isEditing, setIsEditing] = useState(true);

  useEffect(() => {
    if (pieceLabel && annotation && annotation.geometry) {
      const autoAnnotationData = {
        text: pieceLabel,
        label: pieceLabel,
        id: Math.random()
      };
      
      const updatedAnnotation = {
        ...annotation,
        data: {
          // FIXED: Safely handle undefined annotation.data
          ...(annotation.data || {}),
          ...autoAnnotationData
        }
      };
      
      const timer = setTimeout(() => {
        onChange(updatedAnnotation);
        onSubmit(updatedAnnotation);
        setIsEditing(false);
      }, 800);
      
      return () => clearTimeout(timer);
    }
  }, [pieceLabel, annotation, onChange, onSubmit]);

  const handleTextEditorSubmit = (data) => {
    if (annotation && onChange && onSubmit) {
      const updatedAnnotation = {
        ...annotation,
        data: {
          // FIXED: Safely handle undefined annotation.data
          ...(annotation.data || {}),
          ...data
        }
      };
      
      onChange(updatedAnnotation);
      onSubmit(updatedAnnotation);
      setIsEditing(false);
    }
  };
  if (!isEditing || !annotation) {
    return null;
  }

  return (
    <div style={style}>
      <TextEditor
        pieceLabel={pieceLabel}
        onSubmit={handleTextEditorSubmit}
        value={annotation.data?.text || ''}
      />
    </div>
  );
};

export default Editor;