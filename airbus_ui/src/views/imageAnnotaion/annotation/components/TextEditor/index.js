import React, { useState, useEffect } from 'react';
import { styled } from '@mui/material/styles';

const Inner = styled('div')(({ theme }) => ({
  padding: '8px 16px',
  backgroundColor: 'rgba(102, 126, 234, 0.9)',
  borderRadius: '8px',
  border: '2px solid #667eea',
  boxShadow: '0 4px 12px rgba(0, 0, 0, 0.15)',
}));

const LabelDisplay = styled('div')(({ theme }) => ({
  color: 'white',
  fontSize: '14px',
  fontWeight: 'bold',
  textAlign: 'center',
  padding: '4px 0',
  margin: '2px 0',
}));

const Button = styled('div')(({ theme }) => ({
  background: '#667eea',
  border: 0,
  boxSizing: 'border-box',
  color: 'white',
  cursor: 'pointer',
  fontSize: '12px',
  fontWeight: '600',
  margin: '4px 0 0 0',
  outline: 0,
  padding: '6px 12px',
  textAlign: 'center',
  width: '100%',
  borderRadius: '4px',
  transition: 'background 0.21s ease-in-out',
  '&:focus, &:hover': {
    background: '#5a6fd8',
  }
}));

function TextEditor({ pieceLabel, onSubmit, onFocus, onBlur }) {
  const [isVisible, setIsVisible] = useState(true);
  
  // Auto-submit when component mounts since we have the piece label
  useEffect(() => {
    if (pieceLabel && isVisible) {
      // Small delay to show the label briefly before auto-submitting
      const timer = setTimeout(() => {
        handleSubmit();
      }, 800);
      
      return () => clearTimeout(timer);
    }
  }, [pieceLabel, isVisible]);

  const handleSubmit = () => {
    setIsVisible(false);
    // Pass the piece label data to the parent
    if (onSubmit) {
      onSubmit({
        text: pieceLabel,
        label: pieceLabel,
        id: Math.random() // Generate unique ID
      });
    }
  };

  const handleManualSubmit = () => {
    handleSubmit();
  };

  return (
    <React.Fragment>
      {isVisible && pieceLabel && (
        <React.Fragment>
          <Inner>
            <LabelDisplay>
              {pieceLabel}
            </LabelDisplay>
            <Button onClick={handleManualSubmit}>
              Confirm Label
            </Button>
          </Inner>
        </React.Fragment>
      )}
    </React.Fragment>
  );
}

export default TextEditor;