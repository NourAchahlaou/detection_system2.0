import React from 'react';
import { styled } from '@mui/material/styles';

const Container = styled('div')(({ active, theme, isOverlay = false }) => ({
  border: active ? '2px solid yellow' : isOverlay ? '2px dashed #4caf50' : '2px dashed black',
  background: isOverlay ? 'rgba(76, 175, 80, 0.2)' : 'rgba(255, 255, 255, 0.2)',
  boxShadow: active 
    ? '0 0 10px 2px rgba(255, 255, 0, 0.5)' 
    : isOverlay 
      ? '0 0 8px 1px rgba(76, 175, 80, 0.4)' 
      : 'none',
  boxSizing: 'border-box',
  position: 'absolute',
  cursor: active ? 'pointer' : 'default',
  transition: 'box-shadow 0.3s ease, border 0.3s ease',
  pointerEvents: isOverlay ? 'none' : 'auto', // Make overlay rectangles non-interactive
  zIndex: isOverlay ? 10 : 1,
}));

function Rectangle(props) {
  const { geometry, isOverlay = false, active = false } = props.annotation || props;
  
  if (!geometry) return null;

  const { x, y, width, height } = geometry;

  return (
    <Container
      active={active}
      isOverlay={isOverlay}
      style={{
        left: `${x}%`,
        top: `${y}%`,
        width: `${width}%`,
        height: `${height}%`,
        ...props.style
      }}
      className={props.className}
    />
  );
}

Rectangle.defaultProps = {
  className: '',
  style: {},
  annotation: { geometry: null }
};

export default Rectangle;