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
  pointerEvents: isOverlay ? 'none' : 'auto',
  zIndex: isOverlay ? 10 : 1,
}));

const Rectangle = React.forwardRef(({ 
  annotation = { geometry: null }, 
  className = '', 
  style = {},
  ...props 
}, ref) => {
  const { geometry, isOverlay = false, active = false } = annotation || {};
  
  if (!geometry) return null;

  const { x, y, width, height } = geometry;

  return (
    <Container
      ref={ref}
      active={active}
      isOverlay={isOverlay}
      style={{
        left: `${x}%`,
        top: `${y}%`,
        width: `${width}%`,
        height: `${height}%`,
        ...style
      }}
      className={className}
      {...props}
    />
  );
});

Rectangle.displayName = 'Rectangle';

export default Rectangle;