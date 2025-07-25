import React from 'react';
import { styled } from '@mui/material/styles';

const Box = styled('div')(({ submitted, active, theme }) => ({
  position: 'absolute',
  background: submitted ? 'blue' : 'rgba(0, 0, 0, 0.2)',
  border: `2px solid ${active ? 'yellow' : 'rgba(255, 255, 255, 0.8)'}`,
  boxShadow: active ? '0 0 10px yellow' : '0 0 5px rgba(0, 0, 0, 0.7)',
  transition: 'background 0.3s, border-color 0.3s, box-shadow 0.3s',
  cursor: 'pointer',
}));

const Label = styled('div')(({ theme }) => ({
  position: 'absolute',
  background: 'rgba(255, 255, 255, 0.8)',
  color: 'black',
  fontSize: '12px',
  padding: '2px 4px',
  borderRadius: '2px',
  top: '-20px',
  left: '50%',
  transform: 'translateX(-50%)',
  whiteSpace: 'nowrap',
}));

const FancyRectangle = React.forwardRef(({ 
  annotation, 
  className = '', 
  style = {},
  ...props 
}, ref) => {
  const { geometry, data, active, submitted } = annotation;

  if (!geometry) return null;

  return (
    <Box
      ref={ref}
      className={className}
      active={active}
      submitted={submitted}
      style={{
        height: `${geometry.height}%`,
        width: `${geometry.width}%`,
        top: `${geometry.y}%`,
        left: `${geometry.x}%`,
        ...style
      }}
      {...props}
    >
      {data && data.text && (
        <Label>
          {data.text}
        </Label>
      )}
    </Box>
  );
});

FancyRectangle.displayName = 'FancyRectangle';

export default FancyRectangle;