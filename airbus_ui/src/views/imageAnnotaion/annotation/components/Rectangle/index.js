import React from 'react';
import { styled } from '@mui/material/styles';

const Container = styled('div')(({ active, theme }) => ({
  border: active ? '2px solid yellow' : '2px dashed black',
  background: 'rgba(255, 255, 255, 0.2)',
  boxShadow: active ? '0 0 10px 2px rgba(255, 255, 0, 0.5)' : 'none',
  boxSizing: 'border-box',
  position: 'absolute',
  cursor: 'pointer',
  transition: 'box-shadow 0.3s ease, border 0.3s ease',
}));

function Rectangle(props) {
  const { geometry } = props.annotation;
  if (!geometry) return null;

  return (
    <Container
      className={props.className}
      style={{
        left: `${geometry.x}%`,
        top: `${geometry.y}%`,
        height: `${geometry.height}%`,
        width: `${geometry.width}%`,
        ...props.style
      }}
      active={props.active}
    />
  );
}

Rectangle.defaultProps = {
  className: '',
  style: {}
};

export default Rectangle;