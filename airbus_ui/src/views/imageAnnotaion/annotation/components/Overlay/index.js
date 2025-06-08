import React from 'react';
import { styled } from '@mui/material/styles';

const Overlay = styled('div')(({ theme }) => ({
  background: 'rgba(0, 0, 0, .4)',
  borderRadius: '5px',
  bottom: '4px',
  color: 'white',
  fontSize: '12px',
  fontWeight: 'bold',
  opacity: 0,
  padding: '10px',
  pointerEvents: 'none',
  position: 'absolute',
  right: '4px',
  transition: 'opacity 0.21s ease-in-out',
  userSelect: 'none',
}));

export default Overlay;