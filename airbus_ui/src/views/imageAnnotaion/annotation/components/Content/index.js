import React from 'react';
import { styled } from '@mui/material/styles';

const Container = styled('div')(({ theme }) => ({
  background: 'white',
  border: '1px solid black',
  borderRadius: '4px',
  boxShadow: `
    0px 1px 5px 0px rgba(0, 0, 0, 0.2),
    0px 2px 2px 0px rgba(0, 0, 0, 0.14),
    0px 3px 1px -2px rgba(0, 0, 0, 0.12)
  `,
  padding: '8px 16px',
  position: 'absolute',
  fontSize: '14px',
  color: '#333',
}));

function Content(props) {
  const { annotation } = props;
  if (!annotation) return null;

  const { geometry } = annotation;
  if (!geometry) return null;

  const { x = 0, y = 0, width = 0, height = 0 } = geometry;

  return (
    <Container
      style={{
        left: `${x}%`,
        top: `${y + height + 2}%`,
        ...props.style
      }}
      className={props.className}
    >
      {annotation.data && annotation.data.text ? annotation.data.text : 'No text available'}
    </Container>
  );
}

Content.defaultProps = {
  style: {},
  className: ''
};

export default Content;