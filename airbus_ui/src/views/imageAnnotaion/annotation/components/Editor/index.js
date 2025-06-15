import React from 'react';
import { styled, keyframes } from '@mui/material/styles';
import TextEditor from '../TextEditor';

// Keyframes for fading in and scaling the component
const fadeInScale = keyframes`
  from {
    opacity: 0;
    transform: scale(0);
  }
  to {
    opacity: 1;
    transform: scale(1);
  }
`;

// Styled Container component with animation and transition
const Container = styled('div')(({ theme }) => ({
  background: 'white',
  borderRadius: '2px',
  boxShadow: `
    0px 1px 5px 0px rgba(0, 0, 0, 0.2),
    0px 2px 2px 0px rgba(0, 0, 0, 0.14),
    0px 3px 1px -2px rgba(0, 0, 0, 0.12)
  `,
  marginTop: '16px',
  transformOrigin: 'top left',
  animation: `${fadeInScale} 0.31s cubic-bezier(0.175, 0.885, 0.32, 1.275)`,
  overflow: 'hidden',
  position: 'absolute',
  transition: 'left 0.5s cubic-bezier(0.4, 0, 0.2, 1), top 0.5s cubic-bezier(0.4, 0, 0.2, 1), transform 0.5s cubic-bezier(0.4, 0, 0.2, 1)',
  zIndex: 1000,
}));

const Editor = React.forwardRef(({ 
  annotation, 
  className = '', 
  style = {}, 
  onChange, 
  onSubmit,
  ...props 
}, ref) => {
  const { geometry } = annotation;
  if (!geometry) return null;

  return (
    <Container
      ref={ref}
      className={className}
      style={{
        left: `${geometry.x}%`,
        top: `${geometry.y + geometry.height}%`,
        transform: 'translate(-50%, -100%)',
        ...style
      }}
      {...props}
    >
      <TextEditor
        onChange={e => onChange({
          ...annotation,
          data: {
            ...annotation.data,
            text: e.target.value
          }
        })}
        onSubmit={onSubmit}
        value={annotation.data && annotation.data.text}
      />
    </Container>
  );
});

Editor.displayName = 'Editor';

export default Editor;