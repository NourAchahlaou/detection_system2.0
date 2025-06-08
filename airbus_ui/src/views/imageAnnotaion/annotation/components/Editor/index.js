import React from 'react';
import styled, { keyframes } from 'styled-components';
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
const Container = styled.div`
  background: white;
  border-radius: 2px;
  box-shadow:
    0px 1px 5px 0px rgba(0, 0, 0, 0.2),
    0px 2px 2px 0px rgba(0, 0, 0, 0.14),
    0px 3px 1px -2px rgba(0, 0, 0, 0.12);
  margin-top: 16px;
  transform-origin: top left;
  animation: ${fadeInScale} 0.31s cubic-bezier(0.175, 0.885, 0.32, 1.275); /* Fading and scaling effect */
  overflow: hidden;
  position: absolute;
  transition: left 0.5s cubic-bezier(0.4, 0, 0.2, 1), top 0.5s cubic-bezier(0.4, 0, 0.2, 1), transform 0.5s cubic-bezier(0.4, 0, 0.2, 1); /* Smooth transition */
  z-index: 1000;
`;

function Editor(props) {
  const { geometry } = props.annotation;
  if (!geometry) return null;

  return (
    <Container
      className={props.className}
      style={{
        left: `${geometry.x}%`,
        top: `${geometry.y + geometry.height}%`,
        transform: 'translate(-50%, -100%)', // Center above the bounding box
        ...props.style
      }}
    >
      <TextEditor
        onChange={e => props.onChange({
          ...props.annotation,
          data: {
            ...props.annotation.data,
            text: e.target.value
          }
        })}
        onSubmit={props.onSubmit}
        value={props.annotation.data && props.annotation.data.text}
      />
    </Container>
  );
}

Editor.defaultProps = {
  className: '',
  style: {}
};

export default Editor;
