import React from 'react';
import styled from 'styled-components';

const Box = styled.div`
  position: relative; /* Set to relative to position the label inside it */
  background: ${({ submitted }) => (submitted ? 'blue' : 'rgba(0, 0, 0, 0.2)')};
  border: 2px solid ${({ active }) => (active ? 'yellow' : 'rgba(255, 255, 255, 0.8)')};
  box-shadow: ${({ active }) => (active ? '0 0 10px yellow' : '0 0 5px rgba(0, 0, 0, 0.7)')};
  height: ${({ height }) => height};
  width: ${({ width }) => width};
  top: ${({ top }) => top};
  left: ${({ left }) => left};
  transition: background 0.3s, border-color 0.3s, box-shadow 0.3s;
  cursor: pointer;
`;

const Label = styled.div`
  position: absolute;
  background: rgba(255, 255, 255, 0.8);
  color: black;
  font-size: 12px;
  padding: 2px 4px;
  border-radius: 2px;
  top: -20px; /* Adjust this value to position the label above the box */
  left: 50%;
  transform: translateX(-50%);
  white-space: nowrap;
`;

function FancyRectangle(props) {
  const { geometry, data, active, submitted } = props.annotation;

  if (!geometry) return null;

  return (
    <Box
      className={props.className}
      active={active}
      submitted={submitted}
      height={`${geometry.height}%`}
      width={`${geometry.width}%`}
      top={`${geometry.y}%`}
      left={`${geometry.x}%`}
    >
      {data && data.text && (
        <Label>
          {data.text}
        </Label>
      )}
    </Box>
  );
}

FancyRectangle.defaultProps = {
  className: '',
  style: {}
};

export default FancyRectangle;
