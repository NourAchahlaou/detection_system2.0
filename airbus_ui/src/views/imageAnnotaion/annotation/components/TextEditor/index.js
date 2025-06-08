import React, { useState } from 'react';
import { styled } from '@mui/material/styles';

const Inner = styled('div')(({ theme }) => ({
  padding: '8px 16px',
  '& textarea': {
    border: 0,
    fontSize: '14px',
    margin: '6px 0',
    minHeight: '30px',
    outline: 0,
    width: '100%',
    zIndex: 999,
  }
}));

const Button = styled('div')(({ theme }) => ({
  background: 'whitesmoke',
  border: 0,
  boxSizing: 'border-box',
  color: '#363636',
  cursor: 'pointer',
  fontSize: '1rem',
  margin: 0,
  zIndex: 999,
  outline: 0,
  padding: '8px 16px',
  textAlign: 'center',
  textShadow: '0 1px 0 rgba(0, 0, 0, 0.1)',
  width: '100%',
  transition: 'background 0.21s ease-in-out',
  '&:focus, &:hover': {
    background: '#eeeeee',
  }
}));

function TextEditor(props) {
  const [isVisible, setIsVisible] = useState(true);

  const handleSubmit = () => {
    setIsVisible(false);
    props.onSubmit();
  };

  return (
    <React.Fragment>
      {isVisible && (
        <React.Fragment>
          <Inner>
            <textarea
              placeholder="Write description"
              onFocus={props.onFocus}
              onBlur={props.onBlur}
              onChange={props.onChange}
              value={props.value}
            />
          </Inner>
          {props.value && (
            <Button onClick={handleSubmit}>
              Submit
            </Button>
          )}
        </React.Fragment>
      )}
    </React.Fragment>
  );
}

export default TextEditor;