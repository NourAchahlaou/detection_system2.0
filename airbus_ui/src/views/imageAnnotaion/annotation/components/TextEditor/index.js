import React, { useState } from 'react';
import styled from 'styled-components';

const Inner = styled.div`
  padding: 8px 16px;

  textarea {
    border: 0;
    font-size: 14px;
    margin: 6px 0;
    min-height: 30px;
    outline: 0;
    width: 100%;
    z-index:999;
  }
`;

const Button = styled.div`
  background: whitesmoke;
  border: 0;
  box-sizing: border-box;
  color: #363636;
  cursor: pointer;
  font-size: 1rem;
  margin: 0;
  z-index:999;
  outline: 0;
  padding: 8px 16px;
  text-align: center;
  text-shadow: 0 1px 0 rgba(0, 0, 0, 0.1);
  width: 100%;

  transition: background 0.21s ease-in-out;

  &:focus,
  &:hover {
    background: #eeeeee;
  }
`;

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
