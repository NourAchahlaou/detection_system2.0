import React, { PureComponent as Component } from 'react';

// Function to check if the mouse is over a given element
const isMouseOverElement = ({ elem, e }) => {
  if (typeof window !== 'undefined' && elem) {
    const { clientX, clientY } = e;
    const rect = elem.getBoundingClientRect();
    const { left, right, bottom, top } = rect;

    console.log('Element Rect:', rect);
    console.log('Mouse Position:', { clientX, clientY });

    return clientX > left && clientX < right && clientY > top && clientY < bottom;
  }
  return false;
};

// Higher-Order Component to add mouse hovering functionality
const isMouseHovering = (key = 'isMouseHovering') => (DecoratedComponent) => {
  class IsMouseHovering extends Component {
    constructor(props) {
      super(props);
      this.state = {
        isHoveringOver: false
      };
      this.el = null; // Initialize the element reference
    }

    componentDidMount() {
      if (typeof window !== 'undefined') {
        document.addEventListener('mousemove', this.onMouseMove);
      }
    }

    componentWillUnmount() {
      if (typeof window !== 'undefined') {
        document.removeEventListener('mousemove', this.onMouseMove);
      }
    }

    onMouseMove = (e) => {
      if (typeof window !== 'undefined') {
        console.log('onMouseMove event:', e);
        console.log('Current Element:', this.el);
        this.setState({
          isHoveringOver: isMouseOverElement({ elem: this.el, e })
        });
      }
    };

    render() {
      const hocProps = {
        [key]: {
          innerRef: el => {
            console.log('Setting element ref:', el);
            this.el = el; // Set the DOM element reference
            if (el) {
              console.log('Element is set:', el);
            } else {
              console.log('Element is not set');
            }
          },
          isHoveringOver: this.state.isHoveringOver
        }
      };

      return (
        <DecoratedComponent
          {...this.props}
          {...hocProps}
        />
      );
    }
  }

  IsMouseHovering.displayName = `IsMouseHovering(${DecoratedComponent.displayName || DecoratedComponent.name || 'Component'})`;

  return IsMouseHovering;
};

export default isMouseHovering;
