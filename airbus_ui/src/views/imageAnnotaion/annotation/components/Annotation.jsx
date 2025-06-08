import React, { Component } from 'react';
import T from 'prop-types';
import styled from 'styled-components';
import compose from '../utils/compose';
import isMouseHovering from '../utils/isMouseHovering';
import withRelativeMousePos from '../utils/withRelativeMousePos';
import defaultProps from './defaultProps';
import Overlay from './Overlay';

const Container = styled.div`
  clear: both;
  position: relative;
  width: 100%;
  &:hover ${Overlay} {
    opacity: 1;
  }
  touch-action: ${(props) => (props.allowTouch ? 'pinch-zoom' : 'auto')};
`;

const Img = styled.img`
  display: block;
  width: 100%;
 
`;

const Items = styled.div`
  position: absolute;
  top: 0;
  left: 0;
  bottom: 0;
  right: 0;
`;

const Target = Items;

export default compose(
  isMouseHovering(),
  withRelativeMousePos()
)(class Annotation extends Component {
  static propTypes = {

    innerRef: T.func,
    onMouseUp: T.func,
    onMouseDown: T.func,
    onMouseMove: T.func,
    onClick: T.func,
    children: T.node,

    annotations: T.arrayOf(
      T.shape({
        type: T.string
      })
    ).isRequired,
    type: T.string,
    selectors: T.arrayOf(
      T.shape({
        TYPE: T.string,
        intersects: T.func.isRequired,
        area: T.func.isRequired,
        methods: T.object.isRequired
      })
    ).isRequired,
    value: T.shape({
      selection: T.object,
      geometry: T.shape({
        type: T.string.isRequired
      }),
      data: T.object
    }),

    onChange: T.func,
    onSubmit: T.func,

    activeAnnotationComparator: T.func,
    activeAnnotations: T.arrayOf(T.any),

    disableAnnotation: T.bool,
    disableSelector: T.bool,
    renderSelector: T.func,
    disableEditor: T.bool,
    renderEditor: T.func,

    renderHighlight: T.func.isRequired,
    renderContent: T.func.isRequired,

    disableOverlay: T.bool,
    renderOverlay: T.func.isRequired,
    allowTouch: T.bool
  };

  static defaultProps = defaultProps;

  targetRef = React.createRef();

  handleSaveAllAnnotations = () => {
    const { annotations } = this.props;

    // Send all annotations to the backend
    const API_URL = "http://127.0.0.1:8000";
    fetch(`${API_URL}/piece/annotations/save-all`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ annotations }) // Send all annotations at once
    })
    .then(response => response.json())
    .then(data => {
      console.log("All annotations saved successfully:", data);
    })
    .catch(error => {
      console.error("Error saving all annotations:", error);
    });
  };
  handleSaveCoordinates = () => {
    const { coordinates } = this.props.value.geometry || {};
    if (!coordinates) return; // Exit if no coordinates are available
    const { type, x, y, width, height } = coordinates;
    const content = `Type: ${type}\nX: ${x}\nY: ${y}\nWidth: ${width}\nHeight: ${height}`;
    // Create a Blob with the text content
    const blob = new Blob([content], { type: 'text/plain;charset=utf-8' });
    // Create a URL for the Blob
    const url = URL.createObjectURL(blob);
    // Create an anchor element and trigger the download
    const a = document.createElement('a');
    a.href = url;
    a.download = 'coordinates.txt'; // Set the file name
    document.body.appendChild(a); // Append the anchor to the DOM
    a.click(); // Simulate a click to trigger the download
    // Clean up: remove the anchor element and revoke the Blob URL
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };
  
  componentDidMount() {
    if (this.props.allowTouch && this.targetRef.current) {
      this.addTargetTouchEventListeners();
      
    }
  }
  

  addTargetTouchEventListeners = () => {
    if (this.targetRef.current) {
      this.targetRef.current.ontouchstart = this.onTouchStart;
      this.targetRef.current.ontouchend = this.onTouchEnd;
      this.targetRef.current.ontouchmove = this.onTargetTouchMove;
      this.targetRef.current.ontouchcancel = this.onTargetTouchLeave;
    }
  };

  removeTargetTouchEventListeners = () => {
    if (this.targetRef.current) {
      this.targetRef.current.ontouchstart = undefined;
      this.targetRef.current.ontouchend = undefined;
      this.targetRef.current.ontouchmove = undefined;
      this.targetRef.current.ontouchcancel = undefined;
    }
  };

  componentDidUpdate(prevProps) {
    if (this.props.allowTouch !== prevProps.allowTouch) {
      if (this.props.allowTouch) {
        this.addTargetTouchEventListeners();
      } else {
        this.removeTargetTouchEventListeners();
      }
    }
  }

  setInnerRef = (el) => {
    this.container = el;
    this.props.relativeMousePos.innerRef(el);
    this.props.innerRef(el);
  };

  getSelectorByType = (type) => {
    return this.props.selectors.find(s => s.TYPE === type);
  };

  getTopAnnotationAt = (x, y) => {
    const { annotations } = this.props;
    const { container, getSelectorByType } = this;

    if (!container) return;

    const intersections = annotations
      .map(annotation => {
        const { geometry } = annotation;
        const selector = getSelectorByType(geometry.type);

        return selector.intersects({ x, y }, geometry, container)
          ? annotation
          : false;
      })
      .filter(a => !!a)
      .sort((a, b) => {
        const aSelector = getSelectorByType(a.geometry.type);
        const bSelector = getSelectorByType(b.geometry.type);

        return aSelector.area(a.geometry, container) - bSelector.area(b.geometry, container);
      });

    return intersections[0];
  };

  onTargetMouseMove = (e) => {
    this.props.relativeMousePos.onMouseMove(e);
    this.onMouseMove(e);
  };

  onTargetTouchMove = (e) => {
    this.props.relativeMousePos.onTouchMove(e);
    this.onTouchMove(e);
  };

  onTargetMouseLeave = (e) => {
    this.props.relativeMousePos.onMouseLeave(e);
  };

  onTargetTouchLeave = (e) => {
    this.props.relativeMousePos.onTouchLeave(e);
  };

  onMouseUp = (e) => {
    this.callSelectorMethod('onMouseUp', e);
  };

  onMouseDown = (e) => {
    this.callSelectorMethod('onMouseDown', e);
  };

  onMouseMove = (e) => {
    this.callSelectorMethod('onMouseMove', e);
  };

  onTouchStart = (e) => {
    this.callSelectorMethod('onTouchStart', e);
  };

  onTouchEnd = (e) => {
    this.callSelectorMethod('onTouchEnd', e);
  };

  onTouchMove = (e) => {
    this.callSelectorMethod('onTouchMove', e);
  };

  onClick = (e) => {
    this.callSelectorMethod('onClick', e);
  };
  onSubmit = async () => {
    const { annotations, value, onChange } = this.props;
    const piece_label = value.data.text || {};
    const { type, x, y, width, height } = value.geometry;
    
    // Function to format the image URL
    const formatImageUrl = (url) => {
      const baseUrl = "http://localhost:8000/images/";
      if (url.startsWith(baseUrl)) {
        url = url.substring(baseUrl.length);
      }
      return url.replace(/\//g, '\\');
    };
  
    // Format the image URL
    const formattedUrl = formatImageUrl(this.props.src);
  
    try {
      // Fetch the image ID based on the formatted image URL
      const response = await fetch(`http://127.0.0.1:8000/piece/image-id?url=${encodeURIComponent(formattedUrl)}`, {
        method: 'GET',  // Use GET request
        headers: {
          'Content-Type': 'application/json'
        }
      });
  
      if (!response.ok) {
        throw new Error('Failed to fetch image ID');
      }
  
      const { image_id } = await response.json();
      console.log("Formatted URL and Image ID:", formattedUrl, image_id);
  
      // Prepare the annotation data including the image ID
      const annotationData = {
        type,
        x,
        y,
        width,
        height,
        data: value.data, // Add any other data you need to keep
        image_id // Include the image ID
      };
  
      // Save the new annotation to the backend
      const API_URL = "http://127.0.0.1:8000";
      const saveResponse = await fetch(`${API_URL}/piece/annotations/${piece_label}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(annotationData)
      });
  
      if (!saveResponse.ok) {
        throw new Error('Failed to save annotation');
      }
  
      const data = await saveResponse.json();
      console.log("Annotation saved successfully:", data);
  
      // Update the annotations array with the new annotation
      const updatedAnnotations = [...annotations, { ...annotationData, data }];
      onChange({ ...value, annotations: updatedAnnotations });
    } catch (error) {
      console.error("Error saving annotation:", error);
    }
  };
  
  callSelectorMethod = (methodName, e) => {
    
    if (this.props.disableAnnotation) {
      return;
    }

    if (!!this.props[methodName]) {
      this.props[methodName](e);
    } else {
      const selector = this.getSelectorByType(this.props.type);
      if (selector && selector.methods[methodName]) {
        const value = selector.methods[methodName](this.props.value, e);

        if (typeof value === 'undefined') {
          if (process.env.NODE_ENV !== 'production') {
            console.error(`
              ${methodName} of selector type ${this.props.type} returned undefined.
              Make sure to explicitly return the previous state
            `);
          }
        } else {
          this.props.onChange(value);
        }
      }
    }
  };

  shouldAnnotationBeActive = (annotation, top) => {
 
    if (this.props.activeAnnotations) {
      const isActive = !!this.props.activeAnnotations.find(active => (
        this.props.activeAnnotationComparator(annotation, active)
      ));

      return isActive || top === annotation;
    } else {
      return top === annotation;
    }
  };

  render() {
   
    const { props } = this;
    const {
      isMouseHovering,
      renderHighlight,
      renderContent,
      renderSelector,
      renderEditor,
      renderOverlay,
      allowTouch
    } = props;

    const topAnnotationAtMouse = this.getTopAnnotationAt(
      this.props.relativeMousePos.x,
      this.props.relativeMousePos.y
    );
    const coordinates = props.value.geometry || {};
    const { type, x, y, width, height } = coordinates;

    return (
      <Container
        style={props.style}
        innerRef={isMouseHovering.innerRef}
        onMouseLeave={this.onTargetMouseLeave}
        onTouchCancel={this.onTargetTouchLeave}
        allowTouch={allowTouch}
      >
        <Img
          className={props.className}
          style={props.style}
          alt={props.alt}
          src={props.src}
          draggable={false}
          innerRef={this.setInnerRef}
        />
        <Items>
          {props.annotations.map(annotation => (
            renderHighlight({
              key: annotation.data.id,
              annotation,
              active: this.shouldAnnotationBeActive(annotation, topAnnotationAtMouse)
            })
          ))}
          {!props.disableSelector
            && props.value
            && props.value.geometry
            && (
              renderSelector({
                annotation: props.value
              })
            )
          }
        </Items>

        <Target
          innerRef={this.targetRef}
          onClick={this.onClick}
          onMouseUp={this.onMouseUp}
          onMouseDown={this.onMouseDown}
          onMouseMove={this.onTargetMouseMove}
        />
        {!props.disableOverlay && (
          renderOverlay({
            type: props.type,
            annotation: props.value
          })
        )}
        {props.annotations.map(annotation => (
          this.shouldAnnotationBeActive(annotation, topAnnotationAtMouse)
          && (
            renderContent({
              key: annotation.data.id,
              annotation: annotation
            })
          )
        ))}
        {!props.disableEditor
          && props.value
          && props.value.selection
          && props.value.selection.showEditor
          && (
            renderEditor({
              annotation: props.value,
              onChange: props.onChange,
              onSubmit: this.onSubmit
            })
          )
        }
        
        <div>{props.children}</div>
      </Container>
    );
  }
});