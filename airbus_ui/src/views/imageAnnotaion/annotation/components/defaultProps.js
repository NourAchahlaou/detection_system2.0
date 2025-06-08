import React from 'react';
import Editor from './Editor';
import FancyRectangle from './FancyRectangle';
import Rectangle from './Rectangle';
import Content from './Content';
import Overlay from './Overlay';

import {
  RectangleSelector,
} from '../selectors';

export default {
  innerRef: () => {},
  onChange: () => {},
  onSubmit: () => {},
  type: RectangleSelector.TYPE,
  selectors: [
    RectangleSelector,
  ],
  disableAnnotation: false,
  disableSelector: false,
  disableEditor: false,
  disableOverlay: false,
  activeAnnotationComparator: (a, b) => a === b,
  renderSelector: ({ annotation }) => {
    switch (annotation.geometry.type) {
      case RectangleSelector.TYPE:
        return (
          <FancyRectangle
            annotation={annotation}
            submitted={annotation.submitted} // Pass submitted state
          />
        );

      default:
        return null;
    }
  },
  renderEditor: ({ annotation, onChange, onSubmit }) => (
    <Editor
      annotation={annotation}
      onChange={onChange}
      onSubmit={onSubmit}
      style={{
        position: 'absolute',
        left: `${annotation.geometry.x}%`,
        top: `${annotation.geometry.y + annotation.geometry.height}%`,
        transform: 'translate(-50%, -100%)', // Center above the bounding box
        zIndex: 1000, 
        transition: 'left 0.5s cubic-bezier(0.4, 0, 0.2, 1), top 0.5s cubic-bezier(0.4, 0, 0.2, 1), transform 0.5s cubic-bezier(0.4, 0, 0.2, 1)', // Smoother transition effect
      }}
    />


  ),
  renderHighlight: ({ key, annotation, active }) => {
    switch (annotation.geometry.type) {
      case RectangleSelector.TYPE:
        return (
          <Rectangle
            key={key}
            annotation={annotation}
            active={active}
          />
        );

      default:
        return null;
    }
  },
  renderContent: ({ key, annotation }) => (
    <Content
      key={key}
      annotation={annotation}
    />
  ),
  renderOverlay: ({ type, annotation }) => {
    switch (type) {
      default:
        return (
          <Overlay>
            Click and Drag to Annotate
          </Overlay>
        );
    }
  },
};
