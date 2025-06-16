import React from 'react';
import Editor from './Editor';
import FancyRectangle from './FancyRectangle';
import Rectangle from './Rectangle';
import Content from './Content';
import Overlay from './Overlay';

import {
  RectangleSelector,
} from '../selectors';

const defaultProps = {
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
            submitted={annotation.submitted}
          />
        );

      default:
        return null;
    }
  },
  renderEditor: ({ annotation, onChange, onSubmit, pieceLabel }) => (
    <Editor
      annotation={annotation}
      pieceLabel={pieceLabel}
      onChange={(data) => {
        const updatedAnnotation = {
          ...annotation,
          data: {
            // FIXED: Safely handle undefined annotation.data
            ...(annotation.data || {}),
            ...data,
            text: pieceLabel || data.text,
            label: pieceLabel || data.label
          }
        };
        onChange(updatedAnnotation);
      }}
      onSubmit={() => {
        const finalAnnotation = {
          ...annotation,
          data: {
            // FIXED: Safely handle undefined annotation.data
            ...(annotation.data || {}),
            text: pieceLabel,
            label: pieceLabel,
            id: (annotation.data && annotation.data.id) || Math.random()
          }
        };
        onSubmit(finalAnnotation);
      }}
      style={{
        position: 'absolute',
        left: `${annotation.geometry.x}%`,
        top: `${annotation.geometry.y + annotation.geometry.height}%`,
        transform: 'translate(-50%, -100%)',
        zIndex: 1000,
        transition: 'all 0.5s cubic-bezier(0.4, 0, 0.2, 1)',
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

export default defaultProps;