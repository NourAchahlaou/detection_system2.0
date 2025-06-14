import React, { useState, useEffect } from 'react';
import { Box, CardMedia } from '@mui/material';
import Rectangle from './Rectangle'; // Adjust path as needed
import api from '../../../../utils/UseAxios'; // Adjust path as needed

const ImageWithAnnotations = ({ 
  imageUrl, 
  imageId, 
  width = '280px', 
  height = '200px',
  sx = {},
  onError,
  alt = "Image",
  ...otherProps 
}) => {
  const [annotations, setAnnotations] = useState([]);
  const [loading, setLoading] = useState(false);

  // Fetch annotations for this image
  useEffect(() => {
    const fetchAnnotations = async () => {
      if (!imageId) return;
      
      try {
        setLoading(true);
        const response = await api.get(`/api/annotation/annotations/image/${imageId}/annotations`);
        setAnnotations(response.data.annotations || []);
      } catch (error) {
        console.error(`Error fetching annotations for image ${imageId}:`, error);
        setAnnotations([]);
      } finally {
        setLoading(false);
      }
    };

    fetchAnnotations();
  }, [imageId]);

  return (
    <Box
      sx={{
        position: 'relative',
        width: width,
        height: height,
        overflow: 'hidden',
        ...sx
      }}
      {...otherProps}
    >
      {/* Base Image */}
      <CardMedia
        component="img"
        image={imageUrl}
        alt={alt}
        sx={{
          objectFit: 'cover',
          width: '100%',
          height: '100%',
          display: 'block',
          margin: 0,
          padding: 0,
          border: 'none',
          outline: 'none',
          verticalAlign: 'top',
          position: 'absolute',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0
        }}
        onError={onError || ((e) => {
          e.target.src = 'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjgwIiBoZWlnaHQ9IjIwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMTAwJSIgaGVpZ2h0PSIxMDAlIiBmaWxsPSIjZjVmNWY1Ii8+PHRleHQgeD0iNTAlIiB5PSI1MCUiIGZvbnQtZmFtaWx5PSJBcmlhbCwgc2Fucy1zZXJpZiIgZm9udC1zaXplPSIxNCIgZmlsbD0iIzk5OSIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZHk9Ii4zZW0iPkltYWdlIG5vdCBmb3VuZDwvdGV4dD48L3N2Zz4=';
        })}
      />

      {/* Annotation Overlays */}
      {annotations.map((annotation, index) => (
        <Rectangle
          key={`annotation-${annotation.id || index}`}
          annotation={{
            geometry: {
              x: annotation.x,
              y: annotation.y,
              width: annotation.width,
              height: annotation.height
            },
            isOverlay: true,
            active: false
          }}
        />
      ))}

      {/* Loading indicator for annotations (optional) */}
      {loading && (
        <Box
          sx={{
            position: 'absolute',
            top: 8,
            left: 8,
            backgroundColor: 'rgba(0, 0, 0, 0.7)',
            color: 'white',
            padding: '2px 6px',
            borderRadius: '4px',
            fontSize: '0.7rem',
            zIndex: 20
          }}
        >
          Loading annotations...
        </Box>
      )}

      {/* Annotation count indicator */}
      {!loading && annotations.length > 0 && (
        <Box
          sx={{
            position: 'absolute',
            top: 8,
            left: 8,
            backgroundColor: 'rgba(76, 175, 80, 0.9)',
            color: 'white',
            padding: '2px 8px',
            borderRadius: '12px',
            fontSize: '0.7rem',
            fontWeight: 'bold',
            zIndex: 20
          }}
        >
          {annotations.length} annotation{annotations.length !== 1 ? 's' : ''}
        </Box>
      )}
    </Box>
  );
};

export default ImageWithAnnotations;