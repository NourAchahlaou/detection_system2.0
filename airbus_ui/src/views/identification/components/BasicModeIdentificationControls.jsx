// components/BasicModeIdentificationControls.jsx - Fixed version for identification
// This component handles identification controls without needing a targetLabel prop
import React from 'react';
import {
  Box,
  Stack,
  Alert,
  CircularProgress,
  Card,
  CardContent,
  Typography,
  Chip,
  Button,
  ButtonGroup
} from '@mui/material';
import {
  AcUnit, // Freeze icon
  Whatshot, // Unfreeze icon
  PlayArrow,
  Visibility // Piece identification icon
} from '@mui/icons-material';

const BasicModeIdentificationControls = ({
  isStreamFrozen,
  identificationInProgress,
  lastIdentificationResult,
  confidenceThreshold,
  onPieceIdentification,
  onFreezeStream,
  onUnfreezeStream
}) => {
  return (
    <Card>
      <CardContent sx={{ py: 2 }}>
        <Typography variant="h6" gutterBottom color="primary">
          Identification Controls
        </Typography>
        
        {/* Stream Status */}
        <Box sx={{ mb: 2 }}>
          <Stack direction="row" spacing={1} alignItems="center" sx={{ mb: 1 }}>
            <Chip 
              label={isStreamFrozen ? "FROZEN" : "LIVE"} 
              size="small" 
              color={isStreamFrozen ? "warning" : "success"}
              icon={isStreamFrozen ? <AcUnit /> : <PlayArrow />}
            />
            <Typography variant="caption" color="textSecondary">
              Stream Status
            </Typography>
          </Stack>
        </Box>

        {/* Control Buttons */}
        <Stack spacing={1}>
          {/* Piece Identification Button */}
          <Button
            variant="contained"
            size="small"
            fullWidth
            startIcon={<Visibility />}
            onClick={onPieceIdentification}
            disabled={identificationInProgress}
            color="primary"
          >
            {identificationInProgress ? 'Identifying...' : 'Identify Pieces'}
          </Button>

          {/* Freeze/Unfreeze Controls */}
          <ButtonGroup size="small" variant="outlined" fullWidth>
            <Button
              onClick={onFreezeStream}
              disabled={isStreamFrozen || identificationInProgress}
              startIcon={<AcUnit />}
              sx={{ flex: 1 }}
            >
              Freeze
            </Button>
            <Button
              onClick={onUnfreezeStream}
              disabled={!isStreamFrozen || identificationInProgress}
              startIcon={<Whatshot />}
              sx={{ flex: 1 }}
            >
              Unfreeze
            </Button>
          </ButtonGroup>
        </Stack>

        {/* Last Identification Result */}
        {lastIdentificationResult && (
          <Box sx={{ mt: 2, p: 1.5, bgcolor: 'background.paper', borderRadius: 1, border: '1px solid', borderColor: 'divider' }}>
            <Typography variant="subtitle2" gutterBottom>
              Last Identification Result
            </Typography>
            <Stack spacing={0.5}>
              {/* Handle both piece identification and quick analysis results */}
              {lastIdentificationResult.summary ? (
                <>
                  <Typography variant="body2" color={lastIdentificationResult.summary.total_pieces > 0 ? 'success.main' : 'text.secondary'}>
                    {lastIdentificationResult.summary.total_pieces > 0 ? 
                      `✅ ${lastIdentificationResult.summary.total_pieces} PIECES FOUND` : 
                      '❌ NO PIECES FOUND'}
                  </Typography>
                  {lastIdentificationResult.summary.total_pieces > 0 && (
                    <Typography variant="caption" color="success.main">
                      Unique Labels: {lastIdentificationResult.summary.unique_labels}
                    </Typography>
                  )}
                </>
              ) : lastIdentificationResult.piecesFound !== undefined ? (
                <Typography variant="body2" color={lastIdentificationResult.piecesFound > 0 ? 'success.main' : 'text.secondary'}>
                  {lastIdentificationResult.piecesFound > 0 ? 
                    `✅ ${lastIdentificationResult.piecesFound} PIECES FOUND` : 
                    '❌ NO PIECES FOUND'}
                </Typography>
              ) : (
                <Typography variant="body2" color="text.secondary">
                  Analysis completed
                </Typography>
              )}
              
              {lastIdentificationResult.processingTime && (
                <Typography variant="caption" color="textSecondary">
                  Processing Time: {lastIdentificationResult.processingTime}ms
                </Typography>
              )}
              
              {lastIdentificationResult.isQuickAnalysis && (
                <Chip 
                  label="Quick Analysis" 
                  size="small" 
                  color="secondary" 
                  variant="outlined"
                  sx={{ alignSelf: 'flex-start', mt: 0.5 }}
                />
              )}
            </Stack>
          </Box>
        )}

        {/* Stream Frozen Alert */}
        {isStreamFrozen && (
          <Alert severity="info" sx={{ mt: 2 }}>
            <Typography variant="body2">
              Stream is frozen for analysis. Use controls above to unfreeze or perform identification.
            </Typography>
          </Alert>
        )}

        {/* Identification in Progress */}
        {identificationInProgress && (
          <Alert 
            severity="info" 
            sx={{ mt: 2 }}
            icon={<CircularProgress size={16} />}
          >
            <Typography variant="body2">
              Performing identification analysis...
            </Typography>
          </Alert>
        )}

        {/* Helper Text */}
        {!isStreamFrozen && !identificationInProgress && (
          <Box sx={{ mt: 2, p: 1, bgcolor: 'action.hover', borderRadius: 1 }}>
            <Typography variant="caption" color="textSecondary">
              💡 Tip: Use "Identify Pieces" for comprehensive analysis or "Quick Analysis" for faster results.
            </Typography>
          </Box>
        )}
      </CardContent>
    </Card>
  );
};

export default BasicModeIdentificationControls;