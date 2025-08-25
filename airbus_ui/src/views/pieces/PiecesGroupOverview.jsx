import React, { useState, useEffect } from "react";
import { 
  Box, 
  Typography, 
  styled, 
  Card,
  Chip,
  CircularProgress,
  Button,
  Collapse,
  IconButton,
  Pagination
} from "@mui/material";
import { 
  CropFree, 
  PhotoLibrary, 
  Visibility,
  ExpandMore,
  ExpandLess,
  FolderOpen,
  CheckCircle,
  RadioButtonUnchecked,
  Edit
} from "@mui/icons-material";
import { useNavigate } from "react-router-dom";
import api from "../../utils/UseAxios";

// STYLED COMPONENTS - Matching the second file exactly
const Container = styled("div")(({ theme }) => ({
  margin: "30px",
  [theme.breakpoints.down("sm")]: { margin: "16px" },
}));

const HeaderBox = styled(Box)({
  paddingBottom: "24px",
  borderBottom: "1px solid rgba(102, 126, 234, 0.1)",
  marginBottom: "24px",
  textAlign: "center",
});

const GroupSection = styled(Box)(({ theme }) => ({
  marginBottom: "32px",
  border: "1px solid rgba(102, 126, 234, 0.15)",
  borderRadius: "16px",
  overflow: "hidden",
  backgroundColor: "#fff",
  boxShadow: "0 4px 16px rgba(0, 0, 0, 0.08)",
}));

const GroupHeader = styled(Box)(({ theme }) => ({
  padding: "20px 24px",
  backgroundColor: "rgba(102, 126, 234, 0.05)",
  borderBottom: "1px solid rgba(102, 126, 234, 0.1)",
  display: "flex",
  alignItems: "center",
  justifyContent: "space-between",
  cursor: "pointer",
  transition: "all 0.3s ease",
  "&:hover": {
    backgroundColor: "rgba(102, 126, 234, 0.1)",
  },
}));

const GroupTitle = styled(Typography)({
  fontSize: "1.3rem",
  fontWeight: "700",
  color: "#333",
  display: "flex",
  alignItems: "center",
  gap: "12px",
});

const GroupStats = styled(Box)({
  display: "flex",
  alignItems: "center",
  gap: "16px",
});

const GroupContent = styled(Box)({
  padding: "24px",
});

// Pagination controls for groups
const GroupPagination = styled(Box)({
  display: "flex",
  justifyContent: "space-between",
  alignItems: "center",
  padding: "16px 24px",
  backgroundColor: "rgba(102, 126, 234, 0.03)",
  borderTop: "1px solid rgba(102, 126, 234, 0.1)",
});

// EXACT SAME GRID as the second file - FIXED: Simple CSS Grid with exactly 4 columns and auto rows
const CardsGridContainer = styled('div')(({ theme }) => ({
  display: 'grid',
  gridTemplateColumns: 'repeat(4, 1fr)', // Always 4 columns
  gap: '20px',
  width: '100%',
  
  // Responsive adjustments only for smaller screens
  [theme.breakpoints.down('lg')]: {
    gridTemplateColumns: 'repeat(3, 1fr)', // 3 columns for medium screens
  },
  
  [theme.breakpoints.down('md')]: {
    gridTemplateColumns: 'repeat(2, 1fr)', // 2 columns for small screens
  },
  
  [theme.breakpoints.down('sm')]: {
    gridTemplateColumns: '1fr', // 1 column for mobile
  },
}));

// EXACT SAME CARD STYLING as the second file
const PieceCard = styled(Card)(({ theme }) => ({
  padding: "20px",
  cursor: "pointer",
  height: "100%",
  display: "flex",
  flexDirection: "column",
  border: "2px solid rgba(102, 126, 234, 0.1)",
  borderRadius: "16px",
  transition: "all 0.3s cubic-bezier(0.4, 0, 0.2, 1)",
  position: "relative",
  overflow: "hidden",
  boxShadow: "0 2px 12px rgba(0, 0, 0, 0.08)",
  minWidth: "0", // Important for text truncation
  "&:hover": {
    transform: "translateY(-4px)",
    boxShadow: "0 12px 32px rgba(102, 126, 234, 0.2)",
    border: "2px solid #667eea",
    backgroundColor: "rgba(78, 105, 221, 0.45)",
  },
}));

// EXACT SAME STYLING as the second file
const CardHeader = styled(Box)({
  display: "flex",
  alignItems: "flex-start",
  justifyContent: "space-between",
  marginBottom: "16px",
  gap: "12px",
});

const IconContainer = styled(Box)({
  width: "48px",
  height: "48px",
  borderRadius: "12px",
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  backgroundColor: "rgba(102, 126, 234, 0.15)",
  color: "#667eea",
  marginBottom: "12px",
});

const PieceTitle = styled(Typography)({
  fontSize: "1.1rem",
  fontWeight: "700",
  color: "#333",
  marginBottom: "8px",
  overflow: "hidden",
  textOverflow: "ellipsis",
  whiteSpace: "nowrap",
  "&:hover": {
    color: "#e2e2e2",
  },
});

const StatsContainer = styled(Box)({
  display: "flex",
  flexDirection: "column",
  gap: "8px",
  marginTop: "auto",
});

const StatsRow = styled(Box)({
  display: "flex",
  alignItems: "center",
  justifyContent: "space-between",
});

const ImagePreview = styled("img")({
  width: "40px",
  height: "40px",
  objectFit: "cover",
  borderRadius: "6px",
  border: "2px solid rgba(102, 126, 234, 0.2)",
});

// EXACT SAME STATUS CHIP as the second file
const StatusChip = styled(Chip)(({ variant }) => ({
  fontSize: "0.75rem",
  fontWeight: "600",
  height: "24px",
  backgroundColor: variant === 'completed' 
    ? "rgba(76, 175, 80, 0.15)" 
    : variant === 'partial'
    ? "rgba(255, 152, 0, 0.15)"
    : "rgba(244, 67, 54, 0.15)",
  color: variant === 'completed' 
    ? "#4caf50" 
    : variant === 'partial'
    ? "#ff9800"
    : "#f44336",
  "& .MuiChip-icon": {
    fontSize: "14px",
  },
}));

// EXACT SAME ACTION BUTTON as the second file
const ActionButton = styled(Button)(({ variant }) => ({
  textTransform: "none",
  fontWeight: "600",
  borderRadius: "6px",
  fontSize: "0.8rem",
  padding: "4px 12px",
  minWidth: "auto",
  backgroundColor: variant === 'primary' ? "#667eea" : "transparent",
  color: variant === 'primary' ? "white" : "#667eea",
  border: variant === 'primary' ? "none" : "1px solid rgba(102, 126, 234, 0.3)",
  "&:hover": {
    backgroundColor: variant === 'primary' ? "#5a67d8" : "rgba(102, 126, 234, 0.08)",
  },
}));

const LoadingContainer = styled(Box)({
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  minHeight: "300px",
  flexDirection: "column",
  gap: 2,
  color: "#666",
});

const EmptyState = styled(Box)({
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  minHeight: "300px",
  flexDirection: "column",
  gap: 2,
  color: "#666",
  textAlign: "center",
});

// Constants for pagination
const PIECES_PER_PAGE = 12;
const GROUPS_PER_PAGE = 5;

export default function PiecesGroupOverview() {
  const [allPieces, setAllPieces] = useState([]);
  const [groupedPieces, setGroupedPieces] = useState({});
  const [loading, setLoading] = useState(true);
  const [expandedGroups, setExpandedGroups] = useState({});
  const [groupPages, setGroupPages] = useState({});
  const [currentGroupPage, setCurrentGroupPage] = useState(1);
  const [totalGroups, setTotalGroups] = useState(0);
  const [stats, setStats] = useState({
    totalPieces: 0,
    totalGroups: 0,
    totalImages: 0,
    annotated: 0,
    partial: 0,
    nonAnnotated: 0
  });
  
  const navigate = useNavigate();

  useEffect(() => {
    fetchGroupsAndPieces();
  }, []);

  const fetchGroupsAndPieces = async () => {
    try {
      setLoading(true);
      
      let groupsData = [];
      let piecesData = [];
      
      // Fetch groups from the new backend API
      try {
        console.log("Attempting to fetch groups from:", "/api/artifact_keeper/datasetManager/groups");
        const groupsResponse = await api.get("/api/artifact_keeper/datasetManager/groups");
        console.log("Groups API response:", groupsResponse);
        console.log("Groups response data:", groupsResponse.data);
        
        groupsData = groupsResponse.data?.groups || [];
        console.log("Extracted groups data:", groupsData);
        
        if (!groupsData || groupsData.length === 0) {
          console.warn("No groups found in API response, groupsData:", groupsData);
        }
        
      } catch (groupsError) {
        console.error("Error fetching groups from API:", groupsError);
        console.error("Groups API error details:", {
          message: groupsError.message,
          response: groupsError.response?.data,
          status: groupsError.response?.status
        });
        // Continue without groups - will create fallback grouping
      }
      
      // Fetch all pieces data
      try {
        console.log("Attempting to fetch pieces from:", "/api/annotation/annotations/get_all_pieces");
        const piecesResponse = await api.get("/api/annotation/annotations/get_all_pieces");
        piecesData = piecesResponse.data || [];
        console.log("Fetched pieces data:", piecesData);
        
      } catch (piecesError) {
        console.error("Error fetching pieces:", piecesError);
        throw piecesError; // Re-throw to trigger fallback logic
      }
      
      setAllPieces(piecesData);
      
      // Group pieces by the backend-provided groups
      console.log("Grouping pieces. Groups available:", groupsData.length);
      const grouped = groupPiecesByBackendGroups(piecesData, groupsData);
      console.log("Grouped pieces result:", grouped);
      setGroupedPieces(grouped);
      
      // Initialize pagination for each group
      const initialPages = {};
      Object.keys(grouped).forEach(groupName => {
        initialPages[groupName] = 1;
      });
      setGroupPages(initialPages);
      
      // Calculate stats - EXACT SAME as second file
      const totalPieces = piecesData.length;
      const fullyAnnotatedPieces = piecesData.filter(piece => piece.is_fully_annotated).length;
      const partiallyAnnotatedPieces = piecesData.filter(piece => 
        piece.annotated_count > 0 && !piece.is_fully_annotated
      ).length;
      const notStartedPieces = piecesData.filter(piece => piece.annotated_count === 0).length;
      const totalImages = piecesData.reduce((sum, piece) => sum + (piece.nbr_img || 0), 0);
      
      setStats({
        totalPieces: totalPieces,
        totalGroups: Object.keys(grouped).length,
        totalImages: totalImages,
        annotated: fullyAnnotatedPieces,
        partial: partiallyAnnotatedPieces,
        nonAnnotated: notStartedPieces
      });
      
      setTotalGroups(Object.keys(grouped).length);
      
    } catch (error) {
      console.error("Error in fetchGroupsAndPieces:", error);
      
      // Fallback logic - try to fetch pieces without groups
      try {
        console.log("Trying fallback pieces endpoint...");
        const fallbackResponse = await api.get("/api/annotation/annotations/get_Img_nonAnnotated");
        const nonAnnotatedData = fallbackResponse.data || [];
        
        const convertedData = nonAnnotatedData.map(piece => ({
          piece_label: piece.piece_label,
          nbr_img: piece.nbr_img,
          annotated_count: piece.annotated_count || 0,
          url: piece.url,
          is_fully_annotated: false
        }));
        
        setAllPieces(convertedData);
        
        // Create a simple "Other" group for fallback
        const grouped = { "Other": convertedData };
        setGroupedPieces(grouped);
        
        const partiallyAnnotated = convertedData.filter(piece => piece.annotated_count > 0).length;
        const totalImages = convertedData.reduce((sum, piece) => sum + (piece.nbr_img || 0), 0);
        
        setStats({
          totalPieces: convertedData.length,
          totalGroups: 1,
          totalImages: totalImages,
          annotated: 0,
          partial: partiallyAnnotated,
          nonAnnotated: convertedData.length - partiallyAnnotated
        });
        
      } catch (fallbackError) {
        console.error("Fallback fetch also failed:", fallbackError);
        setAllPieces([]);
        setGroupedPieces({});
        setStats({ totalPieces: 0, totalGroups: 0, totalImages: 0, annotated: 0, partial: 0, nonAnnotated: 0 });
      }
    } finally {
      setLoading(false);
    }
  };

  // New function to group pieces based on backend-provided groups
  const groupPiecesByBackendGroups = (pieces, groups) => {
    console.log("groupPiecesByBackendGroups called with:", { pieces: pieces.length, groups });
    
    const grouped = {};
    
    // If no groups from backend, create groups based on piece patterns as fallback
    if (!groups || groups.length === 0) {
      console.log("No backend groups available, creating fallback groups based on piece patterns");
      
      pieces.forEach((piece) => {
        const pieceLabel = piece.piece_label || "";
        // Extract first part before first dot as group name (G053, E539, H123, etc.)
        const groupName = pieceLabel.split('.')[0] || 'Other';
        
        if (!grouped[groupName]) {
          grouped[groupName] = [];
        }
        grouped[groupName].push(piece);
      });
      
      console.log("Fallback groups created:", Object.keys(grouped));
      return grouped;
    }
    
    // Initialize groups from backend
    groups.forEach(groupName => {
      grouped[groupName] = [];
    });
    
    // Add an "Other" group for pieces that don't match any backend group
    grouped["Other"] = [];
    
    // Assign pieces to groups
    pieces.forEach((piece) => {
      const pieceLabel = piece.piece_label || "";
      let assigned = false;
      
      // Check if piece belongs to any of the backend groups
      for (const groupName of groups) {
        // Check if piece label starts with the group name or contains it
        if (pieceLabel.toUpperCase().startsWith(groupName.toUpperCase()) || 
            pieceLabel.toUpperCase().includes(groupName.toUpperCase())) {
          grouped[groupName].push(piece);
          assigned = true;
          console.log(`Assigned piece ${pieceLabel} to group ${groupName}`);
          break;
        }
      }
      
      // If not assigned to any group, add to "Other"
      if (!assigned) {
        grouped["Other"].push(piece);
        console.log(`Assigned piece ${pieceLabel} to Other group`);
      }
    });
    
    // Remove empty groups (including "Other" if empty)
    Object.keys(grouped).forEach(groupName => {
      if (grouped[groupName].length === 0) {
        console.log(`Removing empty group: ${groupName}`);
        delete grouped[groupName];
      }
    });
    
    console.log("Final grouped result:", Object.keys(grouped).map(key => ({ [key]: grouped[key].length })));
    return grouped;
  };

  // EXACT SAME STATUS FUNCTION as second file
  const getStatusInfo = (piece) => {
    if (piece.annotated_count === 0) {
      return {
        variant: 'not-started',
        label: 'Not Started',
        icon: <RadioButtonUnchecked />,
        progress: 0
      };
    } else if (piece.annotated_count < piece.nbr_img) {
      return {
        variant: 'partial',
        label: `${piece.annotated_count}/${piece.nbr_img} Done`,
        icon: <Edit />,
        progress: (piece.annotated_count / piece.nbr_img) * 100
      };
    } else {
      return {
        variant: 'completed',
        label: 'Completed',
        icon: <CheckCircle />,
        progress: 100
      };
    }
  };

  const handleGroupToggle = (groupName) => {
    setExpandedGroups(prev => ({
      ...prev,
      [groupName]: !prev[groupName]
    }));
  };

  const handlePieceClick = (pieceLabel) => {
    navigate(`/annotation?piece=${encodeURIComponent(pieceLabel)}`);
  };

  const handleGroupPageChange = (groupName, page) => {
    setGroupPages(prev => ({
      ...prev,
      [groupName]: page
    }));
  };

  // EXACT SAME RENDER FUNCTION as second file but with piece data
  const renderPieceCard = (piece) => {
    const statusInfo = getStatusInfo(piece);
    
    return (
      <PieceCard key={piece.piece_label} elevation={0} onClick={() => handlePieceClick(piece.piece_label)}>
        <CardHeader>
          <Box sx={{ display: "flex", alignItems: "center", gap: 1, flex: 1, minWidth: 0 }}>
            <IconContainer>
              <CropFree fontSize="medium" />
            </IconContainer>
            <Box sx={{ minWidth: 0, flex: 1 }}>
              <PieceTitle title={piece.piece_label}>{piece.piece_label}</PieceTitle>
              <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                <PhotoLibrary sx={{ fontSize: 14, color: "#667eea" }} />
                <Typography variant="caption" sx={{ color: "#666", fontWeight: "500" }}>
                  {piece.nbr_img} images
                </Typography>
              </Box>
            </Box>
          </Box>
          
          {piece.url && (
            <ImagePreview
              src={piece.url}
              alt={piece.piece_label}
              onError={(e) => {
                console.log(`Failed to load image: ${piece.url}`);
                e.target.style.display = 'none';
              }}
              onLoad={() => {
                console.log(`Successfully loaded image: ${piece.url}`);
              }}
            />
          )}
        </CardHeader>
        
        <StatsContainer>
          <StatsRow>
            <StatusChip
              variant={statusInfo.variant}
              icon={statusInfo.icon}
              label={statusInfo.label}
              size="small"
            />
            
            <ActionButton
              variant={statusInfo.variant === 'completed' ? 'secondary' : 'primary'}
              size="small"
              startIcon={statusInfo.variant === 'completed' ? <Visibility /> : <Edit />}
            >
              {statusInfo.variant === 'completed' ? 'View' : 'Annotate'}
            </ActionButton>
          </StatsRow>
        </StatsContainer>
      </PieceCard>
    );
  };

  const renderGroup = (groupName, pieces) => {
    const isExpanded = expandedGroups[groupName] || false;
    const currentPage = groupPages[groupName] || 1;
    const totalPages = Math.ceil(pieces.length / PIECES_PER_PAGE);
    const startIndex = (currentPage - 1) * PIECES_PER_PAGE;
    const paginatedPieces = pieces.slice(startIndex, startIndex + PIECES_PER_PAGE);
    const totalImages = pieces.reduce((sum, piece) => sum + (piece.nbr_img || 0), 0);
    
    // Calculate group stats
    const fullyAnnotated = pieces.filter(piece => piece.is_fully_annotated).length;
    const partiallyAnnotated = pieces.filter(piece => 
      piece.annotated_count > 0 && !piece.is_fully_annotated
    ).length;
    const notStarted = pieces.filter(piece => piece.annotated_count === 0).length;

    return (
      <GroupSection key={groupName}>
        <GroupHeader onClick={() => handleGroupToggle(groupName)}>
          <GroupTitle>
            <FolderOpen />
            {groupName}
          </GroupTitle>
          
          <GroupStats>
            <Chip
              label={`${pieces.length} pieces`}
              size="small"
              sx={{ 
                backgroundColor: "rgba(102, 126, 234, 0.15)", 
                color: "#667eea",
                fontWeight: "600"
              }}
            />
            <Chip
              label={`${totalImages} images`}
              size="small"
              sx={{ 
                backgroundColor: "rgba(102, 126, 234, 0.15)", 
                color: "#667eea",
                fontWeight: "600"
              }}
            />
            <Chip
              label={`${fullyAnnotated} completed`}
              size="small"
              sx={{ 
                backgroundColor: "rgba(76, 175, 80, 0.15)", 
                color: "#4caf50",
                fontWeight: "600"
              }}
            />
            <IconButton size="small">
              {isExpanded ? <ExpandLess /> : <ExpandMore />}
            </IconButton>
          </GroupStats>
        </GroupHeader>
        
        <Collapse in={isExpanded}>
          <GroupContent>
            <CardsGridContainer>
              {paginatedPieces.map(renderPieceCard)}
            </CardsGridContainer>
            
            {totalPages > 1 && (
              <GroupPagination>
                <Typography variant="body2" sx={{ color: "#666" }}>
                  Showing {startIndex + 1}-{Math.min(startIndex + PIECES_PER_PAGE, pieces.length)} of {pieces.length} pieces
                </Typography>
                <Pagination
                  count={totalPages}
                  page={currentPage}
                  onChange={(event, page) => handleGroupPageChange(groupName, page)}
                  size="small"
                  color="primary"
                />
              </GroupPagination>
            )}
          </GroupContent>
        </Collapse>
      </GroupSection>
    );
  };

  if (loading) {
    return (
      <Container>
        <LoadingContainer>
          <CircularProgress sx={{ color: '#667eea' }} size={48} />
          <Typography variant="h6" sx={{ opacity: 0.8, mt: 2 }}>
            Loading groups and pieces...
          </Typography>
        </LoadingContainer>
      </Container>
    );
  }

  // Paginate groups
  const groupNames = Object.keys(groupedPieces);
  const totalGroupPages = Math.ceil(groupNames.length / GROUPS_PER_PAGE);
  const startGroupIndex = (currentGroupPage - 1) * GROUPS_PER_PAGE;
  const currentGroups = groupNames.slice(startGroupIndex, startGroupIndex + GROUPS_PER_PAGE);

  return (
    <Container>
      {/* EXACT SAME HEADER as second file but with group stats */}
      <HeaderBox>       
        <Box sx={{ 
          display: 'flex', 
          gap: 3, 
          mt: 3, 
          flexWrap: 'wrap',
          justifyContent: 'center'
        }}>
          <Box sx={{ textAlign: 'center' }}>
            <Typography variant="h4" sx={{ color: '#667eea', fontWeight: '700' }}>
              {stats.totalGroups}
            </Typography>
            <Typography variant="caption" sx={{ color: '#666', textTransform: 'uppercase', letterSpacing: 1 }}>
              Groups
            </Typography>
          </Box>
          <Box sx={{ textAlign: 'center' }}>
            <Typography variant="h4" sx={{ color: '#667eea', fontWeight: '700' }}>
              {stats.totalPieces}
            </Typography>
            <Typography variant="caption" sx={{ color: '#666', textTransform: 'uppercase', letterSpacing: 1 }}>
              Total Pieces
            </Typography>
          </Box>
          <Box sx={{ textAlign: 'center' }}>
            <Typography variant="h4" sx={{ color: '#4caf50', fontWeight: '700' }}>
              {stats.annotated}
            </Typography>
            <Typography variant="caption" sx={{ color: '#666', textTransform: 'uppercase', letterSpacing: 1 }}>
              Completed
            </Typography>
          </Box>
          <Box sx={{ textAlign: 'center' }}>
            <Typography variant="h4" sx={{ color: '#ff9800', fontWeight: '700' }}>
              {stats.partial}
            </Typography>
            <Typography variant="caption" sx={{ color: '#666', textTransform: 'uppercase', letterSpacing: 1 }}>
              In Progress
            </Typography>
          </Box>
          <Box sx={{ textAlign: 'center' }}>
            <Typography variant="h4" sx={{ color: '#f44336', fontWeight: '700' }}>
              {stats.nonAnnotated}
            </Typography>
            <Typography variant="caption" sx={{ color: '#666', textTransform: 'uppercase', letterSpacing: 1 }}>
              Not Started
            </Typography>
          </Box>
          <Box sx={{ textAlign: 'center' }}>
            <Typography variant="h4" sx={{ color: '#9c27b0', fontWeight: '700' }}>
              {stats.totalImages}
            </Typography>
            <Typography variant="caption" sx={{ color: '#666', textTransform: 'uppercase', letterSpacing: 1 }}>
              Total Images
            </Typography>
          </Box>
        </Box>
      </HeaderBox>

      {/* Groups Display */}
      {currentGroups.length > 0 ? (
        <>
          {currentGroups.map(groupName => 
            renderGroup(groupName, groupedPieces[groupName])
          )}
          
          {/* Main Groups Pagination */}
          {totalGroupPages > 1 && (
            <Box sx={{ 
              display: 'flex', 
              justifyContent: 'center', 
              mt: 4,
              p: 2,
              backgroundColor: 'rgba(102, 126, 234, 0.05)',
              borderRadius: '12px'
            }}>
              <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 2 }}>
                <Typography variant="body2" sx={{ color: "#666" }}>
                  Showing groups {startGroupIndex + 1}-{Math.min(startGroupIndex + GROUPS_PER_PAGE, groupNames.length)} of {groupNames.length}
                </Typography>
                <Pagination
                  count={totalGroupPages}
                  page={currentGroupPage}
                  onChange={(event, page) => setCurrentGroupPage(page)}
                  color="primary"
                />
              </Box>
            </Box>
          )}
        </>
      ) : (
        <EmptyState>
          <FolderOpen sx={{ fontSize: 64, opacity: 0.4, mb: 2 }} />
          <Typography variant="h6" sx={{ opacity: 0.9, mb: 1 }}>
            No Groups Found
          </Typography>
          <Typography variant="body2" sx={{ opacity: 0.7 }}>
            No groups are available in the system
          </Typography>
        </EmptyState>
      )}
    </Container>
  );
}