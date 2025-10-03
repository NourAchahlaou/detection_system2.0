# piece_statistics_service.py - Statistics service for Artifact Keeper pieces
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from sqlalchemy.orm import Session
from sqlalchemy import func, extract, and_, or_
import logging

from artifact_keeper.app.db.models.piece import Piece

logger = logging.getLogger(__name__)

class PieceStatisticsService:
    """Service for generating piece-related statistics"""
    
    def __init__(self):
        self.logger = logger
    
    def get_pieces_added_by_period(self, db: Session, period: str = "week") -> Dict[str, Any]:
        """
        Get statistics on how many pieces were added by time period (week, day, month, year)
        
        Args:
            db: Database session
            period: Time period ('day', 'week', 'month', 'year')
        
        Returns:
            Dictionary with piece addition statistics
        """
        try:
            now = datetime.utcnow()
            
            if period == "day":
                # Last 30 days
                start_date = now - timedelta(days=30)
                date_trunc = func.date(Piece.created_at)
                date_format = "YYYY-MM-DD"
            elif period == "week":
                # Last 12 weeks
                start_date = now - timedelta(weeks=12)
                date_trunc = func.date_trunc('week', Piece.created_at)
                date_format = "YYYY-WW"
            elif period == "month":
                # Last 12 months
                start_date = now - timedelta(days=365)
                date_trunc = func.date_trunc('month', Piece.created_at)
                date_format = "YYYY-MM"
            elif period == "year":
                # Last 5 years
                start_date = now - timedelta(days=1825)
                date_trunc = func.date_trunc('year', Piece.created_at)
                date_format = "YYYY"
            else:
                raise ValueError("Period must be 'day', 'week', 'month', or 'year'")
            
            # Query pieces added by period
            results = db.query(
                date_trunc.label('period'),
                func.count(Piece.id).label('pieces_added')
            ).filter(
                Piece.created_at >= start_date
            ).group_by(
                date_trunc
            ).order_by(
                date_trunc.desc()
            ).all()
            
            # Format results
            period_stats = []
            for result in results:
                period_stats.append({
                    'period': result.period.strftime("%Y-%m-%d" if period == "day" else 
                             "%Y-W%U" if period == "week" else
                             "%Y-%m" if period == "month" else "%Y"),
                    'pieces_added': result.pieces_added,
                    'date': result.period
                })
            
            # Get total pieces added in the period
            total_pieces = sum(stat['pieces_added'] for stat in period_stats)
            
            return {
                'period': period,
                'statistics': period_stats,
                'total_pieces_in_period': total_pieces,
                'generated_at': now.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting pieces added by {period}: {e}")
            raise
    
    def get_annotation_statistics(self, db: Session) -> Dict[str, Any]:
        """
        Get statistics on annotated vs non-annotated pieces
        
        Returns:
            Dictionary with annotation statistics
        """
        try:
            # Count annotated and non-annotated pieces
            total_pieces = db.query(func.count(Piece.id)).scalar() or 0
            annotated_pieces = db.query(func.count(Piece.id)).filter(Piece.is_annotated == True).scalar() or 0
            not_annotated_pieces = total_pieces - annotated_pieces
            
            # Get annotation percentage
            annotation_percentage = (annotated_pieces / total_pieces * 100) if total_pieces > 0 else 0
            
            # Get pieces by group (extract from piece_label)
            annotation_by_group = db.query(
                func.substring(Piece.piece_label, 1, 4).label('piece_group'),
                func.count(Piece.id).label('total_pieces'),
                func.sum(func.cast(Piece.is_annotated, db.bind.dialect.name == 'postgresql' and 'integer' or 'signed')).label('annotated_pieces')
            ).group_by(
                func.substring(Piece.piece_label, 1, 4)
            ).all()
            
            group_stats = []
            for result in annotation_by_group:
                group = result.piece_group
                total = result.total_pieces
                annotated = result.annotated_pieces or 0
                not_annotated = total - annotated
                percentage = (annotated / total * 100) if total > 0 else 0
                
                group_stats.append({
                    'group': group,
                    'total_pieces': total,
                    'annotated_pieces': annotated,
                    'not_annotated_pieces': not_annotated,
                    'annotation_percentage': round(percentage, 2)
                })
            
            return {
                'overall': {
                    'total_pieces': total_pieces,
                    'annotated_pieces': annotated_pieces,
                    'not_annotated_pieces': not_annotated_pieces,
                    'annotation_percentage': round(annotation_percentage, 2)
                },
                'by_group': group_stats,
                'generated_at': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting annotation statistics: {e}")
            raise
    
    def get_training_statistics(self, db: Session) -> Dict[str, Any]:
        """
        Get statistics on trained vs non-trained pieces
        
        Returns:
            Dictionary with training statistics
        """
        try:
            # Count trained and non-trained pieces
            total_pieces = db.query(func.count(Piece.id)).scalar() or 0
            trained_pieces = db.query(func.count(Piece.id)).filter(Piece.is_yolo_trained == True).scalar() or 0
            not_trained_pieces = total_pieces - trained_pieces
            
            # Get training percentage
            training_percentage = (trained_pieces / total_pieces * 100) if total_pieces > 0 else 0
            
            # Get pieces by group (extract from piece_label)
            training_by_group = db.query(
                func.substring(Piece.piece_label, 1, 4).label('piece_group'),
                func.count(Piece.id).label('total_pieces'),
                func.sum(func.cast(Piece.is_yolo_trained, db.bind.dialect.name == 'postgresql' and 'integer' or 'signed')).label('trained_pieces')
            ).group_by(
                func.substring(Piece.piece_label, 1, 4)
            ).all()
            
            group_stats = []
            for result in training_by_group:
                group = result.piece_group
                total = result.total_pieces
                trained = result.trained_pieces or 0
                not_trained = total - trained
                percentage = (trained / total * 100) if total > 0 else 0
                
                group_stats.append({
                    'group': group,
                    'total_pieces': total,
                    'trained_pieces': trained,
                    'not_trained_pieces': not_trained,
                    'training_percentage': round(percentage, 2)
                })
            
            return {
                'overall': {
                    'total_pieces': total_pieces,
                    'trained_pieces': trained_pieces,
                    'not_trained_pieces': not_trained_pieces,
                    'training_percentage': round(training_percentage, 2)
                },
                'by_group': group_stats,
                'generated_at': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting training statistics: {e}")
            raise
    
    def get_pieces_with_images_statistics(self, db: Session) -> Dict[str, Any]:
        """
        Get statistics on pieces with/without images
        
        Returns:
            Dictionary with image statistics
        """
        try:
            # Count pieces with and without images
            total_pieces = db.query(func.count(Piece.id)).scalar() or 0
            pieces_with_images = db.query(func.count(Piece.id)).filter(
                and_(Piece.nbre_img.isnot(None), Piece.nbre_img > 0)
            ).scalar() or 0
            pieces_without_images = total_pieces - pieces_with_images
            
            # Get average number of images per piece
            avg_images = db.query(func.avg(Piece.nbre_img)).filter(
                and_(Piece.nbre_img.isnot(None), Piece.nbre_img > 0)
            ).scalar() or 0
            
            # Get total number of images
            total_images = db.query(func.sum(Piece.nbre_img)).scalar() or 0
            
            # Get image statistics by group
            image_by_group = db.query(
                func.substring(Piece.piece_label, 1, 4).label('piece_group'),
                func.count(Piece.id).label('total_pieces'),
                func.count(func.nullif(Piece.nbre_img, 0)).label('pieces_with_images'),
                func.sum(func.coalesce(Piece.nbre_img, 0)).label('total_images'),
                func.avg(func.nullif(Piece.nbre_img, 0)).label('avg_images')
            ).group_by(
                func.substring(Piece.piece_label, 1, 4)
            ).all()
            
            group_stats = []
            for result in image_by_group:
                group = result.piece_group
                total = result.total_pieces
                with_images = result.pieces_with_images or 0
                without_images = total - with_images
                total_imgs = result.total_images or 0
                avg_imgs = float(result.avg_images or 0)
                
                group_stats.append({
                    'group': group,
                    'total_pieces': total,
                    'pieces_with_images': with_images,
                    'pieces_without_images': without_images,
                    'total_images': total_imgs,
                    'avg_images_per_piece': round(avg_imgs, 2)
                })
            
            return {
                'overall': {
                    'total_pieces': total_pieces,
                    'pieces_with_images': pieces_with_images,
                    'pieces_without_images': pieces_without_images,
                    'total_images': total_images,
                    'avg_images_per_piece': round(float(avg_images), 2)
                },
                'by_group': group_stats,
                'generated_at': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting image statistics: {e}")
            raise
    
    def get_comprehensive_piece_statistics(self, db: Session) -> Dict[str, Any]:
        """
        Get comprehensive piece statistics combining all metrics
        
        Returns:
            Dictionary with comprehensive statistics
        """
        try:
            # Get all statistics
            annotation_stats = self.get_annotation_statistics(db)
            training_stats = self.get_training_statistics(db)
            image_stats = self.get_pieces_with_images_statistics(db)
            
            # Get pieces added in the last week/month
            weekly_additions = self.get_pieces_added_by_period(db, "week")
            monthly_additions = self.get_pieces_added_by_period(db, "month")
            
            # Combine group statistics
            all_groups = set()
            group_data = {}
            
            # Collect all groups
            for stat in [annotation_stats['by_group'], training_stats['by_group'], image_stats['by_group']]:
                for group_stat in stat:
                    group = group_stat['group']
                    all_groups.add(group)
                    if group not in group_data:
                        group_data[group] = {}
            
            # Merge group statistics
            for group in all_groups:
                # Find annotation data for this group
                ann_data = next((g for g in annotation_stats['by_group'] if g['group'] == group), {})
                train_data = next((g for g in training_stats['by_group'] if g['group'] == group), {})
                img_data = next((g for g in image_stats['by_group'] if g['group'] == group), {})
                
                group_data[group] = {
                    'group': group,
                    'total_pieces': ann_data.get('total_pieces', 0),
                    'annotated_pieces': ann_data.get('annotated_pieces', 0),
                    'not_annotated_pieces': ann_data.get('not_annotated_pieces', 0),
                    'annotation_percentage': ann_data.get('annotation_percentage', 0),
                    'trained_pieces': train_data.get('trained_pieces', 0),
                    'not_trained_pieces': train_data.get('not_trained_pieces', 0),
                    'training_percentage': train_data.get('training_percentage', 0),
                    'pieces_with_images': img_data.get('pieces_with_images', 0),
                    'pieces_without_images': img_data.get('pieces_without_images', 0),
                    'total_images': img_data.get('total_images', 0),
                    'avg_images_per_piece': img_data.get('avg_images_per_piece', 0)
                }
            
            return {
                'overall_summary': {
                    'total_pieces': annotation_stats['overall']['total_pieces'],
                    'annotated_pieces': annotation_stats['overall']['annotated_pieces'],
                    'not_annotated_pieces': annotation_stats['overall']['not_annotated_pieces'],
                    'annotation_percentage': annotation_stats['overall']['annotation_percentage'],
                    'trained_pieces': training_stats['overall']['trained_pieces'],
                    'not_trained_pieces': training_stats['overall']['not_trained_pieces'],
                    'training_percentage': training_stats['overall']['training_percentage'],
                    'pieces_with_images': image_stats['overall']['pieces_with_images'],
                    'pieces_without_images': image_stats['overall']['pieces_without_images'],
                    'total_images': image_stats['overall']['total_images'],
                    'avg_images_per_piece': image_stats['overall']['avg_images_per_piece']
                },
                'by_group': list(group_data.values()),
                'time_series': {
                    'weekly_additions': weekly_additions['statistics'][:8],  # Last 8 weeks
                    'monthly_additions': monthly_additions['statistics'][:6]  # Last 6 months
                },
                'generated_at': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting comprehensive statistics: {e}")
            raise

# Global instance
piece_statistics_service = PieceStatisticsService()