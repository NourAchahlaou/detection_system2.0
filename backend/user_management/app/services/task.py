"""
Task Management Module

This module handles all aspects of task creation, assignment, and management
for the user management application.
"""

# task.py - Service functions for task management
from datetime import datetime
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc
from fastapi import HTTPException

from user_management.app.db.models.user import User
from user_management.app.db.models.task import Task
from user_management.app.db.models.actionType import ActionType
from user_management.app.services.activity import log_activity

class TaskManagementError(Exception):
    """Exception raised for errors in the task management process."""
    pass

async def create_task(
    title: str, 
    description: Optional[str], 
    assigned_user_id: Optional[int],
    status: str,
    due_date: Optional[datetime],
    created_by_id: int,
    session: Session
) -> Task:
    """
    Create a new task with the provided data
    
    Args:
        title: The title of the task
        description: The description of the task (optional)
        assigned_user_id: The ID of the user assigned to the task (optional)
        status: The initial status of the task
        due_date: The due date of the task (optional)
        created_by_id: The ID of the user creating the task
        session: Database session
        
    Returns:
        The created task object
        
    Raises:
        TaskManagementError: If assigned user is not found or other errors occur
    """
    # Validate assigned user if provided
    if assigned_user_id:
        assigned_user = session.query(User).filter(User.id == assigned_user_id).first()
        if not assigned_user:
            raise TaskManagementError("Assigned user not found")
        
        # Check if assigned user is active
        if not assigned_user.is_active:
            raise TaskManagementError("Cannot assign task to inactive user")
    
    # Validate status
    valid_statuses = ["pending", "in_progress", "completed"]
    if status not in valid_statuses:
        raise TaskManagementError(f"Invalid status. Must be one of: {', '.join(valid_statuses)}")
    
    # Create task
    task = Task(
        title=title,
        description=description,
        assigned_user_id=assigned_user_id,
        status=status,
        due_date=due_date,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow()
    )
    
    try:
        session.add(task)
        session.commit()
        session.refresh(task)
        
        # Log activity
        if assigned_user_id:
            action_details = f"Task '{title}' created and assigned to user ID {assigned_user_id}"
        else:
            action_details = f"Task '{title}' created"
            
        await log_activity(
            user_id=created_by_id,
            action_type=ActionType.TASK_CREATE,
            details=action_details,
            session=session
        )
        
        return task
    except Exception as e:
        session.rollback()
        raise TaskManagementError(f"Error creating task: {str(e)}")

async def update_task(
    task_id: int,
    title: Optional[str] = None,
    description: Optional[str] = None,
    assigned_user_id: Optional[int] = None,
    status: Optional[str] = None,
    due_date: Optional[datetime] = None,
    updated_by_id: int = None,
    session: Session = None
) -> Task:
    """
    Update an existing task with the provided data
    
    Args:
        task_id: The ID of the task to update
        title: The updated title (optional)
        description: The updated description (optional)
        assigned_user_id: The ID of the user assigned to the task (optional)
        status: The updated status (optional)
        due_date: The updated due date (optional)
        updated_by_id: The ID of the user updating the task
        session: Database session
        
    Returns:
        The updated task object
        
    Raises:
        TaskManagementError: If task or assigned user is not found, or other errors occur
    """
    task = session.query(Task).filter(Task.id == task_id).first()
    
    if not task:
        raise TaskManagementError("Task not found")
    
    # Track changes for activity log
    changes = []
    old_assigned_user_id = task.assigned_user_id
    
    # Update fields if provided
    if title is not None:
        if task.title != title:
            changes.append(f"title: '{task.title}' -> '{title}'")
            task.title = title
    
    if description is not None:
        if task.description != description:
            changes.append("description updated")
            task.description = description
    
    if assigned_user_id is not None:
        if assigned_user_id != task.assigned_user_id:
            # Validate assigned user
            if assigned_user_id > 0:  # Allow unassignment by passing 0
                assigned_user = session.query(User).filter(User.id == assigned_user_id).first()
                if not assigned_user:
                    raise TaskManagementError("Assigned user not found")
                
                # Check if assigned user is active
                if not assigned_user.is_active:
                    raise TaskManagementError("Cannot assign task to inactive user")
                
                changes.append(f"assigned user: {task.assigned_user_id or 'None'} -> {assigned_user_id}")
                task.assigned_user_id = assigned_user_id
            else:
                changes.append(f"task unassigned (was: {task.assigned_user_id})")
                task.assigned_user_id = None
    
    if status is not None:
        valid_statuses = ["pending", "in_progress", "completed"]
        if status not in valid_statuses:
            raise TaskManagementError(f"Invalid status. Must be one of: {', '.join(valid_statuses)}")
        
        if task.status != status:
            changes.append(f"status: {task.status} -> {status}")
            task.status = status
    
    if due_date is not None:
        if task.due_date != due_date:
            changes.append(f"due date updated")
            task.due_date = due_date
    
    # Only update if there were changes
    if changes:
        task.updated_at = datetime.utcnow()
        
        try:
            session.commit()
            session.refresh(task)
            
            # Log activity if there were changes
            if changes:
                action_details = f"Task '{task.title}' updated: " + ", ".join(changes)
                await log_activity(
                    user_id=updated_by_id,
                    action_type=ActionType.TASK_UPDATE,
                    details=action_details,
                    session=session
                )
                
                # If assignment changed, log additional activity
                if old_assigned_user_id != task.assigned_user_id:
                    if task.assigned_user_id:
                        await log_activity(
                            user_id=updated_by_id,
                            action_type=ActionType.TASK_ASSIGN,
                            details=f"Task '{task.title}' assigned to user ID {task.assigned_user_id}",
                            session=session
                        )
                    else:
                        await log_activity(
                            user_id=updated_by_id,
                            action_type=ActionType.TASK_UNASSIGN,
                            details=f"Task '{task.title}' unassigned (was assigned to user ID {old_assigned_user_id})",
                            session=session
                        )
            
            return task
        except Exception as e:
            session.rollback()
            raise TaskManagementError(f"Error updating task: {str(e)}")
    
    return task

async def delete_task(task_id: int, user_id: int, session: Session) -> bool:
    """
    Delete a task
    
    Args:
        task_id: The ID of the task to delete
        user_id: The ID of the user deleting the task
        session: Database session
        
    Returns:
        True if the task was deleted successfully
        
    Raises:
        TaskManagementError: If task is not found or other errors occur
    """
    task = session.query(Task).filter(Task.id == task_id).first()
    
    if not task:
        raise TaskManagementError("Task not found")
    
    try:
        # Store task info for activity log
        task_title = task.title
        assigned_user_id = task.assigned_user_id
        
        # Delete task
        session.delete(task)
        session.commit()
        
        # Log activity
        action_details = f"Task '{task_title}' deleted"
        if assigned_user_id:
            action_details += f" (was assigned to user ID {assigned_user_id})"
            
        await log_activity(
            user_id=user_id,
            action_type=ActionType.TASK_DELETE,
            details=action_details,
            session=session
        )
        
        return True
    except Exception as e:
        session.rollback()
        raise TaskManagementError(f"Error deleting task: {str(e)}")

async def get_user_tasks(
    user_id: int, 
    status: Optional[str] = None,
    include_overdue: Optional[bool] = False,
    session: Session = None
) -> List[Dict[str, Any]]:
    """
    Get all tasks assigned to a user, with optional filtering
    
    Args:
        user_id: The ID of the user
        status: Filter by task status (optional)
        include_overdue: Include only overdue tasks if True (optional)
        session: Database session
        
    Returns:
        A list of tasks with formatted data
        
    Raises:
        TaskManagementError: If user is not found
    """
    user = session.query(User).filter(User.id == user_id).first()
    
    if not user:
        raise TaskManagementError("User not found")
    
    # Build query with filters
    query = session.query(Task).filter(Task.assigned_user_id == user_id)
    
    if status:
        valid_statuses = ["pending", "in_progress", "completed"]
        if status not in valid_statuses:
            raise TaskManagementError(f"Invalid status. Must be one of: {', '.join(valid_statuses)}")
        query = query.filter(Task.status == status)
    
    if include_overdue:
        query = query.filter(
            and_(
                Task.due_date < datetime.utcnow(),
                Task.status != "completed"
            )
        )
    
    # Order by due date (closest first), then created date (newest first)
    query = query.order_by(
        Task.due_date.asc().nullslast(),
        desc(Task.created_at)
    )
    
    tasks = query.all()
    
    # Format tasks for response
    formatted_tasks = []
    for task in tasks:
        formatted_tasks.append({
            "id": task.id,
            "title": task.title,
            "description": task.description,
            "status": task.status,
            "due_date": task.due_date.isoformat() if task.due_date else None,
            "is_overdue": task.due_date < datetime.utcnow() if task.due_date else False,
            "created_at": task.created_at.isoformat(),
            "updated_at": task.updated_at.isoformat()
        })
    
    return formatted_tasks

async def get_all_tasks(
    status: Optional[str] = None,
    assigned_user_id: Optional[int] = None,
    include_unassigned: Optional[bool] = False,
    include_overdue: Optional[bool] = False,
    session: Session = None
) -> List[Dict[str, Any]]:
    """
    Get all tasks in the system, with optional filtering
    
    Args:
        status: Filter by task status (optional)
        assigned_user_id: Filter by assigned user ID (optional)
        include_unassigned: Include unassigned tasks if True (optional)
        include_overdue: Include only overdue tasks if True (optional)
        session: Database session
        
    Returns:
        A list of tasks with formatted data
    """
    # Build query with filters
    query = session.query(Task)
    
    if status:
        valid_statuses = ["pending", "in_progress", "completed"]
        if status not in valid_statuses:
            raise TaskManagementError(f"Invalid status. Must be one of: {', '.join(valid_statuses)}")
        query = query.filter(Task.status == status)
    
    if assigned_user_id:
        query = query.filter(Task.assigned_user_id == assigned_user_id)
    elif include_unassigned:
        query = query.filter(Task.assigned_user_id == None)
    
    if include_overdue:
        query = query.filter(
            and_(
                Task.due_date < datetime.utcnow(),
                Task.status != "completed"
            )
        )
    
    # Order by due date (closest first), then created date (newest first)
    query = query.order_by(
        Task.due_date.asc().nullslast(),
        desc(Task.created_at)
    )
    
    tasks = query.all()
    
    # Format tasks for response with assigned user info
    formatted_tasks = []
    for task in tasks:
        task_data = {
            "id": task.id,
            "title": task.title,
            "description": task.description,
            "status": task.status,
            "due_date": task.due_date.isoformat() if task.due_date else None,
            "is_overdue": task.due_date < datetime.utcnow() if task.due_date else False,
            "assigned_user_id": task.assigned_user_id,
            "assigned_user_name": task.assigned_user.name if task.assigned_user else None,
            "created_at": task.created_at.isoformat(),
            "updated_at": task.updated_at.isoformat()
        }
        formatted_tasks.append(task_data)
    
    return formatted_tasks

async def get_task_summary(user_id: Optional[int] = None, session: Session = None) -> Dict[str, Any]:
    """
    Get a summary of tasks by status
    
    Args:
        user_id: Get summary for specific user tasks (optional, None for all tasks)
        session: Database session
        
    Returns:
        A dictionary with counts by status and overdue info
    """
    # Base query
    query = session.query(Task)
    
    # Filter by user if specified
    if user_id:
        query = query.filter(Task.assigned_user_id == user_id)
    
    # Get all tasks
    tasks = query.all()
    
    # Count by status
    pending_count = sum(1 for task in tasks if task.status == "pending")
    in_progress_count = sum(1 for task in tasks if task.status == "in_progress")
    completed_count = sum(1 for task in tasks if task.status == "completed")
    
    # Count overdue tasks
    now = datetime.utcnow()
    overdue_count = sum(
        1 for task in tasks 
        if task.due_date and task.due_date < now and task.status != "completed"
    )
    
    # Get nearest due task
    active_tasks = [task for task in tasks if task.status != "completed" and task.due_date]
    nearest_due = min(active_tasks, key=lambda x: x.due_date) if active_tasks else None
    
    # Build summary
    summary = {
        "total_tasks": len(tasks),
        "pending_tasks": pending_count,
        "in_progress_tasks": in_progress_count,
        "completed_tasks": completed_count,
        "overdue_tasks": overdue_count,
        "completion_rate": (completed_count / len(tasks)) * 100 if tasks else 0,
        "nearest_due_task": {
            "id": nearest_due.id,
            "title": nearest_due.title,
            "due_date": nearest_due.due_date.isoformat()
        } if nearest_due else None
    }
    
    return summary