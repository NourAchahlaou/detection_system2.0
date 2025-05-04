from user_management.app.db.session import Base

# Import all models to register them with Base.metadata
# NOTE: These imports are for registering models with SQLAlchemy
# Don't use these imports directly in your application code
from user_management.app.db.models.user import User, UserToken
from user_management.app.db.models.shift import Shift
from user_management.app.db.models.activities import Activity
from user_management.app.db.models.workHours import WorkHours
from user_management.app.db.models.task import Task

# Define __all__ to control what gets imported with "from models import *"
__all__ = ['User', 'UserToken', 'Shift', 'Activity', 'WorkHours', 'Task']
