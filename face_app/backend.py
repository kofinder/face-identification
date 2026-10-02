"""AI backend bootstrap.

DeepFace checks the backend during import, so this environment variable must be
set before importing ``deepface.DeepFace`` anywhere in the application.
"""

import os

os.environ.setdefault("DEEPFACE_BACKEND_ENGINE", "pytorch")
