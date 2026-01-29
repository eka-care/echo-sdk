"""
Pytest configuration for tests.

Adds the examples directory to the Python path so tests can import from examples/crews.
"""

import sys
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add the project root to sys.path so 'examples' can be imported
project_root = Path(__file__).parent.parent
examples_path = project_root / "examples"

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

if str(examples_path) not in sys.path:
    sys.path.insert(0, str(examples_path))
