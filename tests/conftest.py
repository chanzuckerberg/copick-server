import json
import tempfile
from pathlib import Path

import copick
import pytest
from fastapi.testclient import TestClient

from copick_server.server import create_copick_app


@pytest.fixture
def example_config():
    """Create a temporary example Copick configuration."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create the overlay directory structure
        overlay_path = Path(temp_dir) / "overlay"
        overlay_path.mkdir()

        # Create a simple Copick config
        config_data = {
            "name": "Test Project",
            "description": "Test project for unit tests",
            "version": "0.5.0",
            "pickable_objects": [
                {
                    "name": "test_object",
                    "is_particle": True,
                    "label": 1,
                    "color": [0, 117, 220, 255],
                    "radius": 150.0,
                }
            ],
            "user_id": "test_user",
            "config_type": "cryoet_data_portal",
            "overlay_root": f"local://{str(overlay_path)}/",
            "dataset_ids": [12345],
            "overlay_fs_args": {"auto_mkdir": True},
        }

        # Write config to a temporary file
        config_path = Path(temp_dir) / "test_config.json"
        with open(config_path, "w") as f:
            json.dump(config_data, f)

        yield str(config_path)


@pytest.fixture
def mock_copick_root(monkeypatch, example_config):
    """Mock a CopickRoot instance for testing."""
    # Create a minimal mock root with just enough functionality for testing
    # In a real setup, you would mock specific methods of CopickRoot

    # This is a simple case - for more complex mocking, consider using unittest.mock
    return copick.from_file(example_config)


@pytest.fixture
def app(mock_copick_root):
    """Create a test FastAPI application."""
    return create_copick_app(mock_copick_root, cors_origins=["*"])


@pytest.fixture
def client(app):
    """Create a test client for the application."""
    return TestClient(app)
