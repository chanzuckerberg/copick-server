import json
import types
import pytest
from unittest.mock import MagicMock, patch


def test_create_copick_app(app):
    """Test that the app is created correctly."""
    assert app is not None
    assert app.routes is not None
    # Check that our catch-all route exists
    assert any(route.path == "/{path:path}" for route in app.routes)


def test_cors_middleware(mock_copick_root):
    """Test that CORS middleware is added correctly."""
    from copick_server.server import create_copick_app
    from fastapi.middleware.cors import CORSMiddleware
    
    # Create app with CORS origins
    app = create_copick_app(mock_copick_root, cors_origins=["https://example.com"])
    
    # Check that the CORS middleware was added
    # Get all middlewares from the app
    middlewares = app.user_middleware
    
    # Debug middleware information
    middleware_types = [type(m).__name__ for m in middlewares]
    
    # Look for the CORSMiddleware
    cors_middleware_found = False
    for middleware in middlewares:
        if type(middleware).__name__ == "CORSMiddleware":
            cors_middleware_found = True
            break
    
    assert cors_middleware_found, f"CORS middleware not found among middleware types: {middleware_types}"


@pytest.mark.asyncio
async def test_handle_request_invalid_path(client):
    """Test handling of an invalid path."""
    response = client.get("/invalid/path")
    assert response.status_code == 404


@pytest.mark.asyncio
@patch("copick_server.server.CopickRoute._handle_tomogram")
async def test_handle_tomogram_request(mock_handle_tomogram, client, monkeypatch):
    """Test that tomogram requests are routed correctly."""
    # Mock the get_run method to return a valid run
    run_mock = MagicMock()
    root_mock = MagicMock()
    root_mock.get_run.return_value = run_mock
    
    # Set up mock for _handle_tomogram
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_handle_tomogram.return_value = mock_response
    
    # Find the route handler in the application
    route_handler = None
    for route in client.app.routes:
        if isinstance(route.endpoint, types.MethodType) and route.endpoint.__self__.__class__.__name__ == 'CopickRoute':
            route_handler = route.endpoint.__self__
            break
    
    assert route_handler is not None, "Could not find CopickRoute handler"
    
    # Save the original root
    original_root = route_handler.root
    
    # Temporarily replace the root
    route_handler.root = root_mock
    
    try:
        # Make the request
        response = client.get("/test_run/Tomograms/VoxelSpacing10.0/test.zarr")
        
        # Verify the response
        assert response.status_code == 200
        
        # Verify the correct run was obtained
        root_mock.get_run.assert_called_once_with("test_run")
        
        # Verify _handle_tomogram was called
        mock_handle_tomogram.assert_called_once()
    finally:
        # Restore the original root
        route_handler.root = original_root


@pytest.mark.asyncio
@patch("copick_server.server.CopickRoute._handle_picks")
async def test_handle_picks_request(mock_handle_picks, client, monkeypatch):
    """Test that picks requests are routed correctly."""
    # Mock the get_run method to return a valid run
    run_mock = MagicMock()
    root_mock = MagicMock()
    root_mock.get_run.return_value = run_mock
    
    # Set up mock for _handle_picks
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_handle_picks.return_value = mock_response
    
    # Find the route handler in the application
    route_handler = None
    for route in client.app.routes:
        if isinstance(route.endpoint, types.MethodType) and route.endpoint.__self__.__class__.__name__ == 'CopickRoute':
            route_handler = route.endpoint.__self__
            break
    
    assert route_handler is not None, "Could not find CopickRoute handler"
    
    # Save the original root
    original_root = route_handler.root
    
    # Temporarily replace the root
    route_handler.root = root_mock
    
    try:
        # Make the request
        response = client.get("/test_run/Picks/user_session_test.json")
        
        # Verify the response
        assert response.status_code == 200
        
        # Verify the correct run was obtained
        root_mock.get_run.assert_called_once_with("test_run")
        
        # Verify _handle_picks was called
        mock_handle_picks.assert_called_once()
    finally:
        # Restore the original root
        route_handler.root = original_root


@pytest.mark.asyncio
@patch("copick_server.server.CopickRoute._handle_segmentation")
async def test_handle_segmentation_request(
    mock_handle_segmentation, client, monkeypatch
):
    """Test that segmentation requests are routed correctly."""
    # Mock the get_run method to return a valid run
    run_mock = MagicMock()
    root_mock = MagicMock()
    root_mock.get_run.return_value = run_mock
    
    # Set up mock for _handle_segmentation
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_handle_segmentation.return_value = mock_response
    
    # Find the route handler in the application
    route_handler = None
    for route in client.app.routes:
        if isinstance(route.endpoint, types.MethodType) and route.endpoint.__self__.__class__.__name__ == 'CopickRoute':
            route_handler = route.endpoint.__self__
            break
    
    assert route_handler is not None, "Could not find CopickRoute handler"
    
    # Save the original root
    original_root = route_handler.root
    
    # Temporarily replace the root
    route_handler.root = root_mock
    
    try:
        # Make the request
        response = client.get("/test_run/Segmentations/10.0_user_session_test.zarr")
        
        # Verify the response
        assert response.status_code == 200
        
        # Verify the correct run was obtained
        root_mock.get_run.assert_called_once_with("test_run")
        
        # Verify _handle_segmentation was called
        mock_handle_segmentation.assert_called_once()
    finally:
        # Restore the original root
        route_handler.root = original_root
