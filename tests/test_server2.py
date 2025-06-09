import types
import pytest
from unittest.mock import MagicMock, patch


def test_create_copick_app(app2):
    """Test that the app is created correctly."""
    assert app2 is not None
    assert app2.routes is not None
    # Check that our catch-all route exists
    assert any(route.path == "/{path:path}" for route in app2.routes)
    assert any(route.path == "/Picks" for route in app2.routes)
    assert any(route.path == "/Segmentations" for route in app2.routes)
    assert any(route.path == "/Tomograms" for route in app2.routes)

def test_cors_middleware(client2):
    """Test that CORS is properly configured."""
    # Make a request with an Origin header
    response = client2.get("/any-path", headers={"Origin": "https://example.com"})

    # Check if CORS headers are present in the response
    assert "access-control-allow-origin" in response.headers, "CORS headers not found in response"
    assert response.headers["access-control-allow-origin"] == "*", "Incorrect CORS origin value"


@pytest.mark.asyncio
async def test_handle_request_invalid_path(client2):
    """Test handling of an invalid path."""
    response = client2.get("/invalid/path")
    assert response.status_code == 404

@pytest.mark.asyncio
@patch("copick_server.server.CopickRoute._handle_tomogram")
async def test_handle_tomogram_request(mock_handle_tomogram, client2, monkeypatch):
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
    for route in client2.app.routes:
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
        response = client2.get("/test_run/Tomograms/VoxelSpacing10.0/test.zarr")
        
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
async def test_handle_picks_request(mock_handle_picks, client2, monkeypatch):
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
    for route in client2.app.routes:
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
        response = client2.get("/test_run/Picks/user_session_test.json")
        
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
    mock_handle_segmentation, client2, monkeypatch
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
    for route in client2.app.routes:
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
        response = client2.get("/test_run/Segmentations/10.0_user_session_test.zarr")
        
        # Verify the response
        assert response.status_code == 200
        
        # Verify the correct run was obtained
        root_mock.get_run.assert_called_once_with("test_run")
        
        # Verify _handle_segmentation was called
        mock_handle_segmentation.assert_called_once()
    finally:
        # Restore the original root
        route_handler.root = original_root


@pytest.mark.asyncio
async def test_handle_picks_request2(client2):
    """Test that picks requests are routed correctly."""
    picks_return = client2.get("/Picks?run_id=test_run&user_id=test_user&session_id=test_session&name=test_object")
    assert picks_return.status_code == 200
    assert picks_return.json() == [{"location": {"x": 1, "y": 2, "z": 3}, "pickable_object_name": "test_object"}]
    

@pytest.mark.asyncio
async def test_handle_put_picks_request2(client2):
    """Test that PUT requests for picks are handled correctly."""
    data = {
        "pickable_object_name": "test_object",
        "user_id": "test_user",
        "session_id": "test_session",
        "picks": [{"location": {"x": 1, "y": 2, "z": 3}}],
    }
    response = client2.put("/Picks?run_id=test_run", json=data)
    
    assert response.status_code == 200
    assert response.json()
    