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

    # Create app with CORS origins
    app = create_copick_app(mock_copick_root, cors_origins=["https://example.com"])

    # Check that the middleware exists
    assert len(app.middleware) > 0
    # Check if any of the middleware contains CORSMiddleware
    cors_middleware_found = False
    for middleware in app.middleware:
        if str(middleware) and "CORSMiddleware" in str(middleware):
            cors_middleware_found = True
            break
    assert cors_middleware_found


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

    # Patch to replace the route handler's root with our mock
    with patch("copick_server.server.CopickRoute.root", root_mock):
        # Set up mock for _handle_tomogram
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_handle_tomogram.return_value = mock_response

        # Make the request
        response = client.get("/test_run/Tomograms/VoxelSpacing10.0/test.zarr")

        # Verify the response
        assert response.status_code == 200

        # Verify the correct run was obtained
        root_mock.get_run.assert_called_once_with("test_run")

        # Verify _handle_tomogram was called
        mock_handle_tomogram.assert_called_once()


@pytest.mark.asyncio
@patch("copick_server.server.CopickRoute._handle_picks")
async def test_handle_picks_request(mock_handle_picks, client, monkeypatch):
    """Test that picks requests are routed correctly."""
    # Mock the get_run method to return a valid run
    run_mock = MagicMock()
    root_mock = MagicMock()
    root_mock.get_run.return_value = run_mock

    # Patch to replace the route handler's root with our mock
    with patch("copick_server.server.CopickRoute.root", root_mock):
        # Set up mock for _handle_picks
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_handle_picks.return_value = mock_response

        # Make the request
        response = client.get("/test_run/Picks/user_session_test.json")

        # Verify the response
        assert response.status_code == 200

        # Verify the correct run was obtained
        root_mock.get_run.assert_called_once_with("test_run")

        # Verify _handle_picks was called
        mock_handle_picks.assert_called_once()


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

    # Patch to replace the route handler's root with our mock
    with patch("copick_server.server.CopickRoute.root", root_mock):
        # Set up mock for _handle_segmentation
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_handle_segmentation.return_value = mock_response

        # Make the request
        response = client.get("/test_run/Segmentations/10.0_user_session_test.zarr")

        # Verify the response
        assert response.status_code == 200

        # Verify the correct run was obtained
        root_mock.get_run.assert_called_once_with("test_run")

        # Verify _handle_segmentation was called
        mock_handle_segmentation.assert_called_once()
