import pytest
import numpy as np
from unittest.mock import MagicMock, patch, AsyncMock


@pytest.mark.asyncio
async def test_handle_tomogram_invalid_path():
    """Test handling of an invalid tomogram path."""
    from copick_server.server import CopickRoute

    # Create mocks
    run_mock = MagicMock()
    request_mock = MagicMock()
    route_handler = CopickRoute(MagicMock())

    # Test with too short path
    response = await route_handler._handle_tomogram(request_mock, run_mock, "invalid")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_handle_tomogram_invalid_voxel_spacing():
    """Test handling of an invalid voxel spacing in tomogram path."""
    from copick_server.server import CopickRoute

    # Create mocks
    run_mock = MagicMock()
    request_mock = MagicMock()
    route_handler = CopickRoute(MagicMock())

    # Test with invalid voxel spacing
    response = await route_handler._handle_tomogram(
        request_mock, run_mock, "VoxelSpacingXYZ/test.zarr"
    )
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_handle_tomogram_unknown_voxel_spacing():
    """Test handling of an unknown voxel spacing."""
    from copick_server.server import CopickRoute

    # Create mocks
    run_mock = MagicMock()
    run_mock.get_voxel_spacing.return_value = None
    request_mock = MagicMock()
    route_handler = CopickRoute(MagicMock())

    # Test with unknown voxel spacing
    response = await route_handler._handle_tomogram(
        request_mock, run_mock, "VoxelSpacing10.0/test.zarr"
    )
    assert response.status_code == 404
    run_mock.get_voxel_spacing.assert_called_once_with(10.0)


@pytest.mark.asyncio
async def test_handle_tomogram_unknown_tomogram():
    """Test handling of an unknown tomogram type."""
    from copick_server.server import CopickRoute

    # Create mocks
    vs_mock = MagicMock()
    vs_mock.get_tomogram.return_value = None
    run_mock = MagicMock()
    run_mock.get_voxel_spacing.return_value = vs_mock
    request_mock = MagicMock()
    route_handler = CopickRoute(MagicMock())

    # Test with unknown tomogram
    response = await route_handler._handle_tomogram(
        request_mock, run_mock, "VoxelSpacing10.0/test.zarr"
    )
    assert response.status_code == 404
    vs_mock.get_tomogram.assert_called_once_with("test")


@pytest.mark.asyncio
async def test_handle_tomogram_put_readonly():
    """Test handling of PUT request to readonly tomogram."""
    from copick_server.server import CopickRoute

    # Create mocks
    tomogram_mock = MagicMock()
    tomogram_mock.read_only = True
    vs_mock = MagicMock()
    vs_mock.get_tomogram.return_value = tomogram_mock
    run_mock = MagicMock()
    run_mock.get_voxel_spacing.return_value = vs_mock
    request_mock = MagicMock()
    request_mock.method = "PUT"
    route_handler = CopickRoute(MagicMock())

    # Test PUT to readonly tomogram
    body_mock = AsyncMock()
    request_mock.body = body_mock

    # Mock the zarr method to return a dict-like object
    zarr_mock = MagicMock()
    zarr_mock.__getitem__.return_value = b"test_data"
    tomogram_mock.zarr.return_value = zarr_mock

    response = await route_handler._handle_tomogram(
        request_mock, run_mock, "VoxelSpacing10.0/test.zarr/0"
    )
    assert response.status_code == 200

    # Since it's readonly, zarr.__getitem__ should be called, not zarr.__setitem__
    zarr_mock.__getitem__.assert_called_once_with("0")
    zarr_mock.__setitem__.assert_not_called()


@pytest.mark.asyncio
async def test_handle_picks_invalid_path():
    """Test handling of an invalid picks path."""
    from copick_server.server import CopickRoute

    # Create mocks
    run_mock = MagicMock()
    request_mock = MagicMock()
    route_handler = CopickRoute(MagicMock())

    # Test with empty path
    response = await route_handler._handle_picks(request_mock, run_mock, "")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_handle_picks_invalid_format():
    """Test handling of picks with invalid filename format."""
    from copick_server.server import CopickRoute

    # Create mocks
    run_mock = MagicMock()
    request_mock = MagicMock()
    route_handler = CopickRoute(MagicMock())

    # Test with invalid format (should have 3 parts: user_session_object.json)
    response = await route_handler._handle_picks(
        request_mock, run_mock, "invalid_format.json"
    )
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_handle_picks_get_not_found():
    """Test handling of GET request for non-existent picks."""
    from copick_server.server import CopickRoute

    # Create mocks
    run_mock = MagicMock()
    run_mock.get_picks.return_value = []  # No picks found
    request_mock = MagicMock()
    request_mock.method = "GET"
    route_handler = CopickRoute(MagicMock())

    # Test GET for non-existent picks
    response = await route_handler._handle_picks(
        request_mock, run_mock, "user_session_object.json"
    )
    assert response.status_code == 404
    run_mock.get_picks.assert_called_once_with(
        object_name="object", user_id="user", session_id="session"
    )


@pytest.mark.asyncio
async def test_handle_segmentation_invalid_path():
    """Test handling of an invalid segmentation path."""
    from copick_server.server import CopickRoute

    # Create mocks
    run_mock = MagicMock()
    request_mock = MagicMock()
    route_handler = CopickRoute(MagicMock())

    # Test with empty path
    response = await route_handler._handle_segmentation(request_mock, run_mock, "")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_handle_segmentation_invalid_format():
    """Test handling of segmentation with invalid filename format."""
    from copick_server.server import CopickRoute

    # Create mocks
    run_mock = MagicMock()
    request_mock = MagicMock()
    route_handler = CopickRoute(MagicMock())

    # Test with invalid format (should have at least 4 parts: voxel_user_session_name.zarr)
    response = await route_handler._handle_segmentation(
        request_mock, run_mock, "invalid_format.zarr"
    )
    assert response.status_code == 404


@pytest.mark.asyncio
@patch("copick_utils.writers.write.segmentation")
async def test_handle_segmentation_put(mock_write_segmentation):
    """Test handling of PUT request for a segmentation."""
    from copick_server.server import CopickRoute

    # Create mocks
    run_mock = MagicMock()
    request_mock = MagicMock()
    request_mock.method = "PUT"
    route_handler = CopickRoute(MagicMock())

    # Mock the request body to return a byte array with shape info and data
    # (24 bytes for shape + data)
    shape = np.array([10, 10, 10], dtype=np.int64)
    data = np.zeros((10, 10, 10), dtype=np.uint8)
    body = shape.tobytes() + data.tobytes()

    body_mock = AsyncMock()
    body_mock.return_value = body
    request_mock.body = body_mock

    # Test PUT for segmentation
    response = await route_handler._handle_segmentation(
        request_mock, run_mock, "10.0_user_session_object.zarr"
    )

    assert response.status_code == 200
    mock_write_segmentation.assert_called_once()

    # Check that the parameters are correct
    args, kwargs = mock_write_segmentation.call_args
    assert kwargs["run"] == run_mock
    assert kwargs["user_id"] == "user"
    assert kwargs["session_id"] == "session"
    assert kwargs["name"] == "object"
    assert kwargs["voxel_size"] == 10.0
    assert not kwargs["multilabel"]

    # Check that the segmentation volume shape is correct
    assert kwargs["segmentation_volume"].shape == (10, 10, 10)


@pytest.mark.asyncio
async def test_handle_segmentation_get_not_found():
    """Test handling of GET request for non-existent segmentation."""
    from copick_server.server import CopickRoute

    # Create mocks
    run_mock = MagicMock()
    run_mock.get_segmentations.return_value = []  # No segmentations found
    request_mock = MagicMock()
    request_mock.method = "GET"
    route_handler = CopickRoute(MagicMock())

    # Test GET for non-existent segmentation
    response = await route_handler._handle_segmentation(
        request_mock, run_mock, "10.0_user_session_object.zarr"
    )

    assert response.status_code == 404
    run_mock.get_segmentations.assert_called_once_with(
        voxel_size=10.0,
        name="object",
        user_id="user",
        session_id="session",
        is_multilabel=False,
    )
