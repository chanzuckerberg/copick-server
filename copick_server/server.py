import json
from typing import List, Optional

import click
import copick
import numpy as np
import uvicorn
import threading

from fastapi import FastAPI, Request, Response


class CopickRoute:
    """Route handler for Copick data entities."""

    def __init__(self, root: copick.models.CopickRoot):
        self.root = root

    async def handle_request(self, request: Request, path: str):
        # Parse path parameters
        path_parts = path.split("/")
        print(path_parts)

        # Handle different path patterns
        try:
            if len(path_parts) >= 3:
                run_name = path_parts[0]
                data_type = path_parts[1]
                print(run_name)
                print(data_type)

                # Get the run
                run = self.root.get_run(run_name)
                print("run", run)
                if run is None:
                    return Response(status_code=404)

                if data_type == "Tomograms":
                    return await self._handle_tomogram(
                        request, run, "/".join(path_parts[2:])
                    )
                elif data_type == "Picks":
                    return await self._handle_picks(
                        request, run, "/".join(path_parts[2:])
                    )
                elif data_type == "Segmentations":
                    return await self._handle_segmentation(
                        request, run, "/".join(path_parts[2:])
                    )

            return Response(status_code=404)

        except Exception as e:
            print(f"Error handling request: {str(e)}")
            return Response(status_code=500)

    async def _handle_tomogram(self, request, run, path):
        # Extract voxel spacing and tomogram type from path
        parts = path.split("/")
        if len(parts) < 2:
            return Response(status_code=404)

        vs_str = parts[0].replace("VoxelSpacing", "")
        try:
            voxel_spacing = float(vs_str)
        except ValueError:
            return Response(status_code=404)

        tomo_type = parts[1].replace(".zarr", "")

        # Get the tomogram
        vs = run.get_voxel_spacing(voxel_spacing)
        if vs is None:
            return Response(status_code=404)
        print("tomo_type", tomo_type)
        print("vs", vs)

        tomogram = vs.get_tomogram(tomo_type)
        print(tomogram)
        if tomogram is None:
            return Response(status_code=404)

        # Handle the request
        if request.method == "PUT" and not tomogram.read_only:
            try:
                blob = await request.body()
                tomogram.zarr()["/".join(parts[2:])] = blob
                return Response(status_code=200)
            except Exception:
                return Response(status_code=500)
        else:
            try:
                body = tomogram.zarr()["/".join(parts[2:])]
                if request.method == "HEAD":
                    body = None
                return Response(body, status_code=200)
            except KeyError:
                return Response(status_code=404)

    async def _handle_picks(self, request, run, path):
        # Extract object name, user ID and session ID from path
        parts = path.split("/")
        if len(parts) < 1:
            return Response(status_code=404)

        pick_file = parts[0]
        pick_parts = pick_file.split("_")
        if len(pick_parts) != 3:
            return Response(status_code=404)

        user_id, session_id, object_name = pick_parts
        object_name = object_name.replace(".json", "")
        print("user_id", user_id)
        print("session_id", session_id)
        print("object_name", object_name)

        # Get or create picks
        picks = None
        if request.method == "PUT":
            try:
                picks = run.new_picks(
                    object_name=object_name, user_id=user_id, session_id=session_id
                )
                data = await request.json()
                picks.meta = copick.models.CopickPicksFile(**data)
                picks.store()
                return Response(status_code=200)
            except Exception as e:
                print(f"Picks write error: {str(e)}")
                return Response(status_code=500)
        else:
            picks = run.picks
            if not picks:
                return Response(status_code=404)

            if request.method == "HEAD":
                return Response(status_code=200)

            return Response(json.dumps(picks[0].meta.dict()), status_code=200)

    async def _handle_segmentation(self, request, run, path):
        # Extract segmentation parameters from path
        parts = path.split("/")
        if len(parts) < 1:
            return Response(status_code=404)

        seg_file = parts[0].replace(".zarr", "")
        seg_parts = seg_file.split("_")
        if len(seg_parts) < 4:
            return Response(status_code=404)

        voxel_size = float(seg_parts[0])
        user_id = seg_parts[1]
        session_id = seg_parts[2]
        name = "_".join(seg_parts[3:])
        is_multilabel = "multilabel" in name

        # Get or create segmentation
        if request.method == "PUT":
            try:
                # Get the data from the request body
                blob = await request.body()

                # Extract shape information (first 24 bytes contain 3 int64 values)
                shape = np.frombuffer(blob[:24], dtype=np.int64)

                # Extract the actual data and reshape it
                data = np.frombuffer(blob[24:], dtype=np.uint8).reshape(shape)

                # Import the writer utility
                from copick_utils.writers.write import segmentation

                # Use the utility function to write the segmentation
                seg = segmentation(
                    run=run,
                    segmentation_volume=data,
                    user_id=user_id,
                    name=name.replace("-multilabel", ""),
                    session_id=session_id,
                    voxel_size=voxel_size,
                    multilabel=is_multilabel,
                )

                return Response(status_code=200)
            except Exception as e:
                print(f"Segmentation write error: {str(e)}")
                return Response(status_code=500)
        else:
            segs = run.get_segmentations(
                voxel_size=voxel_size,
                name=name.replace("-multilabel", ""),
                user_id=user_id,
                session_id=session_id,
                is_multilabel=is_multilabel,
            )
            if not segs:
                return Response(status_code=404)

            seg = segs[0]
            try:
                body = seg.zarr()["/".join(parts[1:])]
                if request.method == "HEAD":
                    body = None
                return Response(body, status_code=200)
            except KeyError:
                return Response(status_code=404)


def create_copick_app(
    root: copick.models.CopickRoot, cors_origins: Optional[List[str]] = None
) -> FastAPI:
    """Create a FastAPI app for serving a Copick project.

    Parameters
    ----------
    root : copick.models.CopickRoot
        Copick project root to serve
    cors_origins : list of str, optional
        List of allowed CORS origins. Use ["*"] to allow all.

    Returns
    -------
    app : FastAPI
        FastAPI application
    """
    app = FastAPI()
    route_handler = CopickRoute(root)

    @app.get("/health/")
    def health_check():
        """Health check endpoint."""
        return {"status": "ok"}

    # Add the catch-all route
    app.add_api_route(
        "/{path:path}", route_handler.handle_request, methods=["GET", "HEAD", "PUT"]
    )

    # Add CORS middleware if origins are specified
    if cors_origins:
        # Ensure CORS middleware is properly initialized
        try:
            from fastapi.middleware.cors import CORSMiddleware

            app.add_middleware(
                CORSMiddleware,
                allow_origins=cors_origins,
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )
            # Print for debugging
            print(f"CORS middleware added with origins: {cors_origins}")
        except Exception as e:
            print(f"Error adding CORS middleware: {str(e)}")

    return app


def serve_copick(
    config_path: Optional[str] = None,
    dataset_ids: Optional[List[int]] = None,
    overlay_root: str = "/tmp/overlay_root",
    allowed_origins: Optional[List[str]] = None,
    **kwargs,
):
    """Start an HTTP server serving a Copick project.

    Parameters
    ----------
    config_path : str, optional
        Path to Copick config file
    dataset_ids : list of int, optional
        Dataset IDs to include in the project
    overlay_root : str, optional
        Root URL for the overlay storage, default is "/tmp/overlay_root"
    allowed_origins : list of str, optional
        List of allowed CORS origins. Use ["*"] to allow all.
    **kwargs
        Additional arguments passed to uvicorn.run()

    Notes
    -----
    Either config_path or dataset_ids must be provided, but not both.
    """
    if config_path and dataset_ids:
        raise ValueError(
            "Either config_path or dataset_ids must be provided, but not both."
        )
    elif config_path:
        root = copick.from_file(config_path)
    elif dataset_ids:
        root = copick.from_czcdp_datasets(
            dataset_ids=dataset_ids,
            overlay_root=overlay_root,
            overlay_fs_args={"auto_mkdir": True},
        )
    else:
        raise ValueError("Either config_path or dataset_ids must be provided.")

    app = create_copick_app(root, allowed_origins)
    uvicorn.run(app, **kwargs)
    return app


def serve_copick_threaded(
    config_path: Optional[str] = None,
    dataset_ids: Optional[List[int]] = None,
    overlay_root: str = "/tmp/overlay_root",
    allowed_origins: Optional[List[str]] = None,
    **kwargs,
):
    """Start an HTTP server in a background thread and return the app.

    Parameters
    ----------
    config_path : str, optional
        Path to Copick config file
    dataset_ids : list of int, optional
        Dataset IDs to include in the project
    overlay_root : str, optional
        Root URL for the overlay storage, default is "/tmp/overlay_root"
    allowed_origins : list of str, optional
        List of allowed CORS origins. Use ["*"] to allow all.
    **kwargs
        Additional arguments passed to uvicorn.run()

    Returns
    -------
    app : FastAPI
        FastAPI application

    Notes
    -----
    Either config_path or dataset_ids must be provided, but not both.
    """
    if config_path and dataset_ids:
        raise ValueError(
            "Either config_path or dataset_ids must be provided, but not both."
        )
    elif config_path:
        root = copick.from_file(config_path)
    elif dataset_ids:
        root = copick.from_czcdp_datasets(
            dataset_ids=dataset_ids,
            overlay_root=overlay_root,
            overlay_fs_args={"auto_mkdir": True},
        )
    else:
        raise ValueError("Either config_path or dataset_ids must be provided.")

    app = create_copick_app(root, allowed_origins)

    # Start the server in a background thread
    server_thread = threading.Thread(
        target=uvicorn.run,
        args=(app,),
        kwargs=kwargs,
        daemon=True,  # This makes the thread exit when the main thread exits
    )
    server_thread.start()

    return app


@click.group()
@click.pass_context
def cli(ctx):
    pass


@cli.command()
@click.option(
    "-c",
    "--config",
    type=click.Path(exists=True),
    help="Path to the configuration file.",
    required=False,
    metavar="PATH",
)
@click.option(
    "-ds",
    "--dataset-ids",
    type=int,
    multiple=True,
    help="Dataset IDs to include in the project (multiple inputs possible).",
    metavar="ID",
)
@click.option(
    "--overlay-root",
    type=str,
    default="/tmp/overlay_root",
    help="Root URL for the overlay storage.",
    show_default=True,
)
@click.option(
    "--cors",
    type=str,
    default=None,
    help="Origin to allow CORS. Use wildcard '*' to allow all.",
)
@click.option(
    "--host",
    type=str,
    default="127.0.0.1",
    help="Bind socket to this host.",
    show_default=True,
)
@click.option(
    "--port",
    type=int,
    default=8000,
    help="Bind socket to this port.",
    show_default=True,
)
@click.option("--reload", is_flag=True, default=False, help="Enable auto-reload.")
@click.pass_context
def serve(
    ctx,
    config: Optional[str] = None,
    dataset_ids: Optional[tuple] = None,
    overlay_root: str = "/tmp/overlay_root",
    cors: Optional[str] = None,
    host: str = "127.0.0.1",
    port: int = 8000,
    reload: bool = False,
):
    """Serve a Copick project over HTTP."""

    if config and dataset_ids:
        ctx.fail("Either --config or --dataset-ids must be provided, not both.")
    elif not config and not dataset_ids:
        ctx.fail("Either --config or --dataset-ids must be provided.")

    try:
        serve_copick(
            config_path=config,
            dataset_ids=dataset_ids if dataset_ids else None,
            overlay_root=overlay_root,
            allowed_origins=[cors] if cors else None,
            host=host,
            port=port,
            reload=reload,
        )
    except Exception as e:
        ctx.fail(f"Error serving Copick project: {str(e)}")


def main():
    cli()


if __name__ == "__main__":
    main()
