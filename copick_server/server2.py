import uvicorn
from uvicorn.supervisors import ChangeReload

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from settings import Settings
import copick
from server import CopickRoute
import numpy as np
from copick_utils.writers.write import segmentation
from pydantic import BaseModel
import ibis
from rstar_python import PyRTree
from collections import defaultdict


### pydantic validators
class NearestPointQuery(BaseModel):
    x: float
    y: float
    z: float
    run_id: int
    dataset_id: str
    model: str  # TODO: convert to enum ?
    k_nearest: int = 10

### helpers

def points_to_index(x: float, y: float, z: float) -> tuple[int, int, int]:
    return (int(x * SCALING_FACTOR), int(y * SCALING_FACTOR), int(z * SCALING_FACTOR))

def get_app(settings: Settings) -> FastAPI:
    # Create and run app
    _app = FastAPI(title="Copick Server", debug=settings.DEBUG, dependencies=[])
    _app.state.settings = settings
    _app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    ## Add a basic fastapi endpoint
    return _app


def get_copick(settings: Settings) -> FastAPI:
    """Get the Copick app."""
    config_path = settings.CONFIG
    dataset_ids = settings.DATASET_IDS

    if config_path and dataset_ids:
        raise ValueError(
            "Either config_path or dataset_ids must be provided, but not both."
        )
    elif config_path:
        root = copick.from_file(config_path)
    elif dataset_ids:
        print(f"Loading datasets: {dataset_ids}")
        root = copick.from_czcdp_datasets(
            dataset_ids=dataset_ids,
            overlay_root=settings.OVERLAY_ROOT,
            overlay_fs_args={"auto_mkdir": True},
        )
        print("finished loading datasets")
    else:
        raise ValueError("Either config_path or dataset_ids must be provided.")

    return root


settings = Settings.load()
app = get_app(settings)
copick_root = get_copick(settings)
app.state.copick_root = copick_root
route_handler = CopickRoute(copick_root)


@app.get("/health/")
def health_check():
    """Health check endpoint."""
    return {"status": "ok"}


@app.get("/Picks")
async def get_picks(
    request: Request, run_id: str, user_id: str, session_id: str, name: str
):
    """Get the picks."""
    copick_root = request.app.state.copick_root
    copick_run = copick_root.get_run(run_id)
    # picks = copick_run.get_picks(user_id=user_id, session_id=session_id, object_name=name)
    picks = copick_run.get_picks()

    return [pick.meta.dict() for pick in picks]


@app.put("/Picks")
async def put_picks(
    request: Request, run_id: str, user_id: str, session_id: str, name: str
):
    """Put the picks."""
    copick_root = request.app.state.copick_root
    copick_run = copick_root.get_run(run_id)
    # picks = copick_run.get_picks(user_id=user_id, session_id=session_id, object_name=name)
    picks = copick_run.new_picks(
        object_name=name, user_id=user_id, session_id=session_id
    )
    data = await request.json()
    picks.meta = copick.models.CopickPicksFile(**data)
    picks.store()

    return [pick.meta.dict() for pick in picks]


@app.get("/Segmentations")
async def get_segmentations(
    request: Request,
    run_id: str,
    voxel_size: float,
    user_id: str,
    session_id: str,
    name: str,
    multilabel: bool = False,
):
    """Get the segmentations."""
    copick_root = request.app.state.copick_root
    copick_run = copick_root.get_run(run_id)
    segs = copick_run.get_segmentations(
        voxel_size=voxel_size,
        name=name,
        user_id=user_id,
        session_id=session_id,
        is_multilabel=multilabel,
    )

    # TODO: figure out what to return here
    return segs[0].zarr()


@app.put("/Segmentations")
async def put_segmentations(
    request: Request,
    run_id: str,
    voxel_size: float,
    user_id: str,
    session_id: str,
    name: str,
    multilabel: bool = False,
):
    """Put the segmentations."""
    copick_root = request.app.state.copick_root
    copick_run = copick_root.get_run(run_id)

    blob = await request.body()
    # Extract shape information (first 24 bytes contain 3 int64 values)
    shape = np.frombuffer(blob[:24], dtype=np.int64)

    # Extract the actual data and reshape it
    data = np.frombuffer(blob[24:], dtype=np.uint8).reshape(shape)

    segmentation(
        run=copick_run,
        segmentation_volume=data,
        user_id=user_id,
        name=name,
        session_id=session_id,
        voxel_size=voxel_size,
        multilabel=multilabel,
    )
    return


@app.get("/Tomograms")
async def get_tomograms(
    request: Request,
    run_id: str,
    voxel_size: float,
    tomo_type: str,
    user_id: str,
    session_id: str,
    name: str,
):
    """Get the tomograms."""
    copick_root = request.app.state.copick_root
    copick_run = copick_root.get_run(run_id)
    vs = copick_run.get_voxel_spacing(voxel_size)

    tomos = vs.get_tomograms(
        tomo_type=tomo_type,
    )
    body = tomos[0].zarr()[name]
    return Response(body, status_code=200)

SCALING_FACTOR = 10000  # TODO: add to settings

### embeddings datafile should look something like:
### X, Y, Z, run, dataset, protein, embeddings: tomotwin: [0.1, 0.2, 0.3, ...]

@app.post("/nearest_points")
async def nearest_points(point_query: NearestPointQuery):
    """Get the nearest points."""
    embeddings_table = ibis.read_parquet("data/embeddings.parquet")
    # TODO: handle embeddings better
    points = (
        embeddings_table.select(
            ["X", "Y", "Z", "run", "protein"] + [str(i) for i in range(32)]
        )
        .filter(embeddings_table["run"] == point_query.run_id)
        .to_pandas()
    )
    point_lookup = {}
    run_to_points_map = defaultdict(list)
    for _, point in points.iterrows():
        run_to_points_map[point.run].append([point.X, point.Y, point.Z])
        point_lookup[points_to_index(point.X, point.Y, point.Z)] = {
            "run_name": point.run,
            "prediction": point.protein,
            "location": (point.X, point.Y, point.Z),
            "embeddings": list(point[[str(i) for i in range(32)]]),
        }
    tree = PyRTree(dims=3)
    tree.bulk_load(run_to_points_map[point_query.run_id])
    k_nearest = tree.k_nearest_neighbors(
        [point_query.x, point_query.y, point_query.z], point_query.k_nearest
    )
    return [
        point_lookup[points_to_index(nearest[0], nearest[1], nearest[2])]
        for nearest in k_nearest
    ]


app.add_api_route(
    "/{path:path}",
    route_handler.handle_request,
    methods=["GET", "HEAD", "PUT"],
)
if __name__ == "__main__":
    config = uvicorn.Config(
        "server2:app", host=settings.HOST, port=settings.PORT, reload=True
    )
    server = uvicorn.Server(config)
    ChangeReload(config, target=server.run, sockets=[config.bind_socket()]).run()
