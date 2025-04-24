# Demo server using the dataset_ids parameter
from copick_server.server import serve_copick

# Use dataset_id instead of config file
serve_copick(dataset_ids=[14268], overlay_root="/tmp/overlay_root", allowed_origins=["*"])

# Previous config-based approach:
# serve_copick("/Users/kharrington/git/copick/copick-server/example_copick.json", allowed_origins=["*"])

# Command line usage:
# With config file:
# $ python -m copick_server.server serve --config path/to/copick_config.json --cors "*"
# 
# With dataset IDs:
# $ python -m copick_server.server serve --dataset-ids 14268 --cors "*" --port 8017
