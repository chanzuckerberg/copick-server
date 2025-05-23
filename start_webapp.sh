#!/bin/bash

# This script is used to start the web application for the Copick project.
uv run copick_server/server.py serve --config example_copick.json --host 0.0.0.0 --port 8012
