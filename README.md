# Copick Server

A server for [copick](https://github.com/kephale/copick) projects.

## Installation

```bash
pip install .
```

## Usage

### Launch a server

#### Using a config file

```bash
python -m copick_server.server serve --config path/to/copick_config.json --cors "*" --port 8000
```

#### Using dataset IDs from CZ cryoET Data Portal

```bash
python -m copick_server.server serve --dataset-ids 10440 --cors "*" --port 8017
```

You can use multiple dataset IDs by repeating the option:

```bash
python -m copick_server.server serve --dataset-ids 10440 --dataset-ids 10441 --cors "*" --port 8017
```

#### Using uv run

With config file:
```bash
uv run copick_server/server.py serve --config ~/Data/copick/full_ml_challenge_czcdp.json --cors "*" --port 8017
```

With dataset IDs:
```bash
uv run copick_server/server.py serve --dataset-ids 10440 --cors "*" --port 8017
```

### Launch a client

```bash
uv run copick_server/client.py
```

## Running Tests

```bash
pip install ".[test]"
pytest
```

For coverage report:
```bash
pytest --cov=copick_server
```

## CLI Options

```
python -m copick_server.server serve --help
```

| Option            | Description                                            | Default       |
|-------------------|--------------------------------------------------------|---------------|
| -c, --config      | Path to the configuration file                         | None          |
| -ds, --dataset-ids| Dataset IDs to include in the project                  | None          |
| --overlay-root    | Root URL for the overlay storage                       | /tmp/overlay_root |
| --cors            | Origin to allow CORS. Use wildcard '*' to allow all    | None          |
| --host            | Bind socket to this host                               | 127.0.0.1     |
| --port            | Bind socket to this port                               | 8000          |
| --reload          | Enable auto-reload                                     | False         |

## References

The idea is influenced by https://github.com/manzt/simple-zarr-server, but aimed at supporting https://github.com/kephale/copick.
