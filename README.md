# bowerbird

[![CI](https://github.com/darcywjoseph/bowerbird/actions/workflows/ci.yml/badge.svg)](https://github.com/darcywjoseph/bowerbird/actions/workflows/ci.yml)

Score robot demonstrations and turn the scores into a ready-to-train dataset.

**Status: planning.** Nothing here is usable yet. The first release (v0.1) will
score a LeRobot dataset on a laptop with no GPU and write an HTML report.

## What it will do

- **Health report:** cameras out of sync, frozen frames, robot not following commands.
- **Quality scores:** per episode and per segment, each with a reason.
- **Coverage map:** which situations are over-recorded and which are thin.
- **Evidence:** a public benchmark of each scorer against the "keep the shortest
  episodes" and random baselines.

See the [product plan](docs/bowerbird.md) and the
[implementation plan](docs/Implementation-plan.md).

## Development

```bash
uv sync
uv run ruff check
uv run ruff format --check
uv run mypy
uv run pytest
```

## Licence

Apache 2.0. See [LICENSE](LICENSE).
