# bowerbird

[![CI](https://github.com/darcywjoseph/bowerbird/actions/workflows/ci.yml/badge.svg)](https://github.com/darcywjoseph/bowerbird/actions/workflows/ci.yml)

Score robot demonstrations and turn the scores into a ready-to-train dataset.

**WIP** 

## Quickstart

Score a LeRobot v3 dataset, from a local folder or the Hugging Face Hub. No GPU needed.

```bash
pip install git+https://github.com/darcywjoseph/bowerbird

bowerbird score lerobot/pusht       # writes scores and bowerbird-report.html
bowerbird show lerobot/pusht --worst 20
```

`score` runs three scorers and prints the lowest-scoring episodes with reasons:

| Scorer | What it looks for |
| --- | --- |
| `health` | Broken recordings: missing values, dropped frames, frozen or unreadable cameras |
| `motion` | Idle stretches, a command that jumps away and straight back, a gripper opening and closing repeatedly, commands held at the edge of their range, a robot far from where it was told to be |
| `length` | Episode length next to others on the same task. A baseline to compare the others with |

A fourth, `random`, is a control: add it with `--scorers health,length,motion,random`.
The camera checks decode every video, and for a Hub dataset download them; `--no-video` skips that.

The same in Python:

```python
import bowerbird

dataset = bowerbird.load("lerobot/pusht")
scores = bowerbird.score(dataset)

for episode in scores.worst(5):
    print(episode.index, round(episode.score, 2), episode.reasons)

keep = scores.keep_episodes(fraction=0.8)  # pass to LeRobotDataset(..., episodes=keep)
```

These scores are rule-based and have not yet been tested against trained-model results.

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
