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
| `health` | Broken recordings: missing values, dropped frames, a robot that jumps in one frame, frozen or unreadable cameras, episodes far shorter than normal for their task |
| `motion` | Idle stretches, a command that jumps away and straight back, a gripper opening and closing repeatedly, commands held at the edge of their range, a robot far from where it was told to be. Also how smooth the motion is and how far the robot travelled, compared with other episodes of the same task |
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

## Benchmark (work in progress)

The benchmark asks whether training on the episodes a scorer keeps gives a better policy.
It trains a Diffusion Policy through LeRobot and tests it in the robosuite simulator on robomimic tasks.
It needs Python 3.12 or newer, and a GPU for anything beyond a smoke test. No results are published yet.

```bash
pip install "bowerbird[bench] @ git+https://github.com/darcywjoseph/bowerbird"

# Download a task and convert it to a LeRobot dataset.
bowerbird bench prepare lift mixed data

# Keep the shortest half, train on it, test the policy, write result.json.
bowerbird bench run data/lift_mixed runs/length --rule length --fraction 0.5 --steps 20000

# Damage a fifth of the episodes in known ways, then see what the scorers find.
bowerbird bench damage data/lift_mixed data/lift_damaged
bowerbird bench detect data/lift_damaged

# Run every condition for one task across five seeds, then write it up.
bowerbird bench round bench square --steps 100000
bowerbird bench report bench square
```

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
