# Isaac GR00T N1.7 — G1 Pick-and-Place Setup

This guide covers how to use Isaac GR00T N1.7 to fine-tune and serve a pick-and-place policy for the Unitree G1 robot. It is a companion to the [official README](https://github.com/NVIDIA/Isaac-GR00T) — installation, hardware requirements, full embodiment tag list, and TensorRT deployment are all documented there.

The simulation client that connects to this server lives in [`../g1-manipulation-challenge`](../g1-manipulation-challenge/README.md).

---

## What Is GR00T N1.7?

An open-source Vision-Language-Action (VLA) model that takes camera frames, joint state, and a language instruction as input and predicts a sequence of joint actions. The architecture combines a Cosmos-Reason2-2B vision-language backbone with a flow-matching diffusion transformer (DiT) action head:

```
Camera frames   → vision encoder ──┐
Language prompt → language encoder ─┤→ DiT (32 layers, 4 denoising steps) → action chunk (40 × D)
Joint state     → embodiment MLP ───┘
```

The diffusion head produces smooth continuous joint targets — no quantization artifacts. Per-embodiment MLPs are lightweight adapters so the large backbone transfers across robots without full retraining.

For the full architecture description see [README.md](https://github.com/NVIDIA/Isaac-GR00T#nvidia-isaac-gr00t).

---

## Installation

Full instructions and hardware requirements: [README.md — Installation](https://github.com/NVIDIA/Isaac-GR00T#installation)

```bash
# Python 3.10, dGPU (RTX / H100 / A100)
sudo apt-get install -y ffmpeg
uv sync --python 3.10

# If CUDA_HOME is unset during fine-tuning:
bash scripts/deployment/dgpu/install_deps.sh
```

---

## Embodiment Tag

Use **`REAL_G1`** (`real_g1_relative_eef_relative_joints`) — the pre-trained tag for the real Unitree G1 robot. This gives the best starting point for our dataset since the backbone already encodes G1 kinematics and scale.

For the full tag list see [README.md — Embodiment Tags](https://github.com/NVIDIA/Isaac-GR00T#model-checkpoints--embodiment-tags).

---

## Data Format

Full spec: [README.md — Data Format](https://github.com/NVIDIA/Isaac-GR00T#data-format)

Each inference step sends one observation to the server and receives an action chunk back:

**Request:**
```python
{
    "video": {
        "cam_left_high":  np.ndarray(1, 1, 480, 640, 3),   # (batch, timestep, H, W, C)
        "cam_right_high": np.ndarray(1, 1, 480, 640, 3),
    },
    "state": {
        "left_arm":   np.ndarray(1, 1, 7),   # shoulder pitch/roll/yaw, elbow, wrist roll/pitch/yaw
        "right_arm":  np.ndarray(1, 1, 7),
        "left_hand":  np.ndarray(1, 1, 7),   # thumb 0/1/2, middle 0/1, index 0/1
        "right_hand": np.ndarray(1, 1, 7),   # thumb 0/1/2, index 0/1, middle 0/1
    },
    "language": {
        "annotation.human.task_description": [["pick the red cube..."]]
    },
}
```

**Response:** `{ "left_arm": (40,7), "right_arm": (40,7), "left_hand": (40,7), "right_hand": (40,7) }` — absolute joint targets.

---

## Dataset

**Source:** [Unitree G1 Dex3 ObjectPlacement Dataset](https://huggingface.co/datasets/unitreerobotics/G1_Dex3_ObjectPlacement_Dataset) — 210 episodes, 98,266 frames of real G1 Dex3 manipulation.

### Preprocessing Pipeline

```
HuggingFace (LeRobot v3)
    │
    ├── convert_v3_to_v2.py          ← format conversion
    ├── recolor_blue_to_yellow.py    ← HSV hue shift: real blue container → yellow (matches sim)
    ├── language relabeling          ← "object_placement" → "pick the red cube and put it in the yellow box"
    └── split_dataset.py             ← 180 train / 30 test
```

### Timestamp Bug in convert_v3_to_v2.py

`scripts/lerobot_conversion/convert_v3_to_v2.py` has a critical misalignment: parquet action rows end at ~17.6 s while video runs to ~22 s. This leaves ~4.4 s of video frames with misaligned action labels per episode — the model learns to close the gripper before the hand reaches the object.

**Symptom:** Open-loop evaluation shows finger joints closing 2–3 frames earlier than ground truth, consistently across all test episodes.

**Fix:** Align timestamps at conversion time so action rows and video frames stay in sync. Apply the patch before running conversion from raw data.

---

## Fine-Tuning

Full guide: [README.md — Fine-tuning](https://github.com/NVIDIA/Isaac-GR00T#fine-tuning) | [getting_started/finetune_new_embodiment.md](getting_started/finetune_new_embodiment.md)

### Configuration

| Parameter | Value |
|-----------|-------|
| Base model | `nvidia/GR00T-N1.7-3B` |
| Embodiment tag | `REAL_G1` |
| Cameras | `cam_left_high`, `cam_right_high` |
| Input resolution | 224 × 224 |
| Action horizon | 16 steps |
| Max steps | 6,000 |
| Global batch size | 64 |
| Learning rate | 1e-4 |
| Optimizer | `adam_torch_fused` (≈15% faster on Blackwell GPUs vs AdamW) |

### Command

```bash
CUDA_VISIBLE_DEVICES=0 uv run python \
    gr00t/experiment/launch_finetune.py \
    --base-model-path nvidia/GR00T-N1.7-3B \
    --dataset-path /path/to/dataset \
    --embodiment-tag REAL_G1 \
    --modality-config-path examples/G1_Dex3/g1_dex3_config.py \
    --num-gpus 1 \
    --output-dir ./outputs/g1_checkpoint \
    --max-steps 2000 \
    --global-batch-size 64 \
    --dataloader-num-workers 8 \
    --use-wandb
```

### What to Tune

```python
config.model.tune_llm = True              # language backbone — important for language grounding
config.model.tune_diffusion_model = True  # action head — most important for manipulation quality
config.model.tune_visual = False          # keep frozen unless you have a large dataset
config.model.state_dropout_prob = 0.2    # lower = more reliant on proprio; higher = more visual
```

---

## Running the Policy Server

Full options: [README.md — Server-Client Inference](https://github.com/NVIDIA/Isaac-GR00T#server-client-inference-for-deployment)

```bash
# On the GPU machine
uv run python gr00t/eval/run_gr00t_server.py \
    --model-path ./outputs/g1_checkpoint \
    --embodiment-tag REAL_G1 \
    --device cuda:0
# Listens on 0.0.0.0:5555

# If the server is remote, open a CloudFlare tunnel on your local machine
bash run_client.sh   # forwards 127.0.0.1:5555 to the remote server
```

Then connect from the simulation — see [g1-manipulation-challenge/README.md](../g1-manipulation-challenge/README.md).

---

## Open-Loop Evaluation

Compare predicted actions against ground truth on held-out episodes:

```bash
uv run python gr00t/eval/open_loop_eval.py \
    --dataset-path /path/to/dataset \
    --embodiment-tag REAL_G1 \
    --model-path ./outputs/g1_checkpoint \
    --traj-ids 0 1 2 3 4 \
    --action-horizon 16 \
    --save-plot-path ./eval_plots
```

Plots saved as `traj_{id}.jpeg`. Things to check:
- Arm trajectories follow the right shape and direction
- Finger closure timing aligns with ground-truth grasps
- Early finger closure across all samples → likely the timestamp bug in data conversion

---

## Inference Performance

| Device | Mode | Latency | Hz |
|--------|------|---------|-----|
| H100 | PyTorch | 85.8 ms | 11.7 |
| H100 | TensorRT | 27.9 ms | 35.9 |
| Jetson Orin | PyTorch | 342.8 ms | 2.9 |
| Jetson Thor | TensorRT | 93.8 ms | 10.7 |

The simulation re-queries the server every 40 steps (EXEC_HORIZON), so ~1 Hz inference rate is sufficient for the control loop. For TensorRT acceleration see [scripts/deployment/README.md](scripts/deployment/README.md).

---

## Key Files

```
gr00t/
  eval/run_gr00t_server.py          # ZMQ policy server — start this on your GPU machine
  eval/open_loop_eval.py            # Open-loop evaluation against dataset
  experiment/launch_finetune.py     # Fine-tuning entry point
  policy/server_client.py           # ZMQ client protocol reference

examples/
  G1_Dex3/g1_dex3_config.py        # Modality config for Unitree G1 + Dex3 hands
  finetune.sh                       # Convenience wrapper for launch_finetune.py

scripts/
  lerobot_conversion/               # LeRobot v3 → v2 conversion (see timestamp bug above)
  deployment/                       # ONNX export, TensorRT build, platform Dockerfiles
```
