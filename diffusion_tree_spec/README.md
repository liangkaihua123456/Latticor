# Diffusion Lattice → SpecInfer Token Tree (A) + Timestep-as-Tree (B)

PyTorch implementation living beside the SpecInfer FlexFlow artifact.

## Environment notes (important)

Original SpecInfer C++ experiments use:

```bash
source /root/autodl-tmp/IEEE_Conference_smart/specinfer-ae-main/activate_specinfer.sh
# conda env: flexflow
```

The `flexflow` conda env currently ships **CPU-only PyTorch** (`torch.version.cuda is None`).
A+B code needs **CUDA PyTorch** on the RTX 6000D.

Options:
1. **Recommended on this machine:** run A+B with `/root/autodl-tmp/envs/spork/bin/python` (CUDA torch already works).
2. Or install CUDA torch into flexflow:
   ```bash
   conda activate flexflow
   pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
   pip install accelerate sentencepiece
   ```

The launcher auto-picks flexflow if CUDA is available, otherwise spork:

```bash
bash run_diffusion_tree_experiments.sh smoke
bash run_diffusion_tree_experiments.sh quick
bash run_diffusion_tree_experiments.sh full
```

## Models (same-card draft + verify, no offload)

| Role | Model | Notes |
|------|-------|-------|
| Target | FlexFlow-converted LLaMA-7B | `hf_home/models/llama-7b-from-flexflow` (preferred) |
| Target (alt) | ModelScope `Qwen/Qwen2.5-7B-Instruct` | `bash download_models_modelscope.sh` |
| Draft | `JackFram/llama-68m` | Already cached |

RTX 6000D (~85GB) keeps draft+target resident → clean latency numbers.

**flexflow torch caveat:** `2.5.1+cu124` is a GPU build but does not fully support Blackwell `sm_120`. Launcher uses **spork** (`torch 2.11+cu130`) for A+B.

## Modes

| mode | meaning |
|------|---------|
| `incr` | Incremental decoding |
| `ar_seq` | AR sequence speculative draft |
| `ar_tree` | SpecInfer-style AR expansion tree |
| `diff_tree` | **A** lattice→tree + **B** adaptive NFE/width |

## Outputs

`FlexFlow/inference/output/diffusion_tree_spec/`

- `main_bs_sweep.csv` — BS / mode latency, alpha, draft/verify ms, tree nodes, speedup
- `ablation.csv` — top-k, prune width, fixed NFE vs adaptive B
