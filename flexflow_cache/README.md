# flexflow_cache_diffusion_tree_spec

Slim stand-in for the FlexFlow model cache used by the diffusion-lattice token-tree experiments. The original cache is about 38 GB of half-precision weights, which is too large for GitHub. This folder keeps only the Hugging Face `config.json` files and a script that downloads the weights.

Committed configs:

| Model | Role | Config |
|---|---|---|
| `JackFram/llama-68m` | candidate | `configs/jackfram/llama-68m/config.json` |
| `facebook/opt-125m` | candidate | `configs/facebook/opt-125m/config.json` |
| `huggyllama/llama-7b` | target | `configs/huggyllama/llama-7b/config.json` |
| `facebook/opt-13b` | target | `configs/facebook/opt-13b/config.json` |

Revisions are pinned in `models.json` to the commits recorded in the original cache.

## Download

Disk needed: about 40 GB. RAM needed: enough to hold OPT-13B in float16 (about 26 GB) while it is converted. LLaMA weights are released under the original LLaMA license; accept that license on Hugging Face if the download asks for a token.

```bash
bash download_models.sh
```

Files land in `./downloaded/`, which is gitignored:

```
downloaded/configs/<org>/<name>/config.json
downloaded/tokenizers/<org>/<name>/
downloaded/weights/<org>/<name>/half-precision/
```

Choose another directory, or download one model:

```bash
FF_CACHE_PATH=/data/flexflow_cache bash download_models.sh
bash download_models.sh --models JackFram/llama-68m facebook/opt-125m
```

If `huggingface.co` is slow:

```bash
HF_ENDPOINT=https://hf-mirror.com bash download_models.sh
```

Tensor filenames match FlexFlow Serve's half-precision cache (`tok_embeddings_weight`, `layers_0_attention_wq_weight`, and so on), so SpecInfer and the existing FlexFlow-to-HuggingFace converters can read the result. This script does not build or run FlexFlow.
