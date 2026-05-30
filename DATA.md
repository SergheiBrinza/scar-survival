# Data and Models

This repo does not require any external datasets. All trap-facts, prompts, and per-experiment fact lists are checked into the corresponding subfolders.

You do need three HuggingFace models — one judge that is reused everywhere, and three subject models (one of which doubles as the judge).

## Models

| Role                      | HuggingFace ID                     | Used in                                                   |
|---------------------------|------------------------------------|-----------------------------------------------------------|
| Subject (main)            | `Qwen/Qwen2.5-3B-Instruct`         | exp1 subject, exp5 subject, exp4 subject (one of three)   |
| Judge + larger subject    | `Qwen/Qwen2.5-7B-Instruct`         | judge in all experiments, exp4 subject (one of three)     |
| Cross-family subject      | `microsoft/Phi-3.5-mini-instruct`  | exp4 subject (one of three)                               |

## Fetching the models

We use the official `huggingface_hub` CLI. Install once:

```
pip install -U "huggingface_hub[cli]"
```

Then download each model into your local HF cache:

```
hf download Qwen/Qwen2.5-3B-Instruct
hf download Qwen/Qwen2.5-7B-Instruct
hf download microsoft/Phi-3.5-mini-instruct
```

## Note on transformers 5.x and Phi-3.5

If you run on `transformers >= 5.0`, load `microsoft/Phi-3.5-mini-instruct` **without** `trust_remote_code=True`. The shipped `modeling_phi3.py` calls a `DynamicCache` method that was removed in 5.x, so the stock HF implementation is the correct path. The included exp4 scripts already do this.

## No external datasets

All fact lists, prompts, and notebooks live in the experiment subfolders. Nothing else needs to be downloaded.
