import os
os.environ["UNSLOTH_USE_FUSED_LOSS"] = "0"
os.environ["UNSLOTH_RETURN_LOGITS"] = "0"
os.environ["UNSLOTH_SKIP_TRANSFORMERS_DEPENDENCY_CHECK"] = "1"
os.environ["UNSLOTH_ENABLE_FIXING_UNTRAINED_TOKENS"] = "0"
os.environ["WANDB_DISABLED"] = "true"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["ACCELERATE_USE_CPU"] = "0"

import json, glob, random, sys
from pathlib import Path

import torch
import numpy as np
from sklearn.model_selection import train_test_split
from datasets import Dataset
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template, train_on_responses_only
from trl import SFTTrainer
from transformers import TrainingArguments
from google.colab import drive

CFG = dict(
    corpus_version   = "v2",                                    
    data_glob        = "/content/drive/MyDrive/New Data/*.json",
    out_dir          = "/content/drive/MyDrive/dentaltrain_runs/v2_masked",
    base_model       = "unsloth/llama-3-8b-Instruct-bnb-4bit",
    max_seq_length   = 2048,
    lora_r           = 16,
    lora_alpha       = 16,
    lora_dropout     = 0.0,
    target_modules   = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    test_size        = 0.15,   # was 0.10 at the conference; see note below
    seed             = 3407,
    epochs           = 3,
    batch_size       = 2,      # effective batch = batch_size * grad_accum = 8
    grad_accum       = 4,
    lr               = 2e-4,
    warmup_ratio     = 0.1,    # proportional, so both arms warm up equivalently
    weight_decay     = 0.01,
    scheduler        = "cosine",
    optim            = "adamw_8bit",
)

# NOTE ON test_size: the conference run used 0.10 (346/39). At 15 classes,
# 10% of 385 leaves ~2.6 conversations per class in validation, too few for
# the per-class metrics this extension reports. 0.15 is used for BOTH arms
# so the v1/v2 ablation stays clean; the conference figure is reported
# separately as a historical result, not as an arm of this comparison.

# IMPORTANT: to train the v1 ablation arm for RQ2, change ONLY
# corpus_version, data_glob and out_dir. Every other value must stay
# identical, otherwise the comparison confounds corpus with recipe.

SEED = CFG["seed"]
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

drive.mount('/content/drive')
Path(CFG["out_dir"]).mkdir(parents=True, exist_ok=True)

# ─── 4. MODEL
print(" Loading base model …")
print(" Loading base model …")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name     = CFG["base_model"],
    max_seq_length = CFG["max_seq_length"],
    dtype          = None,
    load_in_4bit   = True,
    device_map     = {"": 0},
)
tokenizer = get_chat_template(tokenizer, chat_template="llama-3")

model = FastLanguageModel.get_peft_model(
    model,
    r                         = CFG["lora_r"],
    target_modules            = CFG["target_modules"],
    lora_alpha                = CFG["lora_alpha"],
    lora_dropout              = CFG["lora_dropout"],
    bias                      = "none",
    use_gradient_checkpointing= "unsloth",
    random_state              = SEED,
)

# ─── 5. DATA
records = []
for fp in sorted(glob.glob(CFG["data_glob"])):
    if os.path.getsize(fp) == 0:
        print(f"   skipping empty file: {fp}")
        continue
    label = Path(fp).stem                     
    with open(fp, encoding="utf-8") as f:
        content = json.load(f)
    convos = content if isinstance(content, list) else [content]
    for c in convos:
        records.append({"messages": c["messages"], "diagnosis": label})

assert records, f"No conversations loaded from {CFG['data_glob']}"

def formatting_prompts_func(examples):
    return {"text": [
        tokenizer.apply_chat_template(c, tokenize=False, add_generation_prompt=False)
        for c in examples["messages"]
    ]}

dataset = Dataset.from_list(records).map(formatting_prompts_func, batched=True)
labels  = dataset["diagnosis"]

from collections import Counter
dist = Counter(labels)
print(f"\n Corpus: {len(dataset)} conversations across {len(dist)} classes")
for k, v in sorted(dist.items()):
    print(f"     {k:34s} {v:3d}")

assert len(dist) >= 15, (
    f"Expected >=15 diagnostic classes, got {len(dist)}. "
    "Stratification would be meaningless — check the data_glob and filenames."
)
imbalance = max(dist.values()) / min(dist.values())
print(f"     imbalance ratio (max/min): {imbalance:.2f}:1")

train_idx, val_idx = train_test_split(
    list(range(len(dataset))),
    test_size    = CFG["test_size"],
    random_state = SEED,
    stratify     = labels,
)
train_dataset = dataset.select(train_idx)
val_dataset   = dataset.select(val_idx)

tr_dist, va_dist = Counter(train_dataset["diagnosis"]), Counter(val_dataset["diagnosis"])
missing = set(dist) - set(va_dist)
assert not missing, f"Classes absent from validation split: {sorted(missing)}"
print(f"\n Train: {len(train_dataset)} | Val: {len(val_dataset)} "
      f"| classes in val: {len(va_dist)}")

# ─── 6. TRAINER
trainer = SFTTrainer(
    model             = model,
    tokenizer         = tokenizer,
    train_dataset     = train_dataset,
    eval_dataset      = val_dataset,
    dataset_text_field= "text",
    max_seq_length    = CFG["max_seq_length"],
    dataset_num_proc  = 2,
    packing           = False,
    args = TrainingArguments(
        per_device_train_batch_size = CFG["batch_size"],
        gradient_accumulation_steps = CFG["grad_accum"],
        num_train_epochs            = CFG["epochs"],
        warmup_ratio                = CFG["warmup_ratio"],
        learning_rate               = CFG["lr"],
        fp16                        = True,  
        bf16                        = False,
        logging_steps               = 1,
        optim                       = CFG["optim"],
        weight_decay                = CFG["weight_decay"],
        lr_scheduler_type           = CFG["scheduler"],
        seed                        = SEED,
        data_seed                   = SEED,
        output_dir                  = CFG["out_dir"],
        eval_strategy               = "epoch",
        save_strategy               = "epoch",
        save_total_limit            = 1,
        load_best_model_at_end      = True,
        metric_for_best_model       = "eval_loss",
        greater_is_better           = False,
        report_to                   = "none",
    ),
)

trainer = train_on_responses_only(
    trainer,
    instruction_part = "<|start_header_id|>user<|end_header_id|>\n\n",
    response_part    = "<|start_header_id|>assistant<|end_header_id|>\n\n",
)

print("\n" + "=" * 70)
print(" MASK VERIFICATION")
print("=" * 70)

ex = trainer.train_dataset[0]
ids  = torch.tensor(ex["input_ids"])
labs = torch.tensor(ex["labels"])

n_total  = len(labs)
n_masked = int((labs == -100).sum())
pct      = 100 * n_masked / n_total

kept_ids = ids[labs != -100]
trained_text = tokenizer.decode(kept_ids, skip_special_tokens=True)

print(f"tokens total     : {n_total}")
print(f"tokens masked    : {n_masked}  ({pct:.1f}%)")
print(f"tokens trained on: {n_total - n_masked}")
print("\n--- text the model IS trained to produce (first 600 chars) ---")
print(trained_text[:600])
print("---------------------------------------------------------------\n")

assert n_masked > 0, (
    " NOTHING WAS MASKED. train_on_responses_only did not take effect — "
    "check that instruction_part/response_part match the llama-3 template."
)
assert 30 < pct < 95, (
    f" Masked fraction {pct:.1f}% is implausible. Expected roughly 40–80% "
    "for this corpus. Inspect the decoded text above."
)

DIAG_TERMS = ["pericoronitis", "pulpitis", "periodontitis", "necrosis",
              "caries", "abscess", "sialolith", "neuralgia", "otitis",
              "temporomandibular"]
leaks = [t for t in DIAG_TERMS if t in trained_text.lower()]
if leaks:
    print(f"  Diagnosis terms present in trained text: {leaks}")
    print("    If these come from ASSISTANT turns, the corpus itself leaks the")
    print("    diagnosis and needs auditing. If from USER turns, masking failed.")
else:
    print(" No diagnosis terms in the trained (assistant-only) text.")


leak_count = 0
for i in range(min(20, len(trainer.train_dataset))):
    e = trainer.train_dataset[i]
    t = tokenizer.decode(
        torch.tensor(e["input_ids"])[torch.tensor(e["labels"]) != -100],
        skip_special_tokens=True).lower()
    if any(d in t for d in DIAG_TERMS):
        leak_count += 1
print(f"\nDiagnosis-term leakage across 20 sampled examples: {leak_count}/20")
print("=" * 70)

input("\n  Review the verification above, then press ENTER to train…")

print(" Training …")
stats = trainer.train()

save_path = Path(CFG["out_dir"]) / "adapter"
model.save_pretrained(str(save_path))
tokenizer.save_pretrained(str(save_path))

log = [h for h in trainer.state.log_history if "eval_loss" in h]
CFG["results"] = {
    "n_train": len(train_dataset),
    "n_val": len(val_dataset),
    "n_classes": len(dist),
    "class_distribution": dict(sorted(dist.items())),
    "imbalance_ratio": round(imbalance, 3),
    "masked_token_pct": round(pct, 2),
    "eval_loss_history": [{"epoch": h.get("epoch"), "step": h.get("step"),
                           "eval_loss": h["eval_loss"]} for h in log],
    "train_loss_history": [{"step": h.get("step"), "loss": h["loss"]}
                           for h in trainer.state.log_history if "loss" in h],
    "best_eval_loss": min((h["eval_loss"] for h in log), default=None),
    "train_runtime_s": getattr(stats, "metrics", {}).get("train_runtime"),
    "gpu": torch.cuda.get_device_name(0),
    "torch": torch.__version__,
    "python": sys.version.split()[0],
}
with open(Path(CFG["out_dir"]) / "run_config.json", "w") as f:
    json.dump(CFG, f, indent=2)

print(f"\n Adapter saved to {save_path}")
print(f" Run config saved to {CFG['out_dir']}/run_config.json")
print(f" best eval_loss: {CFG['results']['best_eval_loss']}")
