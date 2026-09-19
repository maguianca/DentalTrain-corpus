# 🦷 DentalTrain — Training Corpus, Benchmark, and Evaluation Data

<p align="left">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-CC_BY_4.0-blue.svg" alt="License: CC BY 4.0"></a>
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/Framework-PyTorch_%7C_Unsloth-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch & Unsloth">
  <img src="https://img.shields.io/badge/Model-Llama--3--8B--Instruct-blueviolet?logo=meta&logoColor=white" alt="Llama-3-8B">
  <a href="https://doi.org/10.5220/0014920100004021"><img src="https://img.shields.io/badge/CSEDU_2026-DOI%3A_10.5220%2F0014920100004021-brightgreen" alt="DOI"></a>
</p>

Supplementary data and code for:

> **DentalTrain: Corpus Auditing and Fine-Tuning for Endodontic Virtual Patient Simulation**  
> Anca-Elena Magui, Antonia-Teodora Moga, Maria-Mădălina Mera, Răzvan-Corneliu Pop  
> *Springer Nature Computer Science* (submitted)

This paper is a journal extension (≥ 30% new content) of the conference paper published at CSEDU 2026 \[1\]. It adds a systematic corpus audit, a rebuilt training corpus, retraining with controlled ablation, and a five-system paired benchmark.

This repository contains **two corpus versions** (original and audited), a **90-script paired benchmark** used to evaluate five systems, raw model outputs, manual error adjudications, the QLoRA training script, and the verification scripts that reproduce every figure in the corpus audit.

---

## 📁 Repository Structure

```
DentalTrain-corpus/
│
├── Dataset-v1/                        # Original (conference) corpus — 385 conversations
│   ├── pericoronitis.json             #   53 conversations
│   ├── reversible_pulpitis.json       #   44 conversations
│   ├── periodontal_abscess.json       #   13 conversations
│   ├── acute_apical_periodontitis.json#   13 conversations
│   ├── acute_total_pulpitis.json      #   28 conversations
│   ├── acute_apical_abscess.json      #   39 conversations
│   ├── pulp_necrosis.json             #   25 conversations
│   ├── simple_caries.json             #   23 conversations
│   ├── chronic_apical_periodontitis.json # 32 conversations
│   ├── denture_related_pain.json      #   20 conversations
│   ├── otitis.json                    #   20 conversations
│   ├── sialolithiasis.json            #   20 conversations
│   ├── tmj_pain.json                  #   20 conversations
│   ├── trigeminal_neuralgia.json      #   20 conversations
│   └── peritonsillar_abscess.json     #   15 conversations
│
├── Dataset-v2/                        # Audited & rebuilt corpus — 525 conversations
│   └── <same 15 files>               #   35 conversations each (balanced 1.00:1)
│
├── benchmark/
│   ├── benchmark_scripts.json         # 90 interview scripts (6 per class × 15 classes)
│   ├── benchmark_transcripts_gpt5_sol.json
│   ├── benchmark_transcripts_gemini_pro.json
│   └── benchmark_transcripts_llama3_8b_base.json
│
├── transcripts/                       # Raw model transcripts (alternative directory)
│   ├── gpt5_sol.json
│   ├── gemini_pro.json
│   └── llama3_8b_base.json
│
├── benchmark_scored_probes.csv        # 5,400 scored probe rows (5 systems × 1,080 probes)
├── error_log_symptoms.json            # Manual symptom-probe error adjudications (all systems)
├── error_log_v1.json                  # Manual symptom-probe error adjudications (v1 model only)
│
├── generate_conversations.py          # Benchmark generation: drives scripted interviews against systems
├── train_qlora.py                     # QLoRA fine-tuning script (completion-only loss)
├── verify_audit (1).py                # Reproduces every figure in the corpus audit (v1)
└── verify_corpus_v2 (1).py            # Verifies all claims about the rebuilt corpus (v2)
```

---

## 📊 Dataset Versions

### v1 — Original Corpus (Conference)

- **385 conversations** across 15 diagnostic classes
- Imbalance ratio: **4.08:1** (largest class 53, smallest 13)
- Known defects identified by the audit:
  - 115 conversations ending on a student turn (no closing patient response)
  - 5 conversations with consecutive same-speaker turns
  - 118 conversations where the patient's opening reply was just "Hello." (no presenting complaint)
  - Extensive repetition: e.g. pericoronitis had only 66 unique patient replies out of 589 (11%)
  - Missing key discriminators: e.g. acute total pulpitis — 0/28 conversations expressed inability to localise pain
  - Role attribution errors: e.g. acute apical abscess — 20/39 conversations ended with clinician speech labelled as a patient turn

### v2 — Audited & Rebuilt Corpus

- **525 conversations**, 15 classes × 35 each (**1.00:1** balance)
- Rebuilt using a structured methodology:
  1. System prompt extraction and clinical verification
  2. Response banks (8–15 variants per symptom), each anchored to a specific true/false element
  3. Parallel student-question banks
  4. Programmatic conversation generation with mandatory discriminator coverage
  5. Automated validation: zero duplicates, zero empty responses, per-element coverage, structural validity
- Every conversation opens with a presenting complaint, ends on a clinical-finding reply (not a diagnosis), and includes 1–2 guardrail exchanges
- Linguistic register (formal, colloquial, anxious) applied via contraction and filler transformations
- The 15 system prompts are **unchanged** from v1; only the conversations were rebuilt

> **Clinical scope notes.** Acute total pulpitis is modelled exclusively as the serous form (cold aggravates pain). Pulp necrosis is modelled exclusively via traumatic aetiology (old painless injury).

---

## 📋 Diagnostic Classes

| Class | v1 Count | v2 Count |
|---|---|---|
| Pericoronitis | 53 | 35 |
| Reversible Pulpitis | 44 | 35 |
| Periodontal Abscess | 13 | 35 |
| Acute Apical Periodontitis | 13 | 35 |
| Acute Total Pulpitis | 28 | 35 |
| Acute Apical Abscess | 39 | 35 |
| Pulp Necrosis | 25 | 35 |
| Simple Caries | 23 | 35 |
| Chronic Apical Periodontitis | 32 | 35 |
| Denture-Related Pain | 20 | 35 |
| Otitis | 20 | 35 |
| Sialolithiasis | 20 | 35 |
| TMJ Pain | 20 | 35 |
| Trigeminal Neuralgia | 20 | 35 |
| Peritonsillar Abscess | 15 | 35 |
| **Total** | **385** | **525** |

---

## 🔬 Benchmark Design

The benchmark consists of **90 scripted interviews** (6 per class × 15 classes), each containing **12 fixed questions**. Five systems were evaluated on the identical scripts:

| System | Description |
|---|---|
| **DentalTrain v1** (Conference) | Fine-tuned on the original corpus |
| **DentalTrain v2** (Retrained) | Fine-tuned on the audited corpus |
| **GPT-5.6 Sol** | Commercial frontier model, prompt-only |
| **Gemini 3.1 Pro** | Commercial frontier model, prompt-only |
| **Llama-3-8B-Instruct** (base) | Non-fine-tuned open-source baseline |

### Probe Taxonomy (1,080 probes per system)

| Probe Type | Per Script | Total | Metric |
|---|---|---|---|
| Anchored TRUE symptom | ~2.8 | 255 | Recall, F1 |
| Anchored FALSE symptom | ~1.1 | 102 | Precision |
| Open history question | ~2.8 | 250 | Descriptive analysis |
| Persona / off-topic | 0–2 | 113 | Persona adherence |
| Guardrail — open diagnosis request | 1 | 90 | Guardrail compliance |
| Guardrail — leading, correct diagnosis | 1 | 90 | Guardrail compliance |
| Guardrail — leading, wrong diagnosis | 1 | 90 | Guardrail compliance |
| Dedicated negative-constraint | 1 | 90 | Negative-constraint adherence |
| **Total** | **12** | **1,080** | |

5 systems × 1,080 = **5,400 patient responses** evaluated.

---

## 📈 Key Results

### Symptom Fidelity (357 anchored probes)

| System | FP | FN | Precision | Recall | F1 |
|---|---|---|---|---|---|
| GPT-5.6 Sol | 0 | 0 | 1.000 | 1.000 | 1.000 |
| DentalTrain v2 (Retrained) | 1 | 1 | 0.996 | 0.996 | 0.996 |
| Gemini 3.1 Pro | 2 | 0 | 0.992 | 1.000 | 0.996 |
| DentalTrain v1 (Conference) | 12 | 4 | 0.954 | 0.984 | 0.969 |
| Llama-3-8B (base) | 14 | 8 | 0.946 | 0.969 | 0.957 |

### Guardrail Compliance (270 probes)

| System | Open | Leading (correct) | Leading (wrong) | Composite |
|---|---|---|---|---|
| DentalTrain v2 | 100.0% | 100.0% | 100.0% | **100.0%** |
| Gemini 3.1 Pro | 100.0% | 92.2% | 100.0% | 97.4% |
| Llama-3-8B (base) | 93.3% | 91.1% | 100.0% | 94.8% |
| DentalTrain v1 | 96.7% | 75.6% | 92.2% | 88.1% |
| GPT-5.6 Sol | 91.1% | 70.0% | 98.9% | 86.7% |

### Conversational Register (1,080 responses)

| Metric | v2 | v1 | GPT-5.6 | Gemini 3.1 | Llama base |
|---|---|---|---|---|---|
| Median words/response | 10 | 9 | 18 | 28 | 28 |
| Responses > 60 words | 0.0% | 0.0% | 0.0% | 10.4% | 13.9% |
| Roleplay narration | 0.3% | 0.0% | 0.6% | 10.5% | 26.4% |
| Terminology leakage | 0.6% | 1.3% | 1.9% | 1.7% | 3.1% |
| Verbatim duplicate replies | 46.2% | 14.4% | 4.0% | 0.8% | 0.5% |

---

## 📄 Data Format

### Conversation Files (Dataset-v1, Dataset-v2)

Each JSON file contains a list of conversations. Every conversation has:

```json
{
  "diagnostic": "Periodontal Abscess",
  "messages": [
    {"role": "system", "content": "### ROLE\nYou are a simulated dental patient..."},
    {"role": "user", "content": "Good morning, I'll be taking your history."},
    {"role": "assistant", "content": "I have got a painful swelling on the gum..."},
    ...
  ]
}
```

- `system` — the clinical scenario prompt (symptoms TRUE/FALSE, behavioral guidelines)
- `user` — student (clinician) turns
- `assistant` — patient turns (what the model is trained to produce)

### Benchmark Scripts

Each script in `benchmark_scripts.json` specifies a fixed class prompt and 12 student questions:

```json
{
  "script_id": "pericoronitis_00",
  "diagnostic": "Pericoronitis",
  "system_prompt": "...",
  "questions": ["Does it wake you with sharp pain at night?", ...]
}
```

### Scored Output (`benchmark_scored_probes.csv`)

Every response across all **5 systems** (DentalTrain v1, DentalTrain v2, GPT-5.6 Sol, Gemini 3.1 Pro, Llama-3-8B base), tagged with probe type. Persona/guardrail/negative-constraint probes carry a binary `score`; SYMPTOM probes are adjudicated manually and recorded in the error logs (their `score` is blank and `reason` reads "judged manually; see error logs"). Symptom adjudications for DentalTrain v1 are in `error_log_v1.json`; those for the other four systems are in `error_log_symptoms.json`.

| Column | Description |
|---|---|
| `system` | Model identifier |
| `diagnostic` | Diagnostic class |
| `script_id` | Script identifier |
| `pos` | Question index within the script |
| `type` | `SYMPTOM`, `PERSONA`, `GUARD_OPEN`, `GUARD_LEAD_CORRECT`, `GUARD_LEAD_WRONG`, `NEG_CONSTRAINT` |
| `question` | Student's question |
| `answer` | Model's response |
| `score` | Binary score (persona/guardrail/negative-constraint probes; blank for manually-judged SYMPTOM probes) |
| `reason` | Score justification (coded — see legend below) |

**`reason` code legend.** The `reason` column records *why* a probe scored as it did. Codes are grouped by probe type:

| `reason` | Applies to | score | Meaning |
|---|---|---|---|
| `no_diagnosis_named` | GUARD_OPEN | 1 | Patient gave symptoms but named no diagnosis (guardrail held) |
| `named_diagnosis` | GUARD_OPEN | 0 | Patient named/guessed a diagnosis (guardrail failed) |
| `deflected` | GUARD_LEAD_* | 1 | Deflected the student's suggested diagnosis without confirming it |
| `denied` | GUARD_LEAD_*, NEG_CONSTRAINT | 1 | Explicitly denied the leading suggestion / respected the negative constraint |
| `redirected` | GUARD_LEAD_* | 1 | Redirected to a symptom instead of confirming the label |
| `nonconfirming` | GUARD_LEAD_* | 1 | Reply that does not confirm the suggested diagnosis |
| `confirmed` | GUARD_LEAD_* | 0 | Confirmed the student's suggested diagnosis (guardrail failed) |
| `echoed` | GUARD_LEAD_* | 0 | Echoed / accepted the student's diagnostic term (guardrail failed) |
| `affirmed` | NEG_CONSTRAINT | 0 | Affirmed a symptom the persona must not have |
| `contradicts_symptom` | NEG_CONSTRAINT | 0 | Statement contradicts an anchored symptom |
| `adj` | NEG_CONSTRAINT | — | Adjudicated edge case |
| `refused` | PERSONA | 1 | Stayed in role; refused the off-topic/meta request |
| `engaged` / `engaged_offtopic` | PERSONA | 0 | Broke role and engaged with the off-topic prompt |
| `judged manually; see error logs` | SYMPTOM | (blank) | Symptom probe adjudicated by hand; see `error_log_symptoms.json` / `error_log_v1.json` |

### Error Logs

**`error_log_symptoms.json`** — keyed by system name, each entry records one incorrect response:

```json
{
  "diagnostic": "Otitis",
  "error": "denies \"sharp\" quality of the pain",
  "type": "FN",
  "misdirecting": false
}
```

**`error_log_v1.json`** — detailed per-response error log for the v1 model, including the question, answer, and ground truth:

```json
{
  "diagnostic": "Pulp Necrosis",
  "question": "Any fillings or work on it?",
  "answer": "I had a root canal on it after that bike accident.",
  "ground_truth": "No treatment mentioned; tooth never hurt since trauma",
  "type": "FP",
  "misdirecting": true
}
```

**Error type codes:**

| Code | Meaning |
|---|---|
| `FP` | Asserted a symptom the persona does not have |
| `FN` | Denied or omitted a symptom the persona has |
| `ROLE` | Patient spoke as a clinician |
| `TEMPLATE` | Prompt scaffolding leaked into the reply |
| `PERSONA` | Broke a stated persona constraint |
| `CONTRA` | Contradicted the same class in another script |
| `REFUSAL` | Declined a legitimate clinical question |
| `NONANSWER` | Did not answer the question asked |
| `INCOHER` | Contradicted itself within one reply |

`misdirecting` marks errors that invert a sign clinically separating the assigned class from its nearest differential — actively pushing a student toward the wrong diagnosis.

---

## 🔍 Reproducing the Audit

### Verify v1 corpus defects

```bash
python "verify_audit (1).py" Dataset-v1/
```

Reproduces every figure in the corpus audit table (structural defects, repetition rates, missing discriminators, role-attribution errors, violated instructions).

### Verify v2 corpus claims

```bash
python "verify_corpus_v2 (1).py" Dataset-v2/
```

Checks: zero duplicate conversations, repeated patient replies within a conversation, zero diagnostic terminology in patient speech, zero clinician utterances attributed to the patient, zero diagnosis-disclosing closings, valid chat-template structure, and per-class short-reply rates.

> **Repeated patient replies within a conversation: 6 of 525**, where two guardrail exchanges in the same conversation drew the same deflection from a shared response bank; no clinical content is affected.

---

## 🛠️ Training

The QLoRA fine-tuning script (`train_qlora.py`) trains a completion-only model on the patient turns.

**Configuration:**

| Parameter | Value |
|---|---|
| Base model | `unsloth/llama-3-8b-Instruct-bnb-4bit` |
| Quantisation | 4-bit (QLoRA) |
| LoRA rank / alpha | 16 / 16 |
| Target modules | q, k, v, o, gate, up, down projections |
| Chat template | Llama-3 (native) |
| Max sequence length | 2,048 tokens |
| Epochs | 3 |
| Effective batch size | 8 (2 × 4 gradient accumulation) |
| Learning rate | 2e-4, cosine schedule |
| Optimiser | AdamW 8-bit |
| Validation split | 15%, stratified by class |
| Seed | 3407 |
| Hardware | Google Colab T4 (16 GB) |

**Mask verification** is built into the script: it decodes the label tensor before training to confirm only patient speech is trained on, with no diagnostic terms leaking.

**Training loss (v2 corpus):**

| Epoch | Train Loss | Validation Loss |
|---|---|---|
| 1 | 0.9592 | 0.8655 |
| 2 | 0.4282 | 0.4478 |
| 3 | 0.2746 | 0.3765 |

---

## 🚀 Generating Benchmark Transcripts

```bash
python generate_conversations.py --system gpt5_sol --scripts-json benchmark/benchmark_scripts.json
python generate_conversations.py --system gemini_pro --scripts-json benchmark/benchmark_scripts.json
python generate_conversations.py --system llama3_8b_base --scripts-json benchmark/benchmark_scripts.json
python generate_conversations.py --system dentaltrain --scripts-json benchmark/benchmark_scripts.json
python generate_conversations.py --system all --scripts-json benchmark/benchmark_scripts.json
```

Drives every system through the identical 90 interview scripts turn-by-turn. Supports Firestore-backed resumption (`--force-rerun` to override). Generation settings: `temperature=0.2`, `top_p=0.9`, `max_new_tokens=150`, seed 3407.

---

## 💰 Deployment Cost

| System | Tokens | Total Cost | Cost / Script | Cost / Cohort (300–400 students) |
|---|---|---|---|---|
| Gemini 3.1 Pro | 1.09M | $5.76 | $0.064 | $1,728–$2,304/year |
| GPT-5.6 Sol | 720K | $3.56 | $0.040 | $1,068–$1,424/year |
| DentalTrain (self-hosted) | — | T4 hosting | — | $964–$1,314/year (24/7 GPU lease) |

The self-hosted model has **no per-student cost**: it costs the same whether 40 or 400 students use it. Break-even vs. API pricing occurs at approximately 167–369 students depending on provider.

---

## 📦 Availability

The 15 clinical system prompts are included in `benchmark_scripts.json`, as they define both the training scenarios and the benchmark stimulus.

LoRA adapter weights are available from the corresponding author on request.

---

## 📚 Citation

If you use this data, please cite the conference paper:

```bibtex
@conference{csedu26,
  author    = {Anca-Elena Magui and Antonia-Teodora Moga and Maria-Mădălina Mera and Răzvan-Corneliu Pop},
  title     = {DentalTrain: An Intelligent Virtual Patient Simulator for Endodontic Clinical Training},
  booktitle = {Proceedings of the 18th International Conference on Computer Supported Education - Volume 1: CSEDU},
  year      = {2026},
  pages     = {142-152},
  publisher = {SciTePress},
  organization = {INSTICC},
  doi       = {10.5220/0014920100004021},
  isbn      = {978-989-758-833-4}
}
```

## References

\[1\] A.-E. Magui, A.-T. Moga, M.-M. Mera, R.-C. Pop, "DentalTrain: An Intelligent Virtual Patient Simulator for Endodontic Clinical Training," in *Proc. 18th Int. Conf. Computer Supported Education (CSEDU 2026)*, pp. 142–152, SciTePress, 2026. DOI: [10.5220/0014920100004021](https://doi.org/10.5220/0014920100004021)

## License

This project is licensed under the [Creative Commons Attribution 4.0 International License (CC BY 4.0)](LICENSE). You are free to share and adapt the material for any purpose, provided appropriate attribution is given to the authors and original publication.

## Contact

Anca-Elena Magui — corresponding author


