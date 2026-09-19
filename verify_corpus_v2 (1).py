#!/usr/bin/env python3
"""
Verifies every claim made about the rebuilt corpus in Section 4.3.

Usage:
    python3 verify_corpus_v2.py /path/to/corpus_v3_guardrails

Each check prints PASS or FAIL with the count, so the paragraph in the paper
can be confirmed line by line.
"""
import json, re, sys, glob, os
from collections import Counter

ROOT = sys.argv[1] if len(sys.argv) > 1 else '.'
files = sorted(glob.glob(os.path.join(ROOT, '*.json')))
if not files:
    sys.exit(f'no .json files found in {ROOT}')

corpus = []            # (class, conversation)
for f in files:
    cls = os.path.basename(f)[:-5]
    data = json.load(open(f, encoding='utf-8'))
    data = data if isinstance(data, list) else [data]
    for c in data:
        corpus.append((cls, c))


def patient_turns(c):
    return [m['content'].strip() for m in c['messages'] if m['role'] == 'assistant']


def student_turns(c):
    return [m['content'].strip() for m in c['messages'] if m['role'] == 'user']


def report(label, bad, total, detail=None):
    tag = 'PASS' if bad == 0 else 'FAIL'
    print(f'  [{tag}] {label:52s} {bad}/{total}')
    if bad and detail:
        for d in detail[:4]:
            print(f'         {d}')


print('=' * 72)
print(f'REBUILT CORPUS: {len(corpus)} conversations across {len(files)} classes')
print('=' * 72)

# ---- class balance ---------------------------------------------------
dist = Counter(cls for cls, _ in corpus)
print(f'\nclasses: {len(dist)} | per class: {sorted(set(dist.values()))} '
      f'| imbalance {max(dist.values())/min(dist.values()):.2f}:1')

# ---- 1. duplicate conversations --------------------------------------
sigs = [tuple(m['content'] for m in c['messages']) for _, c in corpus]
dups = len(sigs) - len(set(sigs))
report('duplicate conversations', dups, len(corpus))

# ---- 2. repeated response within one conversation --------------------
bad, ex = 0, []
for cls, c in corpus:
    a = patient_turns(c)
    if len(a) != len(set(a)):
        bad += 1
        rep = [t for t, n in Counter(a).items() if n > 1][:1]
        ex.append(f'{cls}: "{rep[0][:60]}"' if rep else cls)
report('conversations with a repeated patient reply', bad, len(corpus), ex)

# ---- 3. medical terminology in patient speech ------------------------
TERMS = (r'pericoronitis|pulpitis|periodontitis|necrosis|necrotic|sialolith|'
         r'neuralgia|otitis|stomatitis|temporomandibular|\btmj\b|quinsy|'
         r'peritonsillar|periapical|granuloma')
DX = re.compile(TERMS, re.I)
bad, ex = 0, []
for cls, c in corpus:
    for t in patient_turns(c):
        if DX.search(t):
            bad += 1
            ex.append(f'{cls}: "{t[:70]}"')
            break
report('conversations with a diagnostic term in patient speech', bad, len(corpus), ex)

# ---- 4. clinician utterance attributed to the patient ----------------
CLIN = re.compile(r"(we'?ll (drain|proceed|prescribe|start|treat)|"
                  r"i'?ll prescribe|we need urgent treatment|"
                  r"let'?s (drain|schedule|start treatment)|i recommend)", re.I)
bad, ex = 0, []
for cls, c in corpus:
    for t in patient_turns(c):
        if CLIN.search(t):
            bad += 1
            ex.append(f'{cls}: "{t[:70]}"')
            break
report('conversations where the patient speaks as clinician', bad, len(corpus), ex)

# ---- 5. closing that names the diagnosis -----------------------------
CLOSER = re.compile(r"(the (findings|diagnosis|symptoms|results)|this is|that'?s|"
                    r"points? to|indicates?|consistent with|i can confirm)", re.I)
bad, ex = 0, []
for cls, c in corpus:
    ms = [m for m in c['messages'] if m['role'] != 'system']
    if len(ms) >= 2:
        pen = ms[-2]
        if pen['role'] == 'user' and DX.search(pen['content']) and CLOSER.search(pen['content']):
            bad += 1
            ex.append(f'{cls}: "{pen["content"][:70]}"')
report('conversations closing on a diagnosis disclosure', bad, len(corpus), ex)

# ---- 6. opening without a presenting complaint -----------------------
COMPLAINT = re.compile(r'\b(pain|hurt|ache|sore|swell|swollen|tooth|teeth|gum|jaw|'
                       r'throb|sensitiv|bleed|problem|trouble|bother|sting|burn|'
                       r'discomfort|cold|hot|chew|bite|ear|face|cheek|filling|'
                       r'denture|lump|taste|swallow|gumboil|throat|tongue|hole|'
                       r'cavity|incisor|molar|dark|trapped|rough|meal|sharp|'
                       r'pressure|raw|chip|match|open|agony)', re.I)
bad, ex = 0, []
for cls, c in corpus:
    a = patient_turns(c)
    if a and not COMPLAINT.search(a[0]):
        bad += 1
        ex.append(f'{cls}: "{a[0][:70]}"')
report('openings with no presenting complaint', bad, len(corpus), ex)

# ---- 7. structural validity ------------------------------------------
bad, ex = 0, []
for cls, c in corpus:
    ms = c['messages']
    roles = [m['role'] for m in ms]
    ok = (roles[0] == 'system'
          and all(m.get('content', '').strip() for m in ms)
          and all(roles[i] != roles[i+1] for i in range(1, len(roles)-1)))
    if not ok:
        bad += 1
        ex.append(cls)
report('conversations with invalid structure', bad, len(corpus), ex)

# ---- 8. very short replies, per class --------------------------------
print('\n  short replies (<=4 words), by class:')
worst = 0
for cls in sorted(dist):
    a = [t for c_, c in corpus if c_ == cls for t in patient_turns(c)]
    short = sum(1 for t in a if len(t.split()) <= 4)
    pctv = 100 * short / len(a)
    worst = max(worst, pctv)
    flag = ' <-- above 5%' if pctv > 5 else ''
    print(f'    {cls:32s} {short:4d}/{len(a):4d}  {pctv:5.1f}%{flag}')
print(f'\n  highest per-class rate: {worst:.1f}%  '
      f'({"claim of <5% holds" if worst < 5 else "claim of <5% does NOT hold"})')

# ---- 9. length statistics --------------------------------------------
lens = [len(c['messages']) for _, c in corpus]
print(f'\n  messages per conversation: min {min(lens)}, median '
      f'{sorted(lens)[len(lens)//2]}, max {max(lens)}')
print()
