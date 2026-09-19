#!/usr/bin/env python3
"""
Reproduces every figure in the corpus audit table.

Usage:
    python3 verify_audit.py /path/to/Toate_Bolile

Each check prints the class, the metric, and the value. Run it against the
385-conversation corpus to confirm the numbers reported in the paper.
"""
import json, re, sys, glob, os
from collections import Counter, defaultdict

ROOT = sys.argv[1] if len(sys.argv) > 1 else '.'


def load(name):
    path = os.path.join(ROOT, f'{name}.json')
    d = json.load(open(path, encoding='utf-8'))
    return d if isinstance(d, list) else [d]


def turns(convs, role):
    return [m['content'].strip() for c in convs
            for m in c['messages'] if m['role'] == role]


def convs_matching(convs, pattern, role='assistant', first_only=False):
    """Count conversations with at least one matching turn of the given role."""
    rx = re.compile(pattern, re.I)
    n = 0
    for c in convs:
        ts = [m['content'] for m in c['messages'] if m['role'] == role]
        if first_only:
            ts = ts[:1]
        if any(rx.search(t) for t in ts):
            n += 1
    return n


def pct(a, b):
    return f'{100*a/b:.1f}%' if b else 'n/a'


print('=' * 68)
print('CORPUS SIZE')
print('=' * 68)
total = 0
for f in sorted(glob.glob(os.path.join(ROOT, '*.json'))):
    d = json.load(open(f, encoding='utf-8'))
    d = d if isinstance(d, list) else [d]
    total += len(d)
    print(f'  {os.path.basename(f)[:-5]:32s} {len(d):3d}')
print(f'  {"TOTAL":32s} {total:3d}')

print()
print('=' * 68)
print('STRUCTURAL DEFECTS (whole corpus)')
print('=' * 68)
end_user = same_role = hello = 0
for f in sorted(glob.glob(os.path.join(ROOT, '*.json'))):
    d = json.load(open(f, encoding='utf-8'))
    d = d if isinstance(d, list) else [d]
    for c in d:
        ms = [m for m in c['messages'] if m['role'] != 'system']
        if ms[-1]['role'] == 'user':
            end_user += 1
        if any(ms[i]['role'] == ms[i+1]['role'] for i in range(len(ms)-1)):
            same_role += 1
        a = [m for m in ms if m['role'] == 'assistant']
        if a and a[0]['content'].strip() == 'Hello.':
            hello += 1
print(f'  conversations ending on a student turn       : {end_user}')
print(f'  conversations with two consecutive same-role  : {same_role}')
print(f'  patient opening turn is exactly "Hello."      : {hello}')

print()
print('=' * 68)
print('PER-CLASS CHECKS')
print('=' * 68)

# --- REPETITION -------------------------------------------------------
for name in ['pericoronitis', 'reversible_pulpitis', 'chronic_apical_periodontitis']:
    d = load(name)
    A = turns(d, 'assistant')
    c = Counter(A)
    short = [t for t in A if len(t.split()) <= 4]
    print(f'\n[{name}]  n={len(d)}')
    print(f'  patient turns            : {len(A)}')
    print(f'  distinct                 : {len(c)}  ({pct(len(c), len(A))})')
    print(f'  repeated share           : {pct(len(A)-len(c), len(A))}')
    print(f'  <=4 words                : {len(short)}  ({pct(len(short), len(A))})')
    print(f'  most frequent            : ' + '; '.join(
        f'{n}x "{t[:40]}"' for t, n in c.most_common(2)))

# chronic apical periodontitis: palpation denial
d = load('chronic_apical_periodontitis')
NEG = r'\b(no|not|nothing|doesn.t|don.t)\b'
PALP = r'\b(press\w*|palpat\w*|touch\w*|push\w*)\b'
n = sum(1 for c in d if any(re.search(PALP, m['content'], re.I)
                            and re.search(NEG, m['content'], re.I)
                            for m in c['messages'] if m['role'] == 'assistant'))
print(f'  palpation tenderness denied in {n}/{len(d)} conversations')

# --- MISSING DISCRIMINATORS -------------------------------------------
d = load('acute_total_pulpitis')
print(f'\n[acute_total_pulpitis]  n={len(d)}')
for label, pat in [
    ('non-localisation', r'\b(can.?t (tell|say|point|localis)|not sure which|whole side|radiat|spreads?|everywhere)\b'),
    ('nocturnal waking', r'\b(night|wakes? me|sleep|asleep)\b'),
    ('heat sensitivity', r'\b(hot|heat|warm|coffee|tea|soup)\b'),
]:
    print(f'  {label:20s}: {convs_matching(d, pat)}/{len(d)}')

d = load('sialolithiasis')
meal = convs_matching(d, r'\b(eat\w*|meal|food|lunch|dinner)\b')
res = convs_matching(d, r'\b(goes down|subsid\w+|settles|shrink\w*|reduc\w*)\b')
both = sum(1 for c in d
           if any(re.search(r'\b(eat\w*|meal|food)\b', m['content'], re.I)
                  for m in c['messages'] if m['role'] == 'assistant')
           and any(re.search(r'\b(goes down|subsid\w+|settles|shrink\w*|reduc\w*)\b',
                             m['content'], re.I)
                   for m in c['messages'] if m['role'] == 'assistant'))
print(f'\n[sialolithiasis]  n={len(d)}')
print(f'  meal trigger        : {meal}/{len(d)}')
print(f'  post-meal resolution: {res}/{len(d)}')
print(f'  both present        : {both}/{len(d)}')

d = load('peritonsillar_abscess')
A = turns(d, 'assistant')
terse = [t for t in A if len(t.split()) <= 3]
voice = convs_matching(d, r'\b(voice|muffl\w+|hot potato|sound\w* different|speak\w* funny|thick)\b')
print(f'\n[peritonsillar_abscess]  n={len(d)}')
print(f'  patient turns <=3 words : {len(terse)}/{len(A)}  ({pct(len(terse), len(A))})')
print(f'  voice change present    : {voice}/{len(d)}  ({pct(voice, len(d))})')

sc, rp = turns(load('simple_caries'), 'assistant'), turns(load('reversible_pulpitis'), 'assistant')
print(f'\n[simple_caries vs reversible_pulpitis]')
print(f'  distinct caries turns   : {len(set(sc))}')
print(f'  verbatim shared         : {len(set(sc) & set(rp))}')

# --- VIOLATED INSTRUCTIONS --------------------------------------------
d = load('otitis')
print(f'\n[otitis]  n={len(d)}')
n_ear = convs_matching(d, r'\bear\b', first_only=True)
print(f'  "ear" named in FIRST patient turn: {n_ear}/{len(d)}')

d = load('denture_related_pain')
n = 0
for c in d:
    ms = [m for m in c['messages'] if m['role'] != 'system']
    fp = next((i for i, m in enumerate(ms) if m['role'] == 'assistant'
               and re.search(r'denture|plate', m['content'], re.I)), None)
    fs = next((i for i, m in enumerate(ms) if m['role'] == 'user'
               and re.search(r'denture|plate', m['content'], re.I)), None)
    if fp is not None and (fs is None or fp < fs):
        n += 1
print(f'\n[denture_related_pain]  n={len(d)}')
print(f'  patient raises denture before student: {n}/{len(d)}')

d = load('tmj_pain')
print(f'\n[tmj_pain]  n={len(d)}')
print(f'  localises to joint/muscle: '
      f'{convs_matching(d, r"(joint|jaw joint|in front of my ear|tmj|hinge|muscle)")}/{len(d)}')

d = load('trigeminal_neuralgia')
A = turns(d, 'assistant')
polite = sum(1 for t in A if re.search(r'\b(thank you|thanks|please|sorry|i appreciate)\b', t, re.I))
address = sum(1 for t in A if re.search(r'\bdoctor\b', t, re.I))
angry = sum(1 for t in A if re.search(r"\b(no!|stop|don.t touch|seriously|enough|frustrat\w+|angry|furious|sick of|fed up)\b", t, re.I))
print(f'\n[trigeminal_neuralgia]  n={len(d)}')
print(f'  patient turns          : {len(A)}')
print(f'  politeness markers     : {polite}  ({pct(polite, len(A))})')
print(f'  irritation markers     : {angry}  ({pct(angry, len(A))})')
print(f'  addresses "doctor"     : {address}  ({pct(address, len(A))})')

# --- PROMPT INCONSISTENCY ---------------------------------------------
d = load('periodontal_abscess')
sps = Counter(next(m['content'] for m in c['messages'] if m['role'] == 'system') for c in d)
named = sum(n for s, n in sps.items() if re.search(r'periodontal abscess', s, re.I))
noneg = sum(1 for s in sps if not re.search(r'DO NOT HAVE|FALSE', s))
print(f'\n[periodontal_abscess]  n={len(d)}')
print(f'  distinct system prompts        : {len(sps)}')
print(f'  variants naming the diagnosis  : {sum(1 for s in sps if re.search(r"periodontal abscess", s, re.I))}')
print(f'  variants without negative block: {noneg}')

# --- ROLE ATTRIBUTION -------------------------------------------------
d = load('acute_apical_abscess')
CLIN = r"(we'?ll drain|we need urgent treatment|we'?ll (proceed|start) )"
n = final = preceded = 0
for c in d:
    ms = [m for m in c['messages'] if m['role'] != 'system']
    for i, m in enumerate(ms):
        if m['role'] == 'assistant' and re.search(CLIN, m['content'], re.I):
            n += 1
            if i == len(ms) - 1:
                final += 1
            if i > 0 and re.search(r"we'?ll proceed with treatment", ms[i-1]['content'], re.I):
                preceded += 1
            break
print(f'\n[acute_apical_abscess]  n={len(d)}')
print(f'  patient turns carrying a clinician line : {n}/{len(d)}')
print(f'  ...of which are the FINAL turn          : {final}')
print(f'  ...preceded by "We\'ll proceed with..."  : {preceded}')

# --- LEAST AFFECTED ---------------------------------------------------
d = load('acute_apical_periodontitis')
A = turns(d, 'assistant')
cold = sum(1 for c in d if any(re.search(r'\bcold\b', m['content'], re.I)
                               and not re.search(NEG, m['content'], re.I)
                               for m in c['messages'] if m['role'] == 'assistant'))
print(f'\n[acute_apical_periodontitis]  n={len(d)}')
print(f'  distinct patient turns      : {len(set(A))}/{len(A)}  ({pct(len(set(A)), len(A))})')
print(f'  affirm cold (prompt denies) : {cold}/{len(d)}')

# --- NOT SUPPORTED ----------------------------------------------------
d = load('pulp_necrosis')
PAIN = r'\b(pain|hurts?|hurting|ache[sd]?|aching|sore|throb\w*|sting\w*|agony|painful)\b'
n = 0
for c in d:
    for m in c['messages']:
        if m['role'] != 'assistant':
            continue
        for s in re.split(r'(?<=[.!?])\s+', m['content']):
            if re.search(PAIN, s, re.I) and not re.search(NEG, s, re.I):
                n += 1
                break
        else:
            continue
        break
print(f'\n[pulp_necrosis]  n={len(d)}')
print(f'  conversations affirming pain: {n}/{len(d)}   <- claim not supported')
print()
