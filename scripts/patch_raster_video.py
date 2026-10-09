# Copyright (c) 2020, NVIDIA Corporation. All rights reserved.
# MI50 compatibility changes (c) 2026 tuandat3019.
# This nvdiffrast-derived patch is governed by licenses/nvdiffrast.txt,
# including the research/evaluation/non-commercial use limitation.
"""Lower nvdiffrast scalar video PTX to ordinary CUDA arithmetic.

Reference: NVIDIA PTX ISA 9.7.20.1 (extract, saturate, secondary-op/merge).
Only the isolated source clone changes; the original is retained.
"""
from pathlib import Path
import os
import re

root = Path(os.environ['MI50_RUNTIME']) / 'trellis'
path = root/'sources/nvdiffrast/csrc/common/cudaraster/impl/Util.inl'
backup = path.with_suffix('.inl.original-mi50')
if not backup.exists():
    backup.write_bytes(path.read_bytes())
text = backup.read_text(encoding='utf-8')
bodies = {}
for op, sign in [('add','+'),('sub','-')]:
    for ah in ('lo','hi'):
        for bh in ('lo','hi'):
            ae = f'(S32)(short)((U32)a >> {0 if ah == "lo" else 16})'
            be = f'(S32)(short)((U32)b >> {0 if bh == "lo" else 16})'
            bodies[f'{op}_s16{ah}_s16{bh}'] = f'return {ae} {sign} {be};'
for ah in ('lo','hi'):
    for bh in ('lo','hi'):
        bodies[f'sub_u16{ah}_u16{bh}'] = f'return (S32)(((a >> {0 if ah == "lo" else 16}) & 0xffffu) - ((b >> {0 if bh == "lo" else 16}) & 0xffffu));'
for byte in range(4):
    a = f'((a >> {byte*8}) & 0xffu)'
    bodies[f'add_b{byte}'] = f'return {a} + b;'
    bodies[f'vmad_b{byte}'] = f'return {a} * b + c;'
    bodies[f'vmad_b{byte}_b3'] = f'return {a} * ((b >> 24) & 0xffu) + c;'
bodies.update({
    'add_mask8':'return (a + b) & 0xffu;',
    'sub_mask8':'return (a - b) & 0xffu;',
    'max_max':'return ::max(::max(a,b),c);',
    'min_min':'return ::min(::min(a,b),c);',
    'max_add':'return (S32)((U32)::max(a,b) + (U32)c);',
    'min_add':'return (S32)((U32)::min(a,b) + (U32)c);',
    'add_add':'return a + b + c;',
    'sub_add':'return a - b + c;',
    'add_sub':'return a - c + b;',
    'add_clamp_0_x':'S64 s=(S64)a+b; U32 u=s<0 ? 0u : s>0xffffffffll ? 0xffffffffu : (U32)s; return (S32)::min(u,(U32)c);',
    'add_clamp_b0':'S64 s=(S64)a+b; U32 u=s<0 ? 0u : s>255 ? 255u : (U32)s; return (S32)(((U32)c & 0xffffff00u) | u);',
    'add_clamp_b2':'S64 s=(S64)a+b; U32 u=s<0 ? 0u : s>255 ? 255u : (U32)s; return (S32)(((U32)c & 0xff00ffffu) | (u << 16));',
})
patched = []
select_count = 0
lines = []
for line in text.splitlines(keepends=True):
    if re.search(r'asm\("v(?:add|sub|min|max|mad)\.',line):
        name = re.search(r'\s(\w+)\s*\([^)]*\)\s*\{',line).group(1)
        assert name in bodies, name
        line = line[:line.index('{')]+'{ '+bodies[name]+' }\n'
        patched.append(name)
    elif 'asm("slct.' in line:
        # PTX slct selects a when the signed selector is non-negative, else b.
        line = line[:line.index('{')]+'{ return c >= 0 ? a : b; }\n'
        select_count += 1
    lines.append(line)
assert len(patched) == len(bodies), (len(patched),len(bodies))
assert select_count == 3, select_count
path.write_text(''.join(lines),encoding='utf-8')
print('RASTER_VIDEO_PATCH',len(patched),'SLCT',select_count,patched)
