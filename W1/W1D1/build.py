#!/usr/bin/env python3
"""
build.py — 读取 sections/*.md，生成 W1D1.ipynb

sections/ 下的 .md 文件格式：
  markdown 说明 + ```python 代码块 交替。
build.py 按 ```python 分隔符拆分，前为 markdown，后为代码。
"""
import glob, re, os, nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

DIR = os.path.dirname(os.path.abspath(__file__))

def split_cells(text):
    """按 ```python ... ``` 拆成 [('md', str), ('code', str), ...]"""
    code_pat = re.compile(r'```python\n(.*?)\n```', re.DOTALL)
    matches = list(code_pat.finditer(text))
    cells = []
    prev = 0
    for m in matches:
        md = text[prev:m.start()].strip()
        if md:
            cells.append(('md', md))
        cells.append(('code', m.group(1).rstrip()))
        prev = m.end()
    tail = text[prev:].strip()
    if tail:
        cells.append(('md', tail))
    return cells

# 十大部分，按顺序排列
SECTION_FILES = [
    '1-tensor.md',
    '2-autograd.md',
    '3-nn-module.md',
    '4-optim.md',
    '5-dataloader.md',
    '6-save-load.md',
    '7-gpu.md',
    '8-finetune.md',
    '9-layers-ref.md',
    '10-training-loop.md',
]

cells = [new_markdown_cell(
    '# W1D1｜PyTorch 张量操作 + 自动求导\n\n'
    '> 学习日期：2026-04-10\n'
    '> 目标：掌握 PyTorch 核心 API，理解自动求导机制，夯实 Day 1 基础'
)]

for fname in SECTION_FILES:
    path = os.path.join(DIR, 'sections', fname)
    if not os.path.exists(path):
        continue
    text = open(path, encoding='utf-8').read()
    parsed = split_cells(text)
    for kind, content in parsed:
        if kind == 'md' and content.strip():
            cells.append(new_markdown_cell(content))
        elif kind == 'code' and content.strip():
            cells.append(new_code_cell(content))

nb = new_notebook()
nb.cells = cells
out = os.path.join(DIR, 'W1D1.ipynb')
with open(out, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)
print(f'Done! {len(cells)} cells -> {out}')
