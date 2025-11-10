# -*- encoding:utf-8 -*-
import sys

lines = []
with open(sys.argv[1], 'r') as fin:
  for line_str in fin:
    lines.append(line_str)

with open(sys.argv[1], 'w') as fout:
  for line_str in lines:
    if '_static/searchtools.js' in line_str:
      fout.write(
        '    <script type="text/javascript" src="_static/language_data.js"></script>\n'
      )
    fout.write(line_str)
