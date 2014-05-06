#!/usr/bin/python
# vim: set fileencoding=UTF-8

'''
Compress a file which is a series of
count,[item-name]
into 
count*[number of those counts]

January 2014
Joseph Bonneau
jbonneau@gmail.com
'''

import sys
import re

count_re = re.compile("^([0-9]+).*$")

current = None
count = 1
for line in sys.stdin:
  line = line.strip()
  m = count_re.match(line)
  if not m: continue
  n = m.group(1)
  if n != current:
    if current: print ("%s*%d" % (current, count)) if count > 1 else current
    count = 1
    current = n
  else:
    count += 1

print ("%s*%d" % (current, count)) if count > 1 else current
