#!/usr/bin/python
# vim: set fileencoding=UTF-8

'''
Test of plotting functionality

January 2014
Joseph Bonneau
jbonneau@gmail.com
'''

from dist import dist
import sys
import os

colors = ['b', 'g', 'r', 'm', 'c', 'y', 'k']

#Test bed code:
to_plot = []
index = 0

for f in sys.argv[1:]:

	d = dist.load(f)
	f = os.path.splitext(os.path.basename(f))[0]
	to_plot.append((d, {'color':colors[index], 'split_plot': True, 'show_projected': True, 'show_tail': False}))
	index = (index + 1) % len(colors)

dist.plot_marginal_guesswork(to_plot, legend_outside = False, plot_type = 'mu')

