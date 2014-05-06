#!/usr/bin/python
# vim: set fileencoding=UTF-8

'''
Compress a series of files specified on the command line

January 2014
Joseph Bonneau
jbonneau@gmail.com
'''

from dist import dist
import sys
import os
import jutils

#Series of labels to use when printing tables or graphs, replacing less elegant filenames
titles = {
"un_uhdr": "Letters [UN UDHR]",
"ry_pins": "Pins [RockYou]",
"ry_passwords": "Passwords [RockYou]",
}

#Place known total population figures here
populations = {
}

#Read from stdin
if len(sys.argv[1:]) == 0:
    d = dist.load(None)
    d.compute_stats()
    d.save(None)

for f in sys.argv[1:]:
    b, e = os.path.splitext(f)
    b_short = os.path.basename(b)

    d = dist.load(f, total_weight = (populations[b_short] if b_short in populations else None), known_title = (titles[b_short] if b_short in titles else None))

    if e == '.dz': 
        np_cum_only = False
        if np_cum_only:
            print dist.title(d)
            d.compute_stats()

            try:
                for s in ['gigp_projected', 'lin_projected', 'sgt_projected', 'zm_projected']:
                    if jutils.memoized_ident(s) in d.memoized:
                        print '\t' + s
                        d.memoized[jutils.memoized_ident(s)].compute_stats()
            except AttributeError:
                pass

        else:
            d.compute_stats(total_weight = (populations[b_short] if b_short in populations else None))
            d.gigp_projected()
            d.lin_projected()
            d.sgt()
            d.sgt_projected()

    d.save(b + '.dz')
    
