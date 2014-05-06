#!/usr/bin/python
# vim: set fileencoding=UTF-8

'''
Test output of statistics

January 2014
Joseph Bonneau
jbonneau@gmail.com
'''

from dist import dist
import sys
import os
import jutils

colors = ['b', 'g', 'r', 'm', 'c', 'y', 'k']

#Test bed code:
to_plot = []
index = 0

for f in sys.argv[1:]:

    d = dist.load(f)
 
    print d.G(0.5), d.gigp_projected().G(0.5)

    #print d.x
    print 'Number of types:', d.num_types()
    print 'Number of tokens:', d.num_tokens()

    for i in (0.25, 0.5, 0.75):
        print 'μ:', i, d.mu(i), d.mu(i, tilde_value = False)

    for i in range(1, 11) + [100, 1000, 10000]:
        if i < d.num_tokens():
            print 'λ:', i, d.lambda_value(i), d.lambda_value(i, tilde_value = False)

    for (i, j) in ((0, 0.25),(0, 0.5), (0.0, 0.75), (0, 1), (0.25, 0.75)):
        if i < d.num_tokens():
            #d.mu_aggregate(i, j)
            print 'M:', i, j, d.mu_aggregate(i, j), d.mu_aggregate(i, j, tilde_value = False)
    
    print '~G:', d.stats['~G']
    print 'G:', d.stats['G']

    print d.dump_stats()

#    print d.projected().mu_aggregate(0, 0.5)

    print 'cutoff point:', d.confidence_p()

    print 'loss:', d.stats['coefficient_of_loss']
    print 'V growth rate:', d.stats['hapax_proportion']

    print 'V(1, M):', d.x[-1][1]
    print 'V(2, M):', d.x[-2][1]
    print 'Central LNRE?', float(d.x[-1][1]) / (2 * d.x[-2][1])


    #d.zm_projected()

