guessing-statistics
===================

Research code for computing guessing statistics on a probability distribution.

This code was written during my PhD at the University of Cambridge for computing advanced guessing metrics on a probability distribution. Examples include partial guesswork and partial guessing difficulty, as well as traditional metrics like Shannon entropy, min-entropy Hartley entropy, Renyi Entropy, and guesswork.

A detailed explanation of the mathematics is provided in my PhD thesis: http://www.cl.cam.ac.uk/~jcb82/doc/2012-jbonneau-phd_thesis.pdf

To make working with (especially plotting) data sets easy, they are stored in a special compressed form with many values pre-computed. The test script demonstrate converting raw text to a compressed distribution, printing basic statistics and plotting them.
