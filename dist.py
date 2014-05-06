#!/usr/bin/python
# vim: set fileencoding=UTF-8

'''
Class for handling a distribution and computing security-related stats

January 2014
Joseph Bonneau
jbonneau@gmail.com
'''

import sys
import os
import math
import status
import gzip
import cPickle
import copy
import re
import math
import numpy
import random
import mpmath
import itertools

from jutils import *
from rpy import r
import scipy.stats.distributions

class dist:

    '''
    ----------------------------------------------------------
    Member variables
    ----------------------------------------------------------
    '''

    title = ""
    x = None    #main data items
    stats = {}    #computed values
    
    known_weight = None     #total weight of all known events in distribution
    total_weight = None     #true weight of all events in distribution (may be known higher)
    known_types = None    #number of known types
    total_types = None    #total number of types (after adding in padding types)

    rungs = None

    '''
    ----------------------------------------------------------
    Initialising/saving/loading methods
    ----------------------------------------------------------
    '''

    def __init__(self, title = 'unknown distribution', total_weight = None, verbose = True, clean_title = True):
        '''use filename as title if unknown, truncating extension and path'''
        self.title = title
        if clean_title: self.title = dist.clean_title(self.title)
        self.stats = {}
        self.x = []
        self.total_weight = total_weight


    def save(self, filename = None, verbose = True):
        '''Save dist to file, using the parsed in path to determine file type'''

        if filename:
            ext = os.path.splitext(filename)[1]
            if ext == ".dist":    
                if verbose: sys.stderr.write("Storing distribution to cPickle file: %s\n" % filename)
                f = open(filename, 'wb')
            elif ext == ".dz":    
                if verbose: sys.stderr.write("Storing distribution to compressed .dz file: %s\n" % filename)
                f = gzip.GzipFile(filename, 'wb')
            
            else:
                sys.stderr.write("Writing to file format not supported: %s\n" % filename)
                assert False            
        else:
            f = sys.stdout

        cPickle.dump(self, f)
        
    def read_from_text(self, f, verbose = True, kept_names = 100, progress_freq = 10000, delim = ",", flip_order = False):
        '''Read a text file and create a distribution.
        Need to support legacy file format'''

        l = []
        need_to_sort = False
        last = None
        n = 0

        for line in f:
        
            line = line.replace("\n", "")
            line = line.replace("\r", "")
            name = None

            #comments and special lines
            if line.startswith("#") or len(line) < 1:
                continue    

            tokens = line.split(delim)

            #two tokens found
            if len(tokens) >= 2:
                #more than one token found
                #if len(tokens) > 2 and verbose: sys.stderr.write("Extra tokens in line: %s\n" % line)
                count = tokens[0]

                line = "".join(tokens[1:])
    
                #csv handling mode
                if flip_order and len(tokens) == 2:
                    count = tokens[1]
                    line = tokens[0].strip().strip("\"").strip()

            #one token found
            else:    
                count = tokens[0]
                line = None
        
            #Allow for *freq modifiers
            if count.find('*') > 0:(count, freq) = count.split('*')
            else: freq = '1'

            #Parse numbers as floating point
            try: 
                count = float(count.strip())
                freq = int(freq.strip())
            except ValueError:
                sys.stderr.write("Couldn't parse numbers: (%s, %s)\n" % (count, freq))
                continue

            assert count > 0

            if need_to_sort is False and last is not None and last < count:
                need_to_sort = True
            last = count

            l.append((count, line, freq))

            #print sequence of dots
            if verbose and progress_freq and n % progress_freq == 0:
                sys.stderr.write('.')
                sys.stderr.flush()

            n += 1

        self.init_from_list(l, need_to_sort = need_to_sort)

        if verbose: sys.stderr.write(' done\n')  

    def init_from_list(self, l, need_to_sort = False, kept_labels = 100):
                
        if need_to_sort: 
            sys.stderr.write("Sorting list...\n")
            l.sort(key = lambda x: x[0], reverse = True)

        last = None
        total_weight = 0
        num_types = 0
        self.x = []

        for s in l:
            count = s[0]
            label = s[1] if len(s) > 1 and num_types < kept_labels else None    
            freq = s[2] if len(s) > 2 else 1

            #attempt to combine with previous item if possible
            if count == last: 
                self.x[len(self.x) - 1][1] += freq
                #Save symbol if needed, only store listed Types
                if label:                 
                    if len(self.x[len(self.x) - 1]) < 2: 
                        self.x[len(self.x) - 1].append([])
                    self.x[len(self.x) - 1][2].extend([label] * freq)

            else: 
                self.x.append([count, freq] if not label else [count, freq, [label] * freq])

            last = count
            total_weight += count * freq
            num_types += freq

        self.known_weight = total_weight
        self.known_types = num_types
        self.total_weight = (self.known_weight if not self.total_weight else float(self.total_weight))

    @staticmethod
    def clean_title(t):
        return re.sub(r'([^\\])_', r'\1\\_', t)

    @staticmethod
    def load(filename, compressed = True, known_title = None, total_weight = None, verbose = True, recompute = False):
        '''
        Load a distribution from a filename.
        The extensions is parsed to determine what format the file is in.
        '''

        #no filename, load from stdin in text format
        if not filename:
            f = sys.stdin
            text_format = True
            ext = '.txt'
        else:

            if not os.path.exists(filename):
                sys.stderr.write("Can't load distribution from file: %s (file doesn't exist)" % filename)
                return None

            ext = os.path.splitext(filename)[1]
            if ext == ".txt" or ext == ".csv":    
                if verbose: sys.stderr.write("Loading distribution from text file: %s\n" % filename)
                f = open(filename, 'r')
                text_format = True
            elif ext == ".dist":    
                if verbose: sys.stderr.write("Loading distribution from cPickle file: %s\n" % filename)
                f = open(filename, 'r')
                text_format = False
            elif ext == ".dz":    
                if verbose: sys.stderr.write("Loading distribution from compressed .dz file: %s\n" % filename)
                f = gzip.GzipFile(filename, 'r')
                text_format = False
            else:
                sys.stderr.write("Can't load distribution from file: %s (unknown extension)" % filename)
                return None

        if not text_format:     
            d = cPickle.load(f)
            if total_weight: d.total_weight = total_weight
            if known_title: d.title = known_title

            if recompute or total_weight: d.compute_stats()

        else:        
            d = dist(title = ('[unknown]' if not filename and not known_title else (known_title if known_title  else os.path.splitext(os.path.basename(filename))[0])), total_weight = total_weight, verbose = verbose)
            d.read_from_text(f, verbose = verbose, delim = ",", flip_order = (ext == ".csv"))

            #Compute d's stats because it hasn't been loaded before
            d.compute_stats()
        if f != sys.stdin: f.close()

        d.title = dist.clean_title(d.title)
        return d

    @staticmethod
    def uniform_dist(N, title = None):

        if not title:
            new_dist = dist(title = '$\\mathcal{U}_{%d}$' % N, clean_title = False)    
        else:
            new_dist = dist(title = title)

        new_dist.x = [[1000, int(N)],]

        #if float(N) != round(float(N)): new_dist.x += [int(1000 * (float(N) - math.floor(float(N)))), 1]

        new_dist.compute_stats(adjust_total = False)
        return new_dist
        
    @staticmethod
    def zipf_dist(N, title = None, s = 0.83):

        if not title:
            new_dist = dist(title = ('$\\text{Zipf}_{%d}$' % N) if s == 1.0 else ('$\\text{Zipf}^{%0.2f}_{%d}$' % (s, N)), clean_title = False)    
        else:
            new_dist = dist(title = title)

        new_dist.x = [0] * N
        current = 100.0
        for i in range(N, 0, -1):
            new_dist.x[i - 1] = [current, 1]
        
            if i > 1: current *= (float(i) / (i - 1)) ** s


        #print new_dist.x

        #if float(N) != round(float(N)): new_dist.x += [int(1000 * (float(N) - math.floor(float(N)))), 1]

        new_dist.compute_stats(adjust_total = False)
        return new_dist

    @staticmethod
    def convert_from_histogram(h, title = None, kept_labels = 100, total_weight = None):
        '''Assumes a histogram of item:count pairs'''

        if not title: title = 'loaded from histogram'
        new_dist = dist(title = title, clean_title = False)    
        if total_weight: new_dist.total_weight = total_weight

        new_dist.init_from_list([(x[1], x[0]) for x in h.items()], need_to_sort = True, kept_labels = kept_labels)       
        new_dist.compute_stats(adjust_total = False)

        return new_dist

    @staticmethod
    def convert_from_compressed_histogram(h, title = None, total_weight = None):
        '''Assume a histogram of count:freq pairs'''
        if not title: title = 'loaded from histogram'
        new_dist = dist(title = title, clean_title = False)
        if total_weight: new_dist.total_weight = total_weight

        new_dist.x = sorted([[i[0], i[1]] for i in h.items()], key = lambda x: x[0], reverse = True)

        new_dist.known_weight = sum(map(lambda x: x[0]* x[1], h.items()))
        new_dist.known_types = sum(h.values())
        new_dist.total_weight = (new_dist.known_weight if not new_dist.total_weight else float(new_dist.total_weight))

        new_dist.compute_stats(adjust_total = False)

        return new_dist


    '''
    ----------------------------------------------------------
    Utility/access functions
    ----------------------------------------------------------
    '''

    def num_tokens(self):
        '''Total number of tokens, or total weight in the distribution'''
        return self.total_weight    

    def num_types(self):
        '''Number of distinct types (elements) in the distribution'''
        return self.total_types
    
    def num_ranks(self):
        '''Number of distinct weights/counts for types in the distribution'''
        return len(self.x)

    def title(self):
        return self.title

    def __str__(self):
        return "Distribution object \"%s\" (N=%s, %s)" % (self.title, self.known_types, self.total_types)

    def __len__(self):
        return self.num_types()


    def probs(self, **kwargs):
        '''Yield probabilities only'''
        for w in self.weights(**kwargs):
            yield float(w) / self.total_weight

    def weights(self, **kwargs):
        '''Yield weight of each item, not normalised by total'''
        for i in self.types(labeled_only = False, **kwargs):
            yield i[0]

    def types(self, labeled_only = False, **kwargs):
        '''Yield types of items by common weight'''
        for p in self.items(**kwargs):

            for j in range(p[1]):
                label = p[2][j] if (len(p) > 2 and len(p[2]) > j and p[2][j]) else None
                if label: yield (p[0], label)
                elif not labeled_only: yield (p[0],)

    def __iter__(self, **kwargs):
        return self.items(**kwargs)

    def item(self, i, known_only = False):
        '''
        Wrapper for direct access to distribution's items.
        Add extra multiples of the last event if needed
        Also guarantee a second argument to the result
        '''
        assert i < len(self.x)

        #Expand the count for the last item if needed
        expanded_count = 0
        if i == len(self.x) - 1 and not known_only:
            expanded_count = int(math.ceil((self.total_weight - self.known_weight) / self.x[i][0]))

        #update the count
        return [self.x[i][0], (self.x[i][1] if len(self.x[i]) >= 2 else 1) + expanded_count,] + self.x[i][2:]


    def items(self, max_types = None, min_weight = None, max_rank = None, known_only = False):
        '''
        Yield the items in the distribution.
        This can be bounded by max_types, min_weight, max_rank, or limited to known items
        '''

        p_cumulative = 0
        total_yielded = 0

        for i in range(len(self.x)):

            next = self.item(i, known_only)

            #check early termination conditions
            if min_weight and next[0] < min_weight: return
            if max_rank and i > max_rank: return
            
            #yield an adjusted count item for the last item (extrapolated) or cut-off
            if (max_types and next[1] + total_yielded >= max_types):
                yield [next[0], min(next_count, max_types - total_yielded)] + next[2:]
                return

            yield next
            total_yielded += next[1]
            p_cumulative += next[0] * next[1]

    @memoized_method
    def V(self, m):        
        '''Yield the total number of events seen m times, the vocabulary at m'''
        i = max(len(self.x) - m - 1, 0)
        while i < len(self.x):
            if self.x[i][0] == m: return self.x[i][1]
            elif self.x[i][0] < m: return 0
            i += 1    
        return 0


    '''
    ----------------------------------------------------------
    Printing functions
    ----------------------------------------------------------
    '''

    @staticmethod
    def print_item(t):
        '''Pretty print a single item from a distribution'''
        return '%s x %s %s' % (str(("%d" if isinstance(t[0], int) else "%0.8f") % t[0]).rjust(10), str(t[1]).rjust(6) if len(t) > 1 else '1', t[2] if len(t) > 2 else '')

    @staticmethod
    def print_items(f):
        '''Pretty print an iterable list of items'''
        for t in f:
            sys.stdout.write(print_item(t) + '\n')

    def print_items(self, **kwargs):
        '''Return a string with all items in the distribution'''
        print_items(self.items(kwargs))

    def dump_spc_format(self):
        '''Pretty print a single item from a distribution'''
        s = 'm Vm\n'
        for x in self.x[::-1]:
            s += '%d %d\n' % (x[0], x[1])

        return s


    def dump_mu(self, items = None, step = 0.01, print_tilde_values = True):
        '''
        Returns a printable string representing the values of mu_alpha computed
        for this distribution. If a list of items is passed in, print a comparison
        table for all of them.
        '''

        if not items: items = (self,) 
        
        #Top matter
        s = "\n".join(("-" * 21,'Marginal Guesswork',"-" * 21)) + '\n'

        #Column headers
        s += "α"
        for x in items:
            s += "\t" + ("~" if print_tilde_values else " ") + "μ-" + x.title[0:4]
        s += "\n"


        current = 0     #current alpha value for each row

        #Print mu values
        while current <= 1.01:
            s += str(current)
        
            for i in range(len(items)):

                #skip stored values if this dist has more computed than we're printing
                mu = items[i].mu(current, tilde_value = print_tilde_values)
                s +=  ("\t%.4f" if print_tilde_values else "\t%d" ) % mu
                    
            s += "\n"
            current += step

        return s


    def dump_lambda(self, items = None, limit = 20, print_tilde_values = True):
        '''
        Returns a printable string representing the values of lambda_beta computed
        for this distribution. If a list of items is passed in, print a comparison
        table for all of them.
        '''
    
        if not items: items = (self,)
        
        #Top matter
        s = "\n".join(("-" * 21,'Marginal Success Rate',"-" * 21)) + '\n'
        
        #Column headers
        s += "β"
        for x in items:
            s += "\t" + ("~" if print_tilde_values else " ") + "λ-" + x.title[0:4]
        s += "\n"

        #Print lambda values
        for i in range(1, limit + 1):
            s += str(i)

            for j in range(len(items)):    
                s +=  "\t%.4f" % items[j].lambda_value(i, print_tilde_values)
                    
            s += "\n"

        return s

    def dump_stats(self):
        ''' Print all known statistics of the distribution.    '''

        s = "\n".join(("-" * 21,"Stats for distibution: %s" % self.title,"-" * 21, ""))

        for k in self.stats:
            s += '%s\t%s\n' % (k, str(self.stats[k]))

        return s


    def find_dist_point(self, max_alpha = None, max_beta = None, min_weight = None, min_p = None, max_freq = 0, mu_acc = 0.01):
        '''
        Find the furthest point in the distribution subject to the specified constraints.
        They are treated as an and if multiple constraints are passed in.

        Return values:
        index = index in self.x of the current point
        pre_added = number of points in the current bucket already added
        n_cumulative = total number of events so far
        p_cumulative = total probability of all events so far
        np_cumulative = p_1 * 1 + p_2 * 2 + ... p_n_cumulative * n_cumulative (partial guesswork summation)
        mu_agg = aggregate mu so far
        '''

        index, pre_added, n_cumulative, p_cumulative, np_cumulative, mu_agg = 0, 0, 0, 0, 0, 0
        
        if min_p is not None: 
            min_weight = min_p * self.total_weight if min_weight is not None else min_p * self.total_weight
            #print min_weight

        if self.rungs:
            j = 0
            while j < len(self.rungs):
                if max_alpha is not None and self.rungs[j][3] > max_alpha: break
                if max_beta is not None and self.rungs[j][2] > max_beta: break
                if min_weight is not None and self.item(self.rungs[j][0])[0] < min_weight: break

                j += 1

            if j > 0:
                (index, pre_added, n_cumulative, p_cumulative, np_cumulative, mu_agg) = self.rungs[j - 1]
                #print "\t", j, self.rungs[j - 1]
    
        rung_pre_added = pre_added

        #print (index, pre_added, n_cumulative, p_cumulative, mu_agg)
        #print min_weight

        while index < len(self.x):
                
            if max_alpha is not None and p_cumulative >= max_alpha: break
            if max_beta is not None and n_cumulative >= max_beta: break

            if rung_pre_added is None: index += 1
            if index >= len(self.x): break

            pre_added = 0
            if rung_pre_added is not None: pre_added = rung_pre_added

            if min_weight is not None and self.item(index)[0] < min_weight:
                #print index
                break
            #print self.item(index)[0]

            next_p = float(self.item(index)[0]) / self.total_weight
    
            can_add = [self.item(index)[1] - pre_added,] #limit from the distribution
            if max_alpha is not None: can_add.append(math.ceil((max_alpha - p_cumulative) / next_p)) #limit from alpha
            if max_beta is not None: can_add.append(max_beta - n_cumulative)

            #add the minimum amount allowed by any constraints
            to_add = max(min(can_add), 1 if not rung_pre_added else 0)

            #print next_p, to_add, p_cumulative, n_cumulative
            added = 0
            while added < to_add:
                next_add = max(1, int(mu_acc / next_p))
                next_add = min(next_add, to_add - added)
                next_add = int(next_add)

                #build up np_cumulative-use Gauss summation formula
                np_cumulative += next_p * (n_cumulative * next_add + ((next_add ** 2 + next_add) / 2) )
                p_cumulative += next_add * next_p
                n_cumulative += next_add

                mu_agg += math.log(n_cumulative / p_cumulative, 2) * next_add * next_p
            
                added += next_add    
                pre_added += next_add
   
            rung_pre_added = None

        return (index, pre_added, n_cumulative, p_cumulative, np_cumulative, mu_agg)


    def brute_force_pg(self, n):

        test, i, t, nc = 0, 0, 0, 0
        while i < len(self.x):
            can_add = min(self.x[i][1], n - nc)
            t = 0
            while t < can_add:
                t += 1
                nc += 1
                test += (nc) * self.x[i][0]
       
            if nc == n: break

            i += 1

        return test / self.total_weight

    '''
    ----------------------------------------------------------
    Mathematical/statistical functions
    ----------------------------------------------------------
    '''

    def G(self, alpha, tilde_value = True):
        '''
        Compute G (or tilde_G) at the specified point
        This requires finding the right point at the distribution where
        the cumulative probability is equal to alpha.
        '''
        (index, pre_added, n_cumulative, p_cumulative, np_cumulative, mu_agg) = self.find_dist_point(max_alpha = alpha)

        if tilde_value:
            if n_cumulative <= 0 or p_cumulative <= 0: return 0.0
            n_rate = np_cumulative +  n_cumulative * (1 - p_cumulative) if p_cumulative < 0.999 else np_cumulative
            n_rate /= p_cumulative

            return math.log(2 * (n_rate) - 1, 2) + (math.log(1 / (2 - p_cumulative), 2) if p_cumulative > 0 else -1)
        else:
            if n_cumulative <= 0 or p_cumulative <= 0: return 0.0
            return (np_cumulative + (1 - p_cumulative) * n_cumulative) if p_cumulative < 0.999 else np_cumulative



    def mu(self, alpha, tilde_value = True):
        '''
        Compute mu (or tilde_mu) at the specified point
        This requires finding the right point at the distribution where
        the cumulative probability is equal to alpha.
        '''
        (index, pre_added, n_cumulative, p_cumulative, np_cumulative, mu_agg) = self.find_dist_point(max_alpha = alpha)

        if tilde_value: 
            if n_cumulative <= 0 or p_cumulative <= 0: return 0.0
            return math.log(n_cumulative / p_cumulative, 2)
            #return math.log(n_cumulative, 2) #experimental, non-adjusted version
        else:
            return n_cumulative


    def mu_aggregate(self, alpha_1 = 0, alpha_2 = 1, tilde_value = True, approximation = None):
        '''
        Compute mu (or tilde_mu) at the specified point
        This requires finding the right point at the distribution where
        the cumulative probability is equal to alpha.
        Makes use of 'rungs' pre-storing points in the distribution.
        '''

#        print alpha_1, alpha_2

        alpha_1 = max(alpha_1, 0)
        alpha_2 = min(alpha_2, 1)
        if alpha_2 < alpha_1: return 0

        (index_1, pre_added_1, n_cumulative_1, p_cumulative_1, np_cumulative_1, mu_agg_1) = self.find_dist_point(max_alpha = alpha_1, mu_acc = 0.00001)
        (index_2, pre_added_2, n_cumulative_2, p_cumulative_2, np_cumulative_2, mu_agg_2) = self.find_dist_point(max_alpha = alpha_2, mu_acc = 0.00001)

        #print (index_1, pre_added_1, n_cumulative_1, p_cumulative_1, mu_agg_1)
        #print (index_2, pre_added_2, n_cumulative_2, p_cumulative_2, mu_agg_2)

        #correct for error on the corners of the corners
        if p_cumulative_1 != 0: mu_agg_1 -= (p_cumulative_1 - alpha_1) * math.log(n_cumulative_2 / p_cumulative_2, 2)
        if p_cumulative_2 != 0: mu_agg_2 -= (p_cumulative_2 - alpha_2) * math.log(n_cumulative_2 / p_cumulative_2, 2)

        #print (index_1, pre_added_1, n_cumulative_1, p_cumulative_1, mu_agg_1)
        #print (index_2, pre_added_2, n_cumulative_2, p_cumulative_2, mu_agg_2)

        result = (mu_agg_2 - mu_agg_1) / (alpha_2 - alpha_1)
        if not tilde_value: result = 2 ** result
        return result

    @staticmethod
    def mu_diff(d1, d2, start = 0, stop = 1.0, inc = 0.01,**kwargs):

        result = 0
        current = start
        while (current < stop):
            next = min(current + inc, stop)
            result += d1.mu_aggregate(current, next,**kwargs) - d2.mu_aggregate(current, next,**kwargs)
            current = next
        result /= (float(stop - start) / inc)
    
        return result

    def lambda_value(self, beta, tilde_value = True):
        '''
        Compute lambda (or tilde_lambda) at the specified point
        This requires finding the right point at the distribution where
        the cumulative number of items is equal to beta.
        Makes use of 'rungs' pre-storing points in the distribution.
        '''

        (index, pre_added, n_cumulative, p_cumulative, np_cumulative, mu_agg) = self.find_dist_point(max_beta = beta)

        if tilde_value: 
            if n_cumulative <= 0 or p_cumulative <= 0: return 0.0
            return math.log(n_cumulative / p_cumulative, 2)
        else:
            return p_cumulative


    def p_below(self, p_value, tilde_value = True):
        '''
        Compute the percentage of the distribution which is below p_value
        '''

        return self.lambda_value(p_value * self.num_types(), tilde_value = False)

    @memoized_method
    def confidence_p(self, min_weight = 6):
        '''
        Compute the propotion of the distribution whose confidence 
        is based on items of weight more than p
        '''    
        (index, pre_added, n_cumulative, p_cumulative, np_cumulative, mu_agg) = self.find_dist_point(min_weight = min_weight)

        return min(p_cumulative, float(self.known_weight) / self.total_weight)

    def compute_stats(self, max_alpha = 1.0, num_rungs = 1000, mu_acc = 1000, debug = False, total_weight = None, adjust_total = True):    
        '''
        Compute a suite of stats on the distribution d
        num_rungs controls the number of hint points in the distribution to store to speed up later calculations
        '''
        self.legacy_rename(); self.stats = {} #erase all known statistics

        #total of all events in distribution
        if not self.known_weight:
            self.known_weight = sum(map(lambda x: (x[0] * x[1] if len(x) > 1 else x[0]), self.x))
        self.stats['known_weight'] = self.known_weight    

        if self.known_weight <= 0: return

        #number of known types
        if not self.known_types:
            self.known_types = sum(map(lambda x: (x[1] if len(x) > 1 else 1), self.x))
        self.stats['known_types'] = self.known_types


        #total of all events, including unknowns. Might be externally specified
        if not self.total_weight:
            self.total_weight = self.known_weight

        #round total weight so that it's an integral number of the final unit
        #if adjust_total:
        #    self.total_weight = sum(map(lambda x: (self.item(x)[0] * self.item(x)[1] if len(self.x) > 1 else x[0]), range(len(self.x))))
        #    self.total_weight = self.known_weight + (round((self.total_weight - self.known_weight) / self.x[len(self.x) - 1][0]) * self.x[len(self.x) - 1][0])


        self.stats['total_weight'] = self.total_weight

        #print self.known_weight, self.total_weight
        self.stats['p_known'] = self.known_weight / self.total_weight

        self.stats['H1'] = 0.0    #Shannon entropy H1
        self.stats['H2'] = 0.0    #Renyi entropy H2
        self.stats['G'] = 0.0        #Guess entropy G
        self.stats['Gini'] = 0    #Gini coefficient

        self.rungs = []
        big_step = 1.0 / (num_rungs)
        small_step = 1.0 / (num_rungs * mu_acc)
        next_rung = big_step
        
        self.stats['Mu_agg'] = 0.0

        index = 0    #current symbol number
        p_cumulative = 0.0    #cumulative probability
        np_cumulative = 0   #for adjusted marginal guesswork

        self.known_tokens = 0; self.total_tokens = 0

        #iterate through each event, building up the summation stats
        for j in range(len(self.x)):

           
            t = self.item(j, known_only = False)
            w = t[0]
            if w <= 0: 
                sys.stdout.write("Negative element in distribution: % s\n" % str(t))                
                break
            n = t[1] if len(t) > 1 else 1
            p = float(w) / self.total_weight

            #print t, w, n, p

            #update summation stats
            try:
                self.stats['H1'] += - n * p * math.log(p, 2)
                self.stats['G'] += n * p * (index) + (p * (n ** 2 + n)) / 2
                self.stats['H2'] += n * p ** 2
                self.stats['Gini'] += n * p_cumulative + (p * (n ** 2 + n - 1)) / 2
            except ValueError:
                sys.stdout.write('Value Error: %d, %.2f, %.2f, %.2f, %.2f\n' % (n, p, p_cumulative, w, self.total_weight))

            #fill in as many rungs as possible
            #may be multiple if probability is bigger than rung_step
            added = 0
            while True:
                diff = (math.ceil(p_cumulative / small_step) * small_step) - p_cumulative
                if diff <= 0.0: diff += small_step            
                to_add = min(max(int(math.ceil(diff / p)), 1), n - added)
                #print to_add, n, added, diff, p_cumulative, p, small_step

                #print np_cumulative, p, index, added, to_add, p * ((index + added) * to_add + ((to_add ** 2 + to_add) / 2) ), ((index + added) * to_add + ((to_add ** 2 + to_add) / 2) ), ((to_add ** 2 + to_add) / 2)
                np_cumulative += p * ((index + added) * to_add + ((to_add ** 2 + to_add) / 2) )
                #print 
                #print np_cumulative
                p_cumulative += to_add * p
                added += to_add
                
                if p_cumulative <= 0: print p_cumulative, to_add, p, self.x

                self.stats['Mu_agg'] += math.log((index + added) / p_cumulative, 2) * to_add * p

                if p_cumulative > next_rung:
                    self.rungs.append((j, added, index + added, p_cumulative, np_cumulative, self.stats['Mu_agg']))
                    next_rung = (1 + math.floor(p_cumulative / big_step)) * big_step            

                if added >= n: break

            index += n

        self.stats['total_types'] = index
        self.total_types = index

        #finalize summation statistics
        self.stats['H2'] = -1 * math.log(self.stats['H2'], 2)        
        self.stats['Gini'] = ((self.stats['Gini'] / self.stats['total_types']) - 0.5) * 2
        self.stats['~G'] = math.log(2 * self.stats['G'] - 1, 2) #guessing entropy conversion
        self.stats['H0'] = math.log(self.stats['total_types'], 2) #Hartley entropy
        self.stats['Hmin'] = -math.log(float(self.x[0][0]) / self.total_weight, 2) #min entropy
        self.stats['coefficient_of_loss'] = sum(i[1] * math.exp(-i[0]) for i in self.items()) / float(self.num_types())    #see pp 56-57 of Baayen WFD book    
        self.stats['hapax_proportion'] = max(map(lambda x: x[1], self.items())) / float(self.num_tokens())    #proportion of items with only a single event

    '''
    ----------------------------------------------------------
    Strength estimators
    ----------------------------------------------------------
    '''

    def unseen_strength(self, smooth_unseen = False):
        return math.log(self.num_types() if not smooth_unseen else self.num_types() + self.x[-1][1], 2)

    def prob_strength(self, count, smooth_unseen = False):
        
        if count == 0: return self.unseen_strength(smooth_unseen)

        return -math.log(float(count) / self.num_tokens(), 2)    


    def index_strength(self, index, half_adjust = True, smooth_unseen = False):
        
        if index > self.num_types(): return self.unseen_strength(smooth_unseen)

        if half_adjust:
            dp = self.find_dist_point(max_beta = index)
            index = dp[2] - dp[1]
            index += float(1 + self.item(dp[0])[1]) / 2
    
        return math.log(2 * index - 1, 2)    

    @staticmethod
    def nist_strength(pw, index = None):
        entropy_per_char = [4.0,] + [2.0,] * 7 + [1.5] * 12
        e = sum(entropy_per_char[:len(pw)]) + 1.0 * max(len(pw) - 20, 0)
        e += 6 if ((not pw.isalpha() and not pw.isdigit()) or pw.lower() != pw) else 0
        e += 6 if (index and index > 50000) else 0

        return e


    def marginal_strength(self, count, index = None, smooth_unseen = False):
        
        if count == 0: return self.stats['~G']

        if index is not None:
            dp = self.find_dist_point(max_beta = index)
        else:
            dp = self.find_dist_point(min_weight = count + 1)            
            dp = self.find_dist_point(max_beta = dp[2] + (self.item(dp[0])[1] + 1) / 2)            
        
        return self.G(dp[3])

#        return math.log(dp[2] / dp[3], 2)
      

    '''
    ----------------------------------------------------------
    Sampling functions
    ----------------------------------------------------------
    '''

    @memoized_method
    def expected_sample(self, m, M = None, poisson = False, accuracy = 0.0001):
        '''
        Compute the expected number of elements which will occur m times in a sample of size M.
        With M unspecified, this is a bootstrap resample.
        '''

        N = self.num_tokens()
        if not M: M = N
        sample_ratio = float(M) / N


        def sample_func(x):
            return (scipy.stats.distributions.poisson.pmf(m, float(x) * sample_ratio) if poisson else scipy.stats.distributions.binom.pmf(m, M, float(x) / N)) * self.V(x) if x > 0 else 0

        result = 0
        i = j = int(m * sample_ratio)
    
        result += sample_func(i)
        while True:
            i -= 1
            j += 1
            next = sample_func(j) + sample_func(i)
            result += next
            if next / result < accuracy: break

        return result


    @memoized_method
    def ideal_sample(self, m, M = None):
        '''
        Compute the 'ideal' number of elements which will occur m times in a sample of size M.
        '''        

        
        N = self.num_tokens()
        if not M: M = N
        sample_ratio = float(M) / N

        lower = 1 if m == 1 else int(round((m - 0.5) * N / M))
        upper = int(round((m + 0.5) * N / M))

        return sum([self.V(k) for k in range(lower, upper)])


    @memoized_method
    def count_bias(self, m, M = None):
        '''
        Compute the bootstap bias of the number of elements occuring m times.
        '''        
        #print m, self.expected_sample(m, M) / self.ideal_sample(m, M)
        #print m, self.V(m), self.ideal_sample(m, M)
        return self.expected_sample(m, M) / self.ideal_sample(m, M)

    @memoized_method
    def cumulative_bias(self, m = None, alpha = None, M = None, assume_correct = 50):
        '''
        Compute the bootstap bias of the number of elements occuring m times.
        '''
          
        N = self.num_tokens()
        if not M: M = N
   
        (index, pre_added, n_cumulative, p_cumulative, np_cumulative, mu_agg) = self.find_dist_point(min_weight = (assume_correct * N) / M)
        e_cumulative = n_cumulative    
    
        while index < len(self.x):
            
            approx = int(round(self.x[index][0] * M / float(N)))            
            if approx < m: break
            #print index, approx, p_cumulative, e_cumulative

            e_cumulative += self.expected_sample(approx, M)
            while index < len(self.x) and int(round(self.x[index][0] * M / float(N))) == approx:
                n_cumulative += self.x[index][1]
                index += 1        

        return e_cumulative / n_cumulative

    def subsampled(self, M, title = None, track_spectrum = False, track_cross_mu_up = None, track_cross_mu_down = False):
        '''
        Compute a random subsample, without replacement, with M elements
        '''

        if M > self.total_weight: return self
        if not title: title = self.title + '(% d)' % M
        new_dist = dist(title = title, clean_title = False)    

        additive = True
        if M > self.total_weight / 2:
            M = self.total_weight - M
            additive = False  #sample by removing instead of adding

        #random sampling WITHOUT replacement--simulates stopping the sample earlier
        selected = sorted(random.sample(xrange(int(self.num_tokens())), M))

        s = 20
        R = 2
        if track_spectrum:         
            track_spectrum_max = int(R * s * self.total_weight / float(M))
            sample_effect = [[0,] * track_spectrum_max for i in range(s + 1)]

        if track_cross_mu_up > 0:
            cross_mu_up = [0.0,]
            track_np_cumulative = 0
            track_n_cumulative = 0
    
        if track_cross_mu_down:
            track_cross_mu_pairs = {}

        scanned = 0
        index = 0
        hist = {}
        for i in self.items():
            w = i[0]
            n = i[1]
            #add all final elements at once
            if w == 1:
                hist[1] = hist.get(1, 0) + ((len(selected) - index) if additive else (n - len(selected) + index))
                if track_spectrum: 
                    sample_effect[1][1] += ((len(selected) - index) if additive else (n - len(selected) + index))
                    sample_effect[0][1] += ((n - len(selected) + index) if additive else (len(selected) - index))
                if track_cross_mu_up:                    
                    while len(cross_mu_up) < 100:
                        #print track_np_cumulative, track_n_cumulative, M, ((len(selected) - index) if additive else (n - len(selected) + index))
                        next = (len(cross_mu_up) * M) / track_cross_mu_up
                        add = int(math.ceil((next - track_np_cumulative) / (float(M) / self.total_weight)))
                        track_n_cumulative += add 
                        track_np_cumulative += add * (float(M) / self.total_weight)
                        cross_mu_up.append(math.log(track_n_cumulative / (float(track_np_cumulative) / M), 2))

                if track_cross_mu_down:
                    track_cross_mu_pairs[(1,1)] = ((len(selected) - index) if additive else (n - len(selected) + index))

                break           

            for j in xrange(n):
                count = 0 if additive else w

                while index < len(selected) and selected[index] < scanned + w:
                    index += 1
                    count += 1 if additive else -1
                    
                if count > 0: 
                    hist[count] = hist.get(count, 0) + 1
                    
                if track_spectrum and count <= s and w < track_spectrum_max:
                    sample_effect[count][int(w)] += 1
                
                if track_cross_mu_up:
                    track_n_cumulative += 1
                    track_np_cumulative += count
                    #print track_np_cumulative, track_n_cumulative, M
                    while (track_cross_mu_up * track_np_cumulative / M)  >= len(cross_mu_up):
                        #print len(cross_mu_up), (track_cross_mu_up * track_np_cumulative / M)
                        cross_mu_up.append(math.log(track_n_cumulative / (float(track_np_cumulative) / M), 2))

                if track_cross_mu_down and count >= 1:
                    track_cross_mu_pairs[(count, w)] = track_cross_mu_pairs.get((count, w), 0) + 1

                scanned += w

        new_dist.x = [[i[0], i[1]] for i in sorted(hist.items(), key = lambda x: x[0], reverse = True)]
        new_dist.compute_stats(adjust_total = False)
        extra = []
        if track_spectrum: extra.append(sample_effect)
        if track_cross_mu_up: extra.append(cross_mu_up)

        if track_cross_mu_down:
            s = sorted(track_cross_mu_pairs.items(), key = lambda x: x[0][0], reverse = True)
            r = []
            i = 0
            #print s
            current = -1
            for i in range(len(s)):
                if len(r) == 0 or s[i][0][0] != current:
                    r.append([s[i][0][1] * s[i][1], s[i][1]])
                    current = s[i][0][0]
                else:
                    r[-1][0] += s[i][0][1]* s[i][1]
                    r[-1][1] += s[i][1]
            #print r
            n_cumulative = 0
            p_cumulative = 0
            cross_mu_down = [0.0]
            for i in range(len(r)):
                p = (r[i][0] / r[i][1]) / self.total_weight
                n = r[i][1]
                added = 0
                #print r[i], p, n
                while added < n:   
                    to_add = min(n, int(math.ceil(((float(len(cross_mu_down)) / track_cross_mu_down) - p_cumulative) / p)))
                    #print n, int(math.ceil((float(len(cross_mu_down)) / track_cross_mu_down) - p_cumulative) / p)
                    #print to_add, to_add * p, (float(len(cross_mu_down)) / track_cross_mu_down) - p_cumulative)
                    n_cumulative += to_add
                    p_cumulative += to_add * p
                    #print n_cumulative, p_cumulative
                
                    while (p_cumulative * track_cross_mu_down > len(cross_mu_down)):
                        cross_mu_down.append(math.log(n_cumulative / p_cumulative, 2))
                        #print len(cross_mu_down) / 100.0, math.log(n_cumulative / p_cumulative, 2)

                    added += to_add

            extra.append(cross_mu_down)

        if extra:
            return [new_dist,] + extra
        else:
            return new_dist

    '''
    ----------------------------------------------------------
    Plotting functions
    ----------------------------------------------------------
    '''

    @staticmethod
    def plot_marginal_guesswork(ds, **kwargs):
        '''
        Plot the marginal guesswork for a series of distributions.
        '''
        lines = []

        plot_type = kwargs.pop('plot_type', 'mu')
        tilde_value = kwargs.pop('tilde_value', True)


        only_G = True

        count = 0
        for x in ds:        
            if plot_type == 'mu':
                y_func = lambda d, i: d.mu(i, tilde_value = tilde_value)
                x_func = lambda d, i: i

            if plot_type == 'gini_absolute': 
                y_func = lambda d, i: d.p_below(float(i) / d.num_types()) * d.num_tokens()
                x_func = lambda d, i: int(i * d.num_types())
            if plot_type == 'gini_percent': 
                y_func = lambda d, i: d.p_below(i) * d.num_tokens()
                x_func = lambda d, i: i
    
            if x is None: continue

            d = x[0]
            props = x[1] if len(x) > 1 else {}

            props.setdefault('color', plot_colors[count % len(plot_colors)])
            count += 1

            split_plot = props.pop('split_plot', True) #indicate the head/tail split by changing lines
            proj_func = None
            if props.pop('show_projected', True): proj_func = lambda x: x.gigp_projected() 
            elif props.pop('show_lin_projected', False): proj_func = lambda x: x.lin_projected() 
            elif props.pop('show_zm_projected', False): proj_func = lambda x: x.zm_projected() 
            show_tail = props.pop('show_tail', False) 
            show_head = props.pop('show_head', True)
            step_size = props.pop('step_size', 0.01)
            min_weight = props.pop('min_weight', 6)
            unsmoothed_mu = props.pop('unsmoothed_mu', False)
            G_value = props.pop('g_value', False)

            if unsmoothed_mu and tilde_value:
                y_func = lambda d, i: math.log((d.mu(i, tilde_value = False) / i), 2)
            elif G_value:
                y_func = lambda d, i: d.G(i, tilde_value = tilde_value)

            if not G_value: only_G = False

            split = int(round(d.confidence_p(min_weight) / step_size))
            plot_length = int(round(1.0 / step_size))

            if show_head:
                p = props.copy()
                p.setdefault('linestyle', '-' if split_plot else '-')
                p.setdefault('title', d.title)
                x = [x_func(d, i * step_size) for i in range(1, min(split, plot_length))]
                y = [y_func(d, i) for i in x]
                lines.append((x, y, p))

            if show_tail:
                p = props.copy()
                p.setdefault('linestyle', '--' if split_plot else '-')
                p['title'] = None
                x = [x_func(d, i * step_size) for i in range(max(split - 1, 1), plot_length)]
                y = [y_func(d, i) for i in x]
                lines.append((x, y, p))

            if proj_func:
                p = props.copy()
                p.setdefault('linestyle', ':' if split_plot else '-')
                p['title'] = None
                x = [x_func(proj_func(d), i * step_size) for i in range(1,plot_length)]
                y = [y_func(proj_func(d), i) for i in x]
                lines.append((x, y, p))

        if plot_type == 'mu':
            kwargs.setdefault('x_title', 'success rate $\\alpha$')

            if only_G:
                if tilde_value:
                    kwargs.setdefault('y_title', r'$\alpha$-guesswork \  $\tilde{G}_\alpha$ (bits)')
                else:
                    kwargs.setdefault('y_title', r'$\alpha$-guesswork \  $G_\alpha$ (\# guesses)')
            else:
                if tilde_value:
                    kwargs.setdefault('y_title', r'$\alpha$-work-factor \  $\tilde{\mu}_\alpha$ (bits)')
                else:
                    kwargs.setdefault('y_title', r'$\alpha$-work-factor \  $\mu_\alpha$ (\# guesses)')

        if plot_type == 'gini_percent':
            kwargs.setdefault('x_title', '\\% of total types')
            kwargs.setdefault('y_title', '\\% of total population')
        if plot_type == 'gini_absolute':
            kwargs.setdefault('x_title', 'number of types')
            kwargs.setdefault('y_title', '\\% of total population')

        second_scale = kwargs.pop('second_scale', None)
        if second_scale == '10':
            kwargs.setdefault('y2_transform', lambda x: x * math.log(2, 10))
            if only_G:
                kwargs.setdefault('y2_title', r'$\alpha$-guesswork \  $\tilde{G}_\alpha$ (dits)')
            else:
                kwargs.setdefault('y2_title', r'$\alpha$-work-factor \  $\tilde{\mu}_\alpha$ (dits)')
        elif second_scale == 'e':
            kwargs.setdefault('y2_transform', lambda x: x * math.log(2, math.e))
            if only_G:
                kwargs.setdefault('y2_title', r'$\alpha$-guesswork \ $\tilde{G}_\alpha$ (nats)')
            else:
                kwargs.setdefault('y2_title', r'$\alpha$-work-factor \  $\tilde{\mu}_\alpha$ (nats)')

        kwargs.setdefault('zero_yaxis', True)
        extra_lines = kwargs.pop('extra_lines', [])
        plot_data(lines + extra_lines, **kwargs)


    @staticmethod
    def plot_frequency_distribution(ds, **kwargs):
        '''
        Plot frequency distributions
        '''
        lines = []

        cumulative = kwargs.pop('cumulative', False)

        count = 0
        
        for m in ds:
            if m is None: continue

            d = m[0] 
            props = m[1] if len(m) > 1 else {}
            props.setdefault('title', d.title)

            x = []
            y = []
            for j in range(len(d.x)):      
                i = d.x[::-1][j]
                x.append(int(round(i[0])))
                y.append(int(round(i[1])))
                if cumulative and len(y) > 1:
                    y[-1] += y[-2]

                #print x, y

            props.setdefault('color', plot_colors[count % len(plot_colors)])
            count += 1

         
            lines.append((x, y, props))

        kwargs.setdefault('x_title', '$m$ (observed count)')
        if cumulative:
            kwargs.setdefault('y_title', '$g(m, M)$')
        else:
            kwargs.setdefault('y_title', '$V(m, M)$')

        kwargs.setdefault('zero_yaxis', True)

        plot_data(lines, **kwargs)

    '''
    ----------------------------------------------------------
    Power-law functions
    ----------------------------------------------------------
    '''

    @staticmethod
    def find_beta(x, y, lbt, ubt, beta_range = range(0, 20), binned = False):

        if binned: 
            (x,y) = map(exp_bin, (x, y))
            lbt = 1 + int(math.log(lbt, 2))
            ubt = 1 + int(math.log(ubt, 2))

        bs = []
        for bt in beta_range:                
            xt = map(lambda k: k+bt, x)
            (a, b, func, y_est, err) = log_regression(xt[lbt:ubt], y[lbt:ubt], x_plot = x, use_binary_log = True, compute_error = True)
            bs.append((bt, err))
            #print bt, err

        b = sorted(bs, key = lambda x: x[1])[0][0]
        return b


    def prep_zm_law(self, beta_range = None):

        d = {'x': None, 'y': [], 'ub': None, 'lb': None}

        
        for j in self.items():
            #exclude items which occur less than twice, or a unique number of times
            if j[1] > 1 and not d['lb']: d['lb'] = len(d['y']) 
            if j[0] < 2 and not d['ub']: d['ub'] = len(d['y']) 
            d['y'].extend([float(j[0]) / self.total_weight,] * j[1])

        #print d['y']

        if beta_range is not None:
            d['beta'] = self.find_beta(range(1, len(d['y']) + 1), d['y'], d['lb'], d['ub'], beta_range)
        else:
            d['beta'] = 0
        d['x'] = range(1 + d['beta'], len(d['y']) + 1 + d['beta'])

        d['min_line'] = ((d['x'][1], d['x'][len(d['x']) - 1]), [1.0 / self.total_weight] * 2)

        d['x_title'] = '$\\text{rank}' + (' + %d' % d['beta'] if d['beta'] > 0 else '') + '$'
        d['y_title'] = '$p$'

        return d

    def prep_power_law(self, pareto = True):


        d = {'x': [0.0] * len(self.x), 'y': [0.0] * len(self.x), 'ub': None, 'lb': None}

        i = len(d['x']) - 1
        cumulative_types, cumulative_p = 0.0, 0.0

        for j in self.items():
            d['x'][i] = float(j[0]) / self.total_weight
            next_p = float(j[1]) / self.stats['total_types']
            cumulative_types += next_p
            cumulative_p += j[0] * j[1] / self.num_tokens()
            d['y'][i] = cumulative_types if pareto else next_p
            #if cumulative_p >= 0.2 and not d['lb']: d['lb'] = i
            #if cumulative_p >= 0.1 and not d['ub']: d['ub'] = i
            i -= 1

        d['lb'] = 5
        d['ub'] = 10

        d['min_line'] = ([d['x'][0]] * 2, ([0], d['y'][len(d['y']) - 1]))

        d['x_title'] = '$p$'
        d['y_title'] =  '$\\text{proportion of tokens} > p$' if pareto else '$\\text{proportion of tokens}$'


        return d

    def fit_power_law(self, d, binned = False, binary_log = True):    

        log_f = (lambda x: math.log(x, 2)) if binary_log else (lambda x: math.log(x))
        log_string = '\\lg' if binary_log else '\\ln'

        d['x_title'] = '$%s(%s)$' % (log_string,d['x_title'].strip('$'))
        d['y_title'] = '$%s(%s)$' % (log_string,d['y_title'].strip('$'))

        if binned: 
            (d['x'], d['y']) = map(exp_bin, (d['x'], d['y']))
            if d['lb']: d['lb'] = 1 + int(log_f(d['lb']))
            if d['ub']: d['ub'] = 1 + int(log_f(d['ub']))

        if not d['lb']: d['lb'] = 0
        if not d['ub']: d['ub'] = len(d['x']) - 1
        d['ub'] = max(d['ub'], d['lb'] + 2)

        #print d['x'], d['y']
        (d['a'], d['b'], func, foo, bar) = log_regression(d['x'][d['lb']:d['ub']], d['y'][d['lb']:d['ub']], x_plot = d['x'], use_binary_log = binary_log)

        d['y_est_func'] = lambda x: d['b'] * (x ** d['a'])
        d['y_est_title'] = '$y = %0.6f \\cdot x^{%0.2f}$' % (d['b'], d['a'])

        return d

    def plot_power_law(self, override_props = {}, show_best_fit = True, show_min_p = True, min_weight = 4, min_support = 15, style = 'loglog_zipf', binned = True, binary_log = True, b = None, **kwargs):


        if style == 'loglog_zipf' or style == 'loglog_zipf_mandelbrot':
            d = self.fit_power_law(self.prep_zm_law((range(0, 20) if style == 'loglog_zipf_mandelbrot' else None)), binned = binned, binary_log = binary_log)

        elif style == 'loglog_pareto' or style == 'loglog_power':
            d = self.fit_power_law(self.prep_power_law(pareto = (style == 'loglog_pareto')), binned = binned, binary_log = binary_log)
                                    
        else:
            sys.stderr.write('Unknown style: %s\n' % style)
            return

        kwargs.setdefault('x_title', d['x_title'])
        kwargs.setdefault('y_title', d['y_title'])

        np_log_f = (numpy.log2 if binary_log else numpy.log)

        ls = []
        props = {'linestyle': '', 'marker': '.', 'color': 'b'}
        props.update(override_props)
        #excluded low points
        ls.append((np_log_f(numpy.array(d['x'][:d['lb']])), np_log_f(numpy.array(d['y'][:d['lb']])), props, self.title))

        props = {'linestyle': '', 'marker': '.', 'color': 'g'}
        props.update(override_props)
        #middle points
        ls.append((np_log_f(numpy.array(d['x'][d['lb']:d['ub']])), np_log_f(numpy.array(d['y'][d['lb']:d['ub']])), props, None))

        props = {'linestyle': '', 'marker': '.', 'color': 'm'}
        props.update(override_props)
        #excluded high points
        ls.append((np_log_f(numpy.array(d['x'][d['ub']:])), np_log_f(numpy.array(d['y'][d['ub']:])), props, None))

        if show_best_fit:
            ls.append((np_log_f(numpy.array(d['x'])), np_log_f(numpy.array([d['y_est_func'](k) for k in d['x']])), {'marker': '', 'linestyle': '-', 'color': 'r', 'title': d['y_est_title']}))
    
        if show_min_p:
            ls.append((np_log_f(d['min_line'][0]), np_log_f(d['min_line'][1]), {'marker': '', 'linestyle': ':', 'color': 'k', 'title': '$p = \\frac{1}{N}$'},))

        kwargs.setdefault('legend_placement', 'upper right')

        plot_data(ls, **kwargs)

    def legacy_rename(self):
        '''
        Replace old attributes names for class read in from cPickle object
        '''
        def replace(old_name, new_name):
            if hasattr(self, old_name):
                setattr(self, new_name, getattr(self, old_name))
                delattr(self, old_name)

        replace('knownWeight', 'known_weight')
        replace('totalWeight', 'total_weight')
        replace('knownTypes', 'known_types')
        replace('totalTypes', 'total_types')

    '''
    ----------------------------------------------------------
    Sampling functions
    ----------------------------------------------------------
    '''

    '''
    ----------------------------------------------------------
    Projection functions
    ----------------------------------------------------------
    '''


    @memoized_method
    def fit_gigp(self, **kwargs):

        #RY full
        #0.00080078125 0.004638671875 -1.09619140625

        #RY 1 M
        #0.00061279296875 0.00188232421875 -1.05261230469


        def coeff1(n, b, c, g):
                return (1.0 / (((1 + c * n) ** (g / 2.0)) * mpmath.besselk(g, b) - mpmath.besselk(g, b * math.sqrt(1 + c * n))))
        
        def coeff2(r, n, b, c):
            return mpmath.power(((b * c * n) / (2.0 * math.sqrt(1 + c * n))),  r) / mpmath.fac(r)

        def coeff3(r, n, b, c, g):
            return mpmath.besselk(r + g, b * math.sqrt(1 + c * n))

        def ptsich3(r, n, b, c, g):
            return coeff1(n, b, c, g) * coeff2(r, n, b, c) * coeff3(r, n, b, c, g)


        def log_likelihood(b, c, g):

            if b <= 0.0 or c <= 0.0: return None

            n = self.num_tokens()
            l = 0

            try:
                c1 = coeff1(n, b, c, g)
            except ZeroDivisionError:
                #print '\t**Zero Div Error 14 b,c,g=', b, c, g
                return None

            if c1 < 0: return None

            c2_top = math.log((b * c * n) / (2.0 * math.sqrt(1 + c * n)))                
            c2 = 0

            c3p = [None, None]            
            q = b * math.sqrt(1 + c * n)

            index = 0           
 
            l += self.num_types() * math.log(c1)
            r = 0
            for j in range(len(self.x) - 1, -1, -1):
                i = self.item(j, known_only = False)
                #print i                    
                while (r < i[0] - 50):#big jump if not close
                    r = i[0]
                    c3 = coeff3(r, n, b, c, g)
                    c2 = float(mpmath.log(coeff2(r, n, b, c)))
                    c3p = [None, None]            

                while(r < i[0]):
                    r += 1
                    c3 = None
                    c2 = c2 + c2_top - math.log(r)

                    if c3p[1] is None or c3p[0] is None:
                        c3 = coeff3(r, n, b, c, g)
                    else:
                        c3 = (2 * (r + g - 1) / q) * c3p[1] + c3p[0]

                    c3p[0] = c3p[1]
                    c3p[1] = c3

                    if c3 < 0: print '******', b, c, g; return None
                    l += (c2 + float(mpmath.log(c3))) * i[1]
                    #testing code
                    #if (abs(c2 - mpmath.log(coeff2(r, n, b, c))) / c2) > 0.0000001:
                    #    print 'c2 mismatch: ', c2, mpmath.log(coeff2(r, n, b, c))
                    #    print 'r, n, b, c, g:', r, n, b, c, g
                    #if (abs(c3 - coeff3(r, n, b, c, g)) / c3) > 0.0000001:
                    #    print 'c3 mismatch: ', c3, coeff3(r, n, b, c, g)
                    #    print 'r, n, b, c, g:', r, n, b, c, g
            return l

        sys.stdout.write('Optimizing parameters of GIGP distribution')
        sys.stdout.flush()
        count = 0
        b, c, g = 0.001, 0.01, -1
        #b, c, g = 0.0008125, 0.12625, -1
        d = [0.001, 0.01, 0.25]
        #d = [6.25e-05, 0.000625, 0.015625]
        l = 0
        

        while count < kwargs.get('num_steps', 100):
            sys.stdout.write('.')
            sys.stdout.flush()
            #print b, c, g, d
            g_diff = (-d[2], 0, d[2]) if kwargs.get('g_free', False) else (0,) #do we allow g to float?
            neighbors = []
            for (d1, d2, d3) in itertools.product((-d[0], 0, d[0]),(-d[1], 0, d[1]), g_diff):
                neighbors.append((log_likelihood(b + d1, c + d2, g + d3), (d1, d2, d3)))
            best = sorted(filter(lambda x: x[0] is not None, neighbors), key = lambda x: x[0], reverse = True)[0]
            #print 'k'
            if best[1] == (0, 0, 0): 
                d = map(lambda x: x/2, d)
            b, c, g = b + best[1][0], c + best[1][1], g + best[1][2]
            l = best[0]
            count += 1
            #print 'kk'

        return (b, c, g, l)

    def skip_projection(self):
        self.skip_projection = True

    @memoized_method
    def gigp_projected(self, **kwargs):

        if 'skip_projection' in dir(self) and self.skip_projection == True: return self

        #don't need to project
        if self.confidence_p(6) >= 0.99: return self

        cutoff = kwargs.pop('cutoff', 10)

        #compute underlying distribution
        b, c, g, l = self.fit_gigp(use_memoized = False, **kwargs)
        #print b, c, g, l
        coeff = ((2 ** (g-1)) / (((b * c) ** g) * mpmath.besselk(g, b)))
        def gigp(x):
            if x <= 0 or x >= 1: return 0
            return coeff * mpmath.power(x, g - 1) * mpmath.exp(-x/c - (c * b ** 2) / (4 * x))

        new_dist = dist(title = self.title, clean_title = False)    

        #add in old items
        new_dist.x = []
        filled = 0.0
        for i in self.items():
            if i[0] < cutoff: break
            new_dist.x.append(i)
            filled += i[0] * i[1]

        filled /= self.num_tokens()        
        current = float(cutoff) / self.num_tokens()
        remainder = []

        if filled >= 1.0: return self

        step = 0
        i = 1
        base = None

        peak = current / 2
        step = current / 4

        #find peak of distribution to focus samples around
        while step > 10 ** -20:
            if gigp(peak + step) > gigp(peak):
                peak += step
            elif gigp(peak - step) > gigp(peak):
                peak -= step
            else:
                step /= 2
        base_step = current / 100
        min_step = peak / 10000  

        while current > 0:
            next_step = min(base_step, current)
            #print next_step          
            #take smaller steps closer to peak of the distribution
            while ((current == next_step and current > peak) or 2 * abs(current - peak) < next_step) and next_step > min_step:
                next_step /= 100               
            coarse, fine = 0,0

            #take smaller steps if it makes a difference in the integral quality
            while next_step >= 10 * min_step:
                coarse = riemann_integrate(gigp, current - next_step, current, 10)
                fine = riemann_integrate(gigp, current - next_step, current, 100)
                if (abs(fine - coarse) / fine) < 0.001: break
                next_step /= 10
            if fine == 0: 
                fine = riemann_integrate(gigp, current - next_step, current, 100)     

            next_p = float(self.num_tokens() * (current - 0.5 * next_step))
            next_w = float(self.num_types() * fine)
            remainder.append([next_p, next_w])
            current -= next_step

        total = sum(map(lambda x: x[0] * x[1], remainder)) / ((1 - filled) * self.num_tokens())
        remainder = map(lambda x: [x[0] , float(x[1]/ total)], remainder)
    

        min_w = 10
        i = 0
        while i < len(remainder):
            next = remainder[i] 

            while next[1] < min_w and i < len(remainder) - 1:
                i += 1
                if remainder[i][1] + next[1] <= 0 : break
                next[0] = (next[0] * next[1] + remainder[i][0] * remainder[i][1]) / (remainder[i][1] + next[1])
                next[1] += remainder[i][1]
                
            if int(next[1]) >= 1:
                fixed = [next[0] * next[1] / float(int(next[1])), int(next[1])]
                new_dist.x.append(fixed)
            i += 1   

        new_dist.compute_stats(adjust_total = True)
        return new_dist


    @memoized_method
    def lin_projected(self, step_size = 0.01, min_weight = 6):
        '''
        Linear projection of line, computed from projecting the mu plot.
        '''

        #number of original items to keep
        kept_p = min(self.confidence_p(min_weight), 1.0)
        
        #enough data to be confident in our accuracy
        if kept_p >= 1.0: return self

        #upper and lower bounds of points to use for our linearization
        #upper_p = max(0.3, kept_p + 10 * step_size)
        #upper_p = min(self.confidence_p(2), 1.0)
        upper_p = kept_p + 8 * step_size
        lower_p = kept_p

        #print lower_p, upper_p

        #x an y values to linearize
        xt = numpy.arange(lower_p, upper_p, step_size)
        yt = [self.mu(i, tilde_value = True) for i in xt]

        #linearize values
        (a, b, y_est_func, y_est, err) = linear_regression(xt, yt, compute_error = True)

        new_dist = dist(title = self.title, clean_title = False)    
        p_cumulative, n_cumulative = 0.0, 0

        #add in original items up to the confidence point
        for i in self.items():
            n, p = i[1], float(i[0]) / self.num_tokens()
            added = min(int(math.ceil((kept_p - p_cumulative) / p)), n)
            if added > 0: new_dist.x.append(i[:1] + [added,] + i[2:])
            p_cumulative += added * p
            n_cumulative += added
            if p_cumulative >= kept_p: break

        #add in new items to make up the projected values of mu            
        while p_cumulative < 1.0:
            next_p = min(p_cumulative + step_size, 1.0)
            est_mu = y_est_func(next_p)
            n_best = max(int((2 ** est_mu) * next_p) - n_cumulative, 1)
            
            p = min((next_p - p_cumulative) / n_best , p)
            n = int(math.ceil((next_p - p_cumulative) / p))

            if n > 0: 
                w = p * self.num_tokens()
                new_dist.x.append([w,n,])
                n_cumulative += n
                p_cumulative += p * n

        new_dist.compute_stats(adjust_total = False)
        return new_dist


    @memoized_method
    def power_projected(self, n_blocks = 100, min_weight = 6):
        '''
        Power law projection using MLE fit (Clauset et al.)
        '''

        #number of original items to keep
        kept_p = min(self.confidence_p(min_weight), 1.0)
        
        #enough data to be confident in our accuracy
        if kept_p >= 1.0: return self

        new_dist = dist(title = self.title, clean_title = False)    
        p_cumulative, n_cumulative = 0.0, 0

        #add in original items up to the confidence point
        for i in self.items():            
            if i[0] < min_weight: break
            n, p = i[1], float(i[0]) / self.num_tokens()
            added = min(int(math.ceil((kept_p - p_cumulative) / p)), n)
            if added > 0: new_dist.x.append(i[:1] + [added,] + i[2:])
            p_cumulative += added * p
            n_cumulative += added

        a = self.fit_power_law_ml()

        fixed_proportion = (p * self.num_tokens()) ** (1-a)
        #print p * self.num_tokens()
        #print fixed_proportion, n_cumulative
        #print p_cumulative, float(n_cumulative) / self.num_types()

        p_step = (1 - p_cumulative) / n_blocks

        #add in new items to make up the projected values of mu            
        while p_cumulative < 1.0:
            next_p = min(p_cumulative + p_step, 1.0)
            w = math.pow(next_p, 1.0 / (1 - a)) - 1
            p = float(w) / self.num_tokens()
            if p == 0: break
            n = int((next_p - p_cumulative) / p)
            if next_p == 1.0: p = (1.0 - p_cumulative) / n
            #print n, p, p * self.num_tokens()
            if n > 0: 
                w = p * self.num_tokens()
                new_dist.x.append([w,n,])
                n_cumulative += n
                p_cumulative += p * n
            if next_p >= 1.0: break

        new_dist.compute_stats(adjust_total = False)
        return new_dist

    @memoized_method
    def zm_projected(self, block_size = 1000, min_weight = 6):
        '''
        Zipf-Mandelbrot fit using linear least square regression
        '''

        #number of original items to keep
        #print '*'
        kept_p = min(self.confidence_p(min_weight), 1.0)
        
        #enough data to be confident in our accuracy
        if kept_p >= 1.0: return self

        new_dist = dist(title = self.title, clean_title = False)    
        p_cumulative, n_cumulative = 0.0, 0

        #print '*'

        #add in original items up to the confidence point
        for i in self.items():
            n, p = i[1], float(i[0]) / self.num_tokens()
            added = min(int(math.ceil((kept_p - p_cumulative) / p)), n)
            if added > 0: new_dist.x.append(i[:1] + [added,] + i[2:])
            p_cumulative += added * p
            n_cumulative += added
            if p_cumulative >= kept_p: break

        try:
            d = self.fit_power_law(self.prep_zm_law(beta_range = None), binned = True, binary_log = True)
        except MemoryError:
            sys.stderr.write("Out of memory doing ZM projection for %s\n" % self.title)
            return self

        fixed_proportion = p * (n_cumulative ** -d['a'])

        #something's gone horribly wrong
        if math.isnan(d['a']): return self

        #print n_cumulative, p, d['a'], fixed_proportion        

        #print '**'

        change_p = 2.0 ** (-30)
        min_p = 2.0 ** (-40)
        next_block_size = block_size
        #add in new items to make up the projected values of mu
        while p_cumulative < 1.0:
            #print len(new_dist.x)
            #print n_cumulative, p_cumulative
            next_n_mid = int(math.ceil(n_cumulative + next_block_size / 2))
            p = max((next_n_mid ** d['a']) * fixed_proportion, min_p) #insert 
            #print p, next_n_mid
            n = min(int(math.ceil((1.0 - p_cumulative) / p)), next_block_size) if p > 0 else next_block_size

            #print n, p
            if n > 0: 
                w = p * self.num_tokens()
                new_dist.x.append([w,n,])
                n_cumulative += n
                p_cumulative += p * n

            try:
                next_block_size = max(block_size, int((change_p / p) * block_size))
            except OverflowError:
                print p, min_p, block_size
                break

        new_dist.compute_stats(adjust_total = False)  
     
        return new_dist


    @memoized_method
    def sgt_plot(self, **kwargs):

        #counts and weights, in Gale/Sampson terminology
        #place the weights backwards, least common items first
        r = [self.item(i)[0] for i in range(len(self.x) - 1, -1, -1)]
        n = [self.item(i)[1] for i in range(len(self.x) - 1, -1, -1)]

        #calculate smoothed counts z to replace n
        z = [0.0] * (len(r))    
        for i in range(len(r)):
            prev = (r[i - 1] if i > 0 else 0)
            next = (r[i + 1] if i < len(r) - 1 else 2 * r[i] - prev)
            z[i] = 2 * float(n[i]) / abs(next - prev)

        #fit power law to the r/z line
        (a, b, s_func, foo, bar) = log_regression(r, z)

        fit = map(s_func, r)

        plot_data(((r, z, {'title' : 'smoothed', 'linestyle': '', 'marker': 'o','markersize': 5, 'color': 'g'}),(r, n, {'title' : 'unsmoothed', 'linestyle': '', 'marker': '^','markersize': 5, 'color': 'r'}),(r, fit, {'title' : '$Z(m) = %.3f \cdot m^{%.3f}$' % (b, a), 'linestyle': '-', 'color': 'k'})), show_plot = False, log_plot = True, x_title = '$m$', y_title = '$V(m, M)$', **kwargs)

    @memoized_method
    def sgt_prob(self, index = 1):
        return self.sgt()[- index]
        

    @memoized_method
    def sgt(self):
        '''
        Simple Good-Turing smoothing for the distribution.
        Each element's weight is replaced by the SGT approximation.
        '''
        #algorithm doesn't make sense if lowest count isn't 1
        #if self.x[len(self.x) - 1][0] != 1: return self 

        #counts and weights, in Gale/Sampson terminology
        #place the weights backwards, least common items first        
        n = [self.item(i)[1] for i in range(len(self.x) - 1, -1, -1)]
        r = [self.item(i)[0] for i in range(len(self.x) - 1, -1, -1)]

        #calculate smoothed counts z to replace n
        z = [0.0] * (len(r))    
        for i in range(len(r)):
            prev = (r[i - 1] if i > 0 else 0)
            next = (r[i + 1] if i < len(r) - 1 else 2 * r[i] - prev)
            z[i] = 2 * float(n[i]) / abs(next - prev)

        #fit power law to the r/z line
        (a, b, s_func, foo, bar) = log_regression(r, z)

        #compute the correct probabilities two ways
        #Change method when errors get smaller (smoothing without tears)
        r_star = [0] * len(r)

        #the number of items using the x-estimate
        using_x = None
        #cumulative probability of items using the x-estimate
        for i in range(len(r)):
            n_rplusone = float(n[i + 1] if i < len(r) - 1 and r[i + 1] == r[i] + 1 else 0)
            n_r = float(n[i])
            x_est = (r[i] + 1) * (n_rplusone / n_r) if not using_x else None
            y_est = (r[i] + 1) * (s_func(r[i] + 1) / s_func(r[i]))

            if not using_x:
                error = 1.96 * math.sqrt( ((r[i] + 1) ** 2) * (n_rplusone / (n_r ** 2)) * (1 + (n_rplusone/n_r)))
                if abs(x_est - y_est) <= error:
                    #print 'switching at i=', i
                    using_x = True
    
            r_star[i] = (y_est if using_x else x_est)

        return r_star

    @memoized_method
    def sgt_projected(self):
        '''
        Crude Good-Turing extrapolation.
        Each element's weight is replaced by the SGT approximation.
        New items of the smallest seen weight are added to the tail.
        '''        

        if self.x[len(self.x) - 1][0] != 1: return self 

        r_star = self.sgt()
        n = [self.item(i)[1] for i in range(len(self.x) - 1, -1, -1)]

        p_0 = self.x[-1][0] / self.total_weight
        #print r_star
        #print n
        total_estimated = sum(r_star[i] * n[i] for i in range(len(r_star)))
        #print total_estimated, self.total_weight * (1 - p_0), self.total_weight, self.num_tokens()

        new_dist = dist(title = self.title, clean_title = False)
        new_dist.x = [[r_star[len(self.x) - i - 1],] + self.item(i)[1:] for i in range(len(self.x))]
        #add in extra unseen items
        new_dist.x[len(self.x) - 1][1] += int((self.total_weight - total_estimated) / r_star[0])

        new_dist.compute_stats(adjust_total = False)

        return new_dist


