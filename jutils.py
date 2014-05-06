#!/usr/bin/python
# vim: set fileencoding=UTF-8

'''
Utility routines
Statistical wrappers, power law fitting, regression

January 2011
Joseph Bonneau
jcb82@cl.cam.ac.uk
'''

import os
from scipy import polyval, polyfit, sqrt
from matplotlib import pyplot, rc, rcParams
from pylab import show, loglog, FuncFormatter, MultipleLocator
import numpy
from decorator import decorator
from rpy import r

plot_colors = ['b', 'g', 'r', 'm', 'c', 'y', 'k']
plot_linestyles = ['-', '--', '-.', ':']

#Parameters for LaTex output
rc('font',**{'family':'serif','serif':['Computer Modern']})
rc('text', usetex=True)
rcParams['text.latex.preamble']=[r"\usepackage{amsmath}",]

def make_hist(x):
    result = {}
    for i in x:
        result[i] = result.get(i, 0) + 1

    return sorted(result.items(), key = lambda x: x[0])

def determination_coefficient(x, y, adjusted = None):
    '''Compute the R^2 coefficient of determination between x and y'''

    xbar = float(sum(x)) / len(x)
    sstot = sum([(x[i] - xbar)**2 for i in range(len(x))])
    sserr = sum([(y[i] - x[i])**2 for i in range(len(x))])

    result = 1 - float(sserr) / float(sstot)

    if adjusted: result = 1 - (1 - result) * (float(adjusted[0] - 1) / float(adjusted[0] - adjusted[1] - 1))

    return result

def linear_residuals(y, yr):
    '''Compute the linear residuals between two lists of values'''
    return map(lambda x: (float(x[1] - x[0])), zip(y, yr))

def linear_error(y, yr):
    '''Compute the average linear error between two lists of values'''
    return sum(map(abs, linear_residuals(y, yr))) / len(y)

def error_residuals(y, yr):
    '''Compute the error residuals between two lists of values'''
    return map(lambda x: (float(x[1] - x[0]) ** 2), zip(y, yr))

def least_squares_error(y, yr):
    '''Compute the least squares error between two lists of values'''
    return sqrt(sum(error_residuals(y, yr)) / len(y))

def linear_regression(x, y, compute_error = False):
    '''Compute a linear best fit between lists x and y, y = a * x + b'''
    (a,b)=polyfit(x,y,1)
    yr=polyval([a,b],x)
    err=sqrt(sum((yr-y)**2)/len(y)) if compute_error else None

    func = lambda x: a * x + b

    return (a, b, func, yr,  err)

def log_regression(x, y, x_plot = None, use_binary_log = False, compute_error = False):
    '''Compute a best fit of y = b * x ^ a by projecting to log-log space.
    If use_binary_log is set, 2 is the base, otherwise base e is used
    '''
    if not x: return (0, 0, None, None, None)

    #take logarithms of both lists
    #print x, y
    log_function = numpy.log2 if use_binary_log else numpy.log
    log_x = log_function(x); log_y = log_function(y)
    #print log_x, log_y

    #compute regression
    (al, bl, lin_func, yrl, err) = linear_regression(log_x, log_y, compute_error)
    blx = numpy.exp2(bl) if use_binary_log else numpy.exp(bl) #convert coefficient b

    #compute projected y values, on x_plot if passed in, otherwise x
    yr = numpy.power(x_plot if x_plot else x, al)
    yr = numpy.multiply(blx, yr)

    func = lambda x: blx * (x ** al)

    return (al, blx, func, yr, err)

def exp_bin(data, bin_mult = 2):
    '''Divide passed in data into bins of exponentially increasing size
    first value returned is the average of the first 1 value
    next is the average of the first bin_mult values
    next is the average of the first bin_mult ** 2 values, and so on
    data is in fact a list of lists to be binned in parallel
    '''
    bin_size = 1
    index = 0
    binned_data = []

    while True:
        bin = data[index:bin_size]
        binned_data.append(float(sum(bin)) / len(bin)) #average bin and append to the result

        index += bin_size
        if index >= len(data): return binned_data
        bin_size *= bin_mult

def matrix_regression(input_vals, output_vals, funcs, categorical = False):
    '''Linear regression on a matrix of input properties and input values.
    Produces a linear model for factors influencing output values.
    Functions must be tuples (title, func).
    Input_vals and output_vals must be the same size
    '''

    assert(len(input_vals) == len(output_vals))

    #Add a constant function for the intercept
    if 'Intercept' not in map(lambda x: x[0], funcs): funcs.append(('Intercept', lambda x: 1))

    #Build up a matrix of each input function on each input val
    m = r.matrix(1.0, nrow=len(input_vals), ncol=len(funcs))
    
    for i in xrange(len(input_vals)):
        for j in xrange(len(funcs)):
            m[i][j] = funcs[j][1](i)
            if categorical and m[i][j]: m[i][j] = 1
            
    #Regression, done in R
    fit = r.lsfit(m,output_vals, intercept = False, )

    #Extract values
    coefficients = map(lambda x: x[1], sorted(fit['coefficients'].items(), key = lambda x: int(x[0][1:])))
    regression_func = lambda x,c=coefficients: sum([c[j] * funcs[j][1](x) for j in xrange(len(funcs))])
    model_vals =  map(regression_func, input_vals)
    least_squares_quality = least_squares_error(model_vals,output_vals)
    linear_quality = linear_error(model_vals,output_vals)

    return coefficients, regression_func, model_vals, least_squares_quality, linear_quality



def plot_data(lines, save_loc = ["plot_output.pdf",], **kwargs):
    '''Plots a series of data lines
    Each line is (x_list, y_list, properties_dict)
    '''
    fig = pyplot.figure()

    #Set up legend area
    if kwargs.get('legend_outside'): 
        ax = fig.add_axes([0.1, 0.1, 0.55, 0.8])
    else: 
        ax = fig.add_subplot(111)

    if kwargs.get('plot_title'): pyplot.title(kwargs.get('plot_title'))

    legend_text = []; legend_lines = []

    #Plot passed in points.
    for line in lines:

        title = line[2].pop('title', None)
        title_on_plot = line[2].pop('title_on_plot', False)

        plot_function = ax.loglog if kwargs.get('log_plot', None) else ax.plot
        plotted = plot_function(line[0],line[1],**(line[2] if len(line) > 2 else {}))

        if title and not title_on_plot:
            legend_lines.append(plotted)
            legend_text.append(title)

    for t in kwargs.get('text_labels', []):
        ax.text(**t)

    if kwargs.get('axis_v'): ax.axis(kwargs.get('axis_v'))
    elif kwargs.get('zero_yaxis'): 
        ax.set_ylim((0.0, ax.get_ylim()[1]))

    if kwargs.get('x_tick_freq'): ax.xaxis.set_major_locator(MultipleLocator(kwargs.get('x_tick_freq')))
    if kwargs.get('y_tick_freq'): ax.yaxis.set_major_locator(MultipleLocator(kwargs.get('y_tick_freq')))

    if kwargs.get('x_tick_func'): ax.xaxis.set_major_formatter(FuncFormatter(kwargs.get('x_tick_func')))
    if kwargs.get('y_tick_func'): ax.yaxis.set_major_formatter(FuncFormatter(kwargs.get('y_tick_func')))
       
    if legend_lines:
        if kwargs.get('legend_outside'): fig.legend(legend_lines, legend_text, loc = kwargs.get('legend_placement', 'best'), numpoints=1)
        else: ax.legend(legend_lines, legend_text, loc = kwargs.get('legend_placement', 'best'), numpoints=1)

    if kwargs.get('x_title'): ax.set_xlabel(kwargs.get('x_title'))
    if kwargs.get('y_title'): ax.set_ylabel(kwargs.get('y_title'))

    #Draw a second scale
    if kwargs.get('y2_transform') is not None:
        ax2 = pyplot.twinx(ax = ax)
        ax2.yaxis.set_view_interval(kwargs.get('y2_transform')(ax.yaxis.get_view_interval()[0]), kwargs.get('y2_transform')(ax.yaxis.get_view_interval()[1]))
        if kwargs.get('y2_title'): ax2.yaxis.set_label_text(kwargs.get('y2_title'))

    if save_loc is not None:
        for l in save_loc:
            pyplot.savefig(l, format=os.path.splitext(l)[1][1:])

    if kwargs.get('show_plot'): show()

def uniq(s): 
    '''uniqify a list, removes ordering'''
    return list(set(s))

def riemann_integrate(f, x1, x2, steps = 1000):
    if x1 > x2: return -riemann_integrate(f, x2, x1, steps)

    step = float(x2 - x1) / steps
    result = 0
    for i in range(1, steps):
        result += f(x1 + i * step)

    result += (f(x1) + f(x2)) / 2.0
    result *= step

    #result /= steps

    return result

@decorator
def _memoize(func, *args, **kw):
    if kw: # frozenset is used to ensure hashability
        key = hash(str(args) + str(frozenset(kw.iteritems())))
    else:
        key = args
    cache = func.cache # attributed added by memoize
    if key in cache:
        return cache[key]
    else:
        cache[key] = result = func(*args, **kw)
        return result

def memoized_function(f):
    f.cache = {}
    return _memoize(f)

def clear_memoized(o):
    o.__dict__.get('memoized', {}).clear()

def memoized_ident(name, *args, **kwargs):
    return hash(str(name) + str(args) + str(kwargs.items()))  

def memoized_method(method):
    '''
    Decorator, causes a method to return the previous version of a method
    if one is stored in the attribute cached. cached is created if it does not
    exist. use_cached is added as a keyword arg, it will override the caching 
    if set to false.
    '''

    def cm(self, *args, **kwargs):
        self.__dict__.setdefault('memoized', {})    #initialise cache if needed
        use_memoized = kwargs.pop('use_memoized', True)
        ident = memoized_ident(str(method.__name__), *args, **kwargs) 

        #call the underlying method and store its result if needed
        if not use_memoized or ident not in self.memoized:
            self.memoized[ident] = method(self, *args, **kwargs)

        return self.memoized[ident]

    return cm

