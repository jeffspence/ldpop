from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from builtins import zip
from builtins import map
from builtins import range
from builtins import object
from .compute_likelihoods import folded_likelihoods, NumericalError
from .moran_augmented import MoranStatesAugmented, MoranRates
from .moran_finite import MoranStatesFinite
from .compute_stationary import stationary

# from numba import njit
from multiprocessing import Pool
import logging
import time
import pandas
import numpy as np
from itertools import product
from io import StringIO


def getKey(num00, num01, num10, num11):
    key = {}
    key[(0, 0)] = num00
    key[(0, 1)] = num01
    key[(1, 0)] = num10
    key[(1, 1)] = num11
    # key[(0,-1)] = 0
    # key[(-1,0)] = 0
    # key[(1,-1)] = 0
    # key[(-1,1)] = 0
    # return frozenset(key.items())
    return tuple(sorted(key.items()))


# columns has the columns of the table
# plus possibly extraneous things, but we just pull out what we want
def getRow(num00, num01, num10, num11, columns, rho_grid):
    toReturn = []
    for rhos in rho_grid:
        key = getKey(num00, num01, num10, num11)
        toReturn.append(columns[tuple(rhos)][key])
    return toReturn


def get_states(n, exact):
    if exact:
        return MoranStatesAugmented(n)
    else:
        return MoranStatesFinite(n)


def getColumnHelper(args):
    return getColumn(*args)


def getColumn(moranRates, rho_list, theta, popSizes, timeLens, init):
    try:
        return folded_likelihoods(moranRates,
                                  rho_list,
                                  theta,
                                  popSizes,
                                  timeLens,
                                  lastEpochInit=init)
    except NumericalError as err:
        print(rho_list)
        print(err)
        assert False, (rho_list, err)


def computeLikelihoods(n, exact, popSizes, theta, timeLens, rhoGrid, cores,
                       store_stationary=None, load_stationary=None):
    ancient_rhos = [rhos[-1] for rhos in rhoGrid]
    assert ancient_rhos == sorted(ancient_rhos)

    # make the pool first to avoid copying large objects.
    # maxtasksperchild=1 to avoid memory issues
    executor = Pool(cores, maxtasksperchild=1)

    # make the states and the rates
    states = get_states(n, exact)
    moranRates = MoranRates(states)

    # compute initial distributions and likelihoods
    prevInit = states.getUnlinkedStationary(popSize=popSizes[-1], theta=theta)
    inits = []

    if load_stationary:
        stationary_dists = np.load(load_stationary)
        for stationary_dist in stationary_dists:
            inits.append(stationary_dist)
    else:
        for rhos in reversed(rhoGrid):
            rates = moranRates.getRates(rho=rhos[-1],
                                        popSize=popSizes[-1],
                                        theta=theta)
            prevInit = stationary(Q=rates,
                                  init=prevInit,
                                  norm_order=float('inf'),
                                  epsilon=1e-2)
            inits.append(prevInit)
    ret = executor.map(getColumnHelper,
                       [(moranRates, rhos, theta, popSizes, timeLens, prevInit)
                        for rhos, prevInit in zip(reversed(rhoGrid), inits)])
    logging.info('Cleaning up results...')
    if store_stationary:
        full_inits = np.array([result[1] for result in ret])
        np.save(store_stationary, full_inits)
    ret = [states.ordered_log_likelihoods(result[0]) for result in ret]
    executor.close()

    return [(tuple(rhos), lik) for rhos, lik in zip(rhoGrid, reversed(ret))]


class LookupTable(object):
    '''
    Lookup table of 2-locus likelihoods. Construct with
        lookup_table = LookupTable(n, theta, rhos, [pop_sizes],
                                   [times], [exact], [processes])
    (optional arguments in square bracket [])

    Printing
    --------
    str(lookup_table) returns a string in the format expected by ldhat
    (or ldhelmet, if the rhos are not a uniformly spaced grid starting at 0)
    to print the table to STDOUT in this format, do
        print lookup_table

    Attributes
    ----------
    lookup_table.table = pandas.DataFrame,
        with rows corresponding to configs and columns corresponding to rho. So
             lookup_table.table.ix['13 0 0 1', 1.0]
        returns the likelihood of sample config '13 0 0 1' at rho=1.0.
    lookup_table.n = number of haplotypes
    lookup_table.theta = 2*mutation rate
    lookup_table.column = [rho0,rho1,...] = the grid of rhos (2*recomb rate)
    lookup_table.index = [config0,config1,...] = the sample configs
    lookup_table.pop_sizes = [size0,size1,...,sizeD]
                           = coalescent scaled population sizes
         size0 = present size, sizeD = ancient size
    lookup_table.times = [t1,...,tD] = the times of size changes,
                                       going backwards from present
         must be increasing positive reals
    lookup_table.exact = boolean
         if False, use finite moran model, a reasonable approximation
         that is much faster than the exact formula. Accuracy of the
         approximation can be improved by taking n larger than needed,
         and subsampling (e.g. using ldhat/lkgen.c). As n->infty,
         the finite moran model -> 'exact' diffusion.

    Parallelization: use
        LookupTable(...,processes)
    to specify the number of parallel processes to use.
    '''
    def __init__(self, n, N, theta, rhos, pop_sizes=[1], times=[],
                 rho_times=[], exact=True,
                 processes=1, store_stationary=None, load_stationary=None):
        assert (list(rhos) == list(sorted(rhos))
                and len(rhos) == len(set(rhos))), 'rhos must be sorted, unique'

        assert len(pop_sizes) == len(times)+1

        if exact:
            rhos[0] = rhos[0] + 1e-10

        timeLens, pop_sizes, rho_counts = refineTimes(pop_sizes,
                                                      times,
                                                      rho_times)

        assert sum(rho_counts) == len(pop_sizes)
        assert len(pop_sizes) == len(timeLens) + 1

        rho_grid = makeRhoGrid(rhos, rho_counts)

        start = time.time()

        if N is None:
            N = n

        results = computeLikelihoods(N, exact, pop_sizes, theta, timeLens,
                                     rho_grid, processes, store_stationary,
                                     load_stationary)

        # halfn = int(N) // 2
        columns = dict(results)
        index, rows = [], []
        # make all these configs then print them out
        for n00 in range(0, N+1):
            for n01 in range(0, N+1-n00):
                for n10 in range(0, N+1-n00-n01):
                    n11 = N-n00-n01-n10
                    index.append('%d_%d_%d_%d' % (n00,
                                                  n01,
                                                  n10,
                                                  n11))
                    rows.append(getRow(n00, n01, n10, n11, columns, rho_grid))

        '''
        for i in range(1, halfn + 1):
            for j in range(1, i + 1):
                for k in range(j, -1, -1):
                    hapMult11 = k
                    hapMult10 = j - k
                    hapMult01 = i - k
                    hapMult00 = N - i - j + k

                    index += ['%d %d %d %d' % (hapMult00,
                                               hapMult01,
                                               hapMult10,
                                               hapMult11)]
                    rows += [getRow(hapMult00,
                                    hapMult01,
                                    hapMult10,
                                    hapMult11,
                                    columns,
                                    rho_grid)]
        '''
        column_names = [','.join(map(str, rhos)) for rhos in rho_grid]
        # self.table = pandas.DataFrame(rows,
        #                               index=index,
        #                                columns=column_names)
        self.table = downsample(pandas.DataFrame(rows, index=index,
                                                 columns=column_names),
                                n)

        end = time.time()
        logging.info('Computed lookup table in %f seconds ' % (end-start))

        self.n = n
        self.theta = theta
        self.pop_sizes = list(pop_sizes)
        self.times = list(times)
        self.exact = exact

    def __str__(self):
        output = StringIO()
        self.table.to_csv(output)
        output.seek(0)
        return output.read().strip()
        '''
        ret = []
        ret += [[self.n, self.table.shape[0]]]
        ret += [[1, self.theta]]

        ret += [rhos_to_string(self.table.columns).split()]

        ret += [[], []]
        for i, (config, row) in enumerate(self.table.iterrows(), start=1):
            ret += [[i, '#', config, ':'] + list(row)]

        return '\n'.join([' '.join(map(str, x)) for x in ret])
        '''


def refineTimes(pop_sizes, times, rho_times):
    assert times == sorted(times)
    assert rho_times == sorted(rho_times)

    fine_times = sorted(list(set(times)) + list(set(rho_times)))
    epoch_lengths = np.diff([0] + fine_times).tolist()

    # how many epochs belong to each rho epoch?
    rho_counts = []
    for rt in rho_times + [float('inf')]:
        rho_counts.append(np.sum(np.array([0] + fine_times) < rt))
    rho_counts = np.diff([0] + rho_counts).astype(int).tolist()

    fine_sizes = []
    i = 0
    for ft in fine_times:
        fine_sizes.append(pop_sizes[i])
        if ft in times:
            i += 1
    assert i == len(pop_sizes) - 1
    fine_sizes.append(pop_sizes[-1])

    return epoch_lengths, fine_sizes, rho_counts


def makeRhoGrid(rhos, rho_counts):
    rho_grid = []
    for raw_rhos in product(rhos, repeat=len(rho_counts)):
        this_list = []
        for count, rr in zip(rho_counts, reversed(raw_rhos)):
            this_list.extend([rr] * count)
        rho_grid.append(this_list)
    return rho_grid


def rhos_to_string(rhos):
    return ' '.join(str(r) for r in rhos)
    rhos = np.array(rhos)
    if rhos[0] == 0 and np.allclose(rhos[1:] - rhos[:-1],
                                    rhos[-1] / float(len(rhos)-1),
                                    atol=0):
        rho_line = [len(rhos), rhos[-1]]
    else:
        rho_line = []
        prev_rho, prev_diff = rhos[0], float('inf')
        for rho in rhos[1:]:
            if not np.isclose(rho - prev_rho, prev_diff, atol=0):
                prev_diff = rho - prev_rho
                rho_line += [prev_rho, prev_diff]
            prev_rho = rho
        rho_line += [prev_rho]
    return ' '.join(map(str, rho_line))


def rhos_from_string(rho_string):
    '''
    Return a list of rhos obtained from a comma
    separated string of rhos in one of two formats:
    if the rho_string has two elements, <num_rho>,<max_rho>
    return a list of size num_rho [0, ...., max_rho]
    otherwise, the rho_string should be <rho_1>,<step_size_1>,<rho_2>...
    and return [rho_1, rho_1 + step_size_1, ..., rho_2,...]
    note that this implies that if rho_string is just <rho>, return [rho].
    '''

    rho_args = rho_string.split(',')

    if len(rho_args) == 2:
        n, R = int(rho_args[0]), float(rho_args[1])
        return list(np.arange(n, dtype=float) / float(n-1) * R)

    rhos = [float(rho_args[0])]
    arg_idx = 1
    while(arg_idx < len(rho_args)):
        assert arg_idx + 1 < len(rho_args)
        step_size = float(rho_args[arg_idx])
        endingRho = float(rho_args[arg_idx+1])
        arg_idx += 2
        cur_rho = rhos[-1]

        # 1e-13 deals with the numeric issues
        while(cur_rho < endingRho-1e-13):
            cur_rho += step_size
            rhos.append(cur_rho)
        if abs(cur_rho - endingRho) > 1e-13:
            print(cur_rho)
            print(endingRho)
            raise IOError('the Rhos you input are not so nice'
                          '(stepsize should divide difference in rhos)')
    return rhos


'''
def downsample(table, desired_size):
    """
    Computes a lookup table for a smaller sample size.

    Takes table and marginalizes over the last individuals to compute
    a lookup table for a sample size one smaller and repeats until reaching
    the desired_size.

    Args:
        table: A pandas.DataFrame containing a lookup table as computed by
            make_table.
        desired_size: The desired smaller sample size.

    Returns:
        A pandas.DataFrame containing a lookup table with sample size
        desired_size. The DataFrame is essentially the same as if make_table
        had been called with a smaller sample size.
    """
    first_config = table.index.values[0].split()
    curr_size = sum(map(int, first_config))
    rhos = table.columns
    curr_table = table.values
    while curr_size > desired_size:
        logging.info('Downsampling...  Currently at n = %d', curr_size)
        curr_table = _single_vec_downsample(curr_table, curr_size)
        curr_size = curr_size - 1
    halfn = curr_size // 2
    index = []
    idx = 0
    for i in range(1, halfn + 1):
        for j in range(1, i + 1):
            for k in range(j, -1, -1):
                n11 = k
                n10 = j - k
                n01 = i - k
                n00 = curr_size - i - j + k
                index.append('{}_{}_{}_{}'.format(n00, n01, n10, n11))
                idx += 1
    table = pandas.DataFrame(curr_table, index=index, columns=rhos)
    return table
'''


# this is the new one including all of the non-polymorphic sites
def downsample(table, desired_size):
    curr_table = table
    first_config = table.index.values[0].split('_')
    curr_size = sum(map(int, first_config))
    while curr_size > desired_size:
        logging.info('Downsampling...  Currently at n = %d', curr_size)
        curr_table = _single_vec_downsample(curr_table, curr_size)
        curr_size -= 1
    return curr_table


def get_table_idx(n00, n01, n10, n11, sample_size):
    return '{}_{}_{}_{}'.format(n00, n01, n10, n11)


def _single_vec_downsample(old_vec, sample_size):
    new_conf_num = (sample_size
                    * (sample_size+1)
                    * (sample_size+2)) // 6
    to_return = np.zeros((new_conf_num, old_vec.shape[1]))
    index = []
    idx = 0
    for n00 in range(0, sample_size):
        for n01 in range(0, sample_size-n00):
            for n10 in range(0, sample_size-n00-n01):
                n11 = sample_size-1-n00-n01-n10
                add00 = get_table_idx(n00+1, n01, n10, n11, sample_size)
                add01 = get_table_idx(n00, n01+1, n10, n11, sample_size)
                add10 = get_table_idx(n00, n01, n10+1, n11, sample_size)
                add11 = get_table_idx(n00, n01, n10, n11+1, sample_size)
                to_return[idx, :] = np.logaddexp(old_vec.loc[add00].to_numpy(),
                                                 old_vec.loc[add01].to_numpy())
                to_return[idx, :] = np.logaddexp(to_return[idx, :],
                                                 old_vec.loc[add10].to_numpy())
                to_return[idx, :] = np.logaddexp(to_return[idx, :],
                                                 old_vec.loc[add11].to_numpy())
                idx += 1
                index.append('{}_{}_{}_{}'.format(n00, n01, n10, n11))
    to_return = pandas.DataFrame(to_return,
                                 index=index,
                                 columns=old_vec.columns)
    return to_return


'''
@njit('int64(int64, int64, int64, int64, int64)', cache=True)
def get_table_idx(n00, n01, n10, n11, sample_size):
    """
    Returns the lookup table index for a two-locus configuration.

    Args:
        n00: Number of 00 haplotypes.
        n01: Number of 01 haplotypes.
        n10: Number of 10 haplotypes.
        n11: Number of 11 haplotypes.
        sample_size: The number of haplotypes for which the lookup table was
            computed.

    Returns:
       The index of the lookup table corresponding to the configuration. i.e.,
       lookup_table.values[get_table_idx(n00, n01, n10, n11, sample_size), :]
       will be the log-likelihood of the configuration.

    Raises:
        ValueError: Cannot obtain a table index for a negative count.
        ValueError: Cannot obtain a table index for the wrong n.
        ValueError: Cannot obtain a table index for a non-segregating allele.
    """
    if n00 < 0 or n01 < 0 or n10 < 0 or n11 < 0:
        raise ValueError('Cannot obtain a table index for a negative count.')
    if sample_size != n00 + n01 + n10 + n11:
        raise ValueError('Cannot obtain a table index for the wrong n.')
    if (
            n00 + n01 == sample_size
            or n10 + n11 == sample_size
            or n01 + n11 == sample_size
            or n10 + n00 == sample_size
    ):
        raise ValueError('Cannot obtain a table index for a non-segregating '
                         'allele.')
    if n00 < n11:
        n00, n11 = n11, n00
    if n01 < n10:
        n01, n10 = n10, n01
    if n11 + n01 > sample_size//2:
        n00, n01, n10, n11 = n01, n00, n11, n10
    i, j, k = n01+n11, n10+n11, n11
    return (j-k) + ((j-1) * j)//2 + (j-1) + round(((i-1)**3)/6 + (i-1)**2 +
                                                  5*(i-1)/6)


@njit('float64[:, :](float64[:, :], int64)', cache=True)
def _single_vec_downsample(old_vec, sample_size):
    halfn = (sample_size - 1) // 2
    new_conf_num = (1 + halfn + halfn*(halfn - 1)*(halfn + 4)//6
                    + (halfn - 1)*(halfn + 2)//2)
    to_return = np.zeros((new_conf_num, old_vec.shape[1]))
    idx = 0
    for i in range(1, halfn+1):
        for j in range(1, i+1):
            for k in range(j, -1, -1):
                n11 = k
                n10 = j - k
                n01 = i - k
                n00 = sample_size - 1 - i - j + k
                add00 = get_table_idx(n00+1, n01, n10, n11, sample_size)
                add01 = get_table_idx(n00, n01+1, n10, n11, sample_size)
                add10 = get_table_idx(n00, n01, n10+1, n11, sample_size)
                add11 = get_table_idx(n00, n01, n10, n11+1, sample_size)
                to_return[idx, :] = np.logaddexp(old_vec[add00, :],
                                                 old_vec[add01, :])
                to_return[idx, :] = np.logaddexp(to_return[idx, :],
                                                 old_vec[add10, :])
                to_return[idx, :] = np.logaddexp(to_return[idx, :],
                                                 old_vec[add11, :])
                idx += 1
    return to_return
'''
