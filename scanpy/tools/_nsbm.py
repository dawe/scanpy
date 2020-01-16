from typing import Optional, Tuple, Sequence, Type, Union

import numpy as np
import pandas as pd
from natsort import natsorted
from anndata import AnnData
from numpy.random.mtrand import RandomState
from scipy import sparse

from .. import _utils
from .. import logging as logg

from ._utils_clustering import rename_groups, restrict_adjacency

def nsbm(
    adata: AnnData,
    sweep_iterations: int = 10000,
    max_iterations: int = 1000000,
    epsilon: float = 1e-3,
    equilibrate: bool = True,
    wait: int = 1000,
    nbreaks: int = 2,
    collect_marginals: bool = True,
    hierarchy_length: int = 10,
    *,
    restrict_to: Optional[Tuple[str, Sequence[str]]] = None,
    random_state: Optional[Union[int, RandomState]] = 0,
    key_added: str = 'nsbm',
    adjacency: Optional[sparse.spmatrix] = None,
    directed: bool = False,
    use_weights: bool = True,
    save_state: bool = True,
    copy: bool = False,
    **partition_kwargs,
) -> Optional[AnnData]:
    """\
    Cluster cells into subgroups [Peixoto14]_.

    Cluster cells using the nested Stochastic Block Model [Peixoto14]_,
    a hierarchical version of Stochastic Block Model [Holland83]_, performing
    Bayesian inference on node groups. NSBM should circumvent classical
    limitations of SBM in detecting small groups in large graphs
    replacing the noninformative priors used by a hierarchy of priors
    and hyperpriors.

    This requires having ran :func:`~scanpy.pp.neighbors` or
    :func:`~scanpy.external.pp.bbknn` first.

    Parameters
    ----------
    adata
        The annotated data matrix.
    sweep_iterations
        Number of iterations to run mcmc_sweep.
        Higher values lead longer runtime.
    max_iterations
        Maximal number of iterations to be performed by the equilibrate step.
    epsilon
        Relative changes in entropy smaller than epsilon will
        not be considered as record-breaking.
    equilibrate
        Whether or not perform the mcmc_equilibrate step.
        Equilibration should always be performed. Note, also, that without
        equilibration it won't be possible to collect marginals.
    collect_marginals
        whether or not collect node probability of belonging
        to a specific partition.
    wait
        Number of iterations to wait for a record-breaking event.
        Higher values result in longer computations. Set it to small values
        when performing quick tests.
    nbreaks
        Number of iteration intervals (of size `wait`) without
        record-breaking events necessary to stop the algorithm.
    hierarchy_length
        Initial length of the hierarchy. When large values are
        passed, the top-most levels will be uninformative as they
        will likely contain the very same groups. Increase this valus
        if a very large number of cells is analyzed (>100.000).
    key_added
        `adata.obs` key under which to add the cluster labels.
    adjacency
        Sparse adjacency matrix of the graph, defaults to
        `adata.uns['neighbors']['connectivities']`.
    directed
        Whether to treat the graph as directed or undirected.
    use_weights
        If `True`, edge weights from the graph are used in the computation
        (placing more emphasis on stronger edges).
    save_state
        Whether to keep the block model state saved for subsequent
        custom analysis with graph-tool.
    copy
        Whether to copy `adata` or modify it inplace.

    Returns
    -------
    `adata.obs[key_added]`
        Array of dim (number of samples) that stores the subgroup id
        (`'0'`, `'1'`, ...) for each cell. Multiple arrays will be
        added when `return_level` is set to `all`
    `adata.uns['nsbm']['params']`
        A dict with the values for the parameters `resolution`, `random_state`,
        and `n_iterations`.
    `adata.uns['nsbm']['stats']`
        A dict with the values returned by mcmc_sweep
    `adata.uns['nsbm']['node_marginals']`
        A `np.ndarray` with cell probability of belonging to a specific group
    `adata.uns['nsbm']['state']`
        The NestedBlockModel state object
    """
    try:
        import graph_tool.all as gt
    except ImportError:
        raise ImportError(
            """Please install the graph-tool library either visiting

            https://git.skewed.de/count0/graph-tool/-/wikis/installation-instructions

            or by conda: `conda install -c conda-forge graph-tool`
            """
        )
    partition_kwargs = dict(partition_kwargs)

    if collect_marginals and not equilibrate:
        raise ValueError(
            "You can't collect marginals without MCMC equilibrate "
            "step. Either set `equlibrate` to `True` or "
            "`collect_marginals` to `False`"
        )

    start = logg.info('running nested Stochastic Block Model')
    adata = adata.copy() if copy else adata
    # are we clustering a user-provided graph or the default AnnData one?
    if adjacency is None:
        if 'neighbors' not in adata.uns:
            raise ValueError(
                'You need to run `pp.neighbors` first '
                'to compute a neighborhood graph.'
            )
        adjacency = adata.uns['neighbors']['connectivities']
    if restrict_to is not None:
        restrict_key, restrict_categories = restrict_to
        adjacency, restrict_indices = restrict_adjacency(
            adata,
            restrict_key,
            restrict_categories,
            adjacency,
        )
    # convert it to igraph
    g = _utils.get_graph_tool_from_adjacency(adjacency, directed=directed)

    if use_weights:
#        tr_weight = g.ep.weight.copy()
#        tr_weight = np.log(tr_weight)
        state = gt.minimize_nested_blockmodel_dl(g, state_args=dict(recs=[g.ep.weight],
                                                                    rec_types=['real-normal']))
    else:
        state = gt.minimize_nested_blockmodel_dl(g)

    bs = state.get_bs()
    if len(bs) < hierarchy_length:
        # increase hierarchy length up to the specified value
        bs += [np.zeros(1)] * (hierarchy_length - len(bs))

    if use_weights:
        state = gt.NestedBlockState(g, bs, state_args=dict(recs=[g.ep.weight],
                                            rec_types=["real-normal"]), sampling=True)
    else:
        state = state.copy(bs=bs, sampling=True)

    # run the MCMC sweep step
    logg.info('running MCMC sweep step')
    s_dS, s_nattempts, s_nmoves = state.mcmc_sweep(niter=sweep_iterations)

    # equilibrate the Markov chain
    if equilibrate:
        logg.info('equlibrating the Markov chain')
        e_dS, e_nattempts, e_nmoves = gt.mcmc_equilibrate(state, wait=wait, nbreaks=nbreaks, epsilon=epsilon,
            max_niter=max_iterations, mcmc_args=dict(niter=10)
        )

    # run 1000 iterations of equilibrate just to collect marginals
    if collect_marginals:
        logg.info('collecting marginals')
        vertex_counts = [None] * len(state.get_levels())
        def collect_marginals(s):
            global vertex_counts
            vertex_counts = [sl.collect_vertex_marginals(vertex_counts[l]) for l, sl in enumerate(s.get_levels())]

        gt.mcmc_equilibrate(state, force_niter=101, mcmc_args=dict(niter=10), wait=wait,
            nbreaks=nbreaks, epsilon=epsilon, callback=collect_marginals
        )

    # everything is in place, we need to fill all slots
    # first build an array with
    groups = np.zeros((g.num_vertices(), len(bs)), dtype=int)
    for x in range(len(bs)):
        # for each level, project labels to the vertex level
        # so that every cell has a name. Note that at this level
        # the labels are not necessarily consecutive
        groups[:, x] = [n for n in state.project_partition(x, 0)]

    groups = pd.DataFrame(groups).astype('category')

    # rename categories from 0 to n

    for c in groups.columns:
        new_cat_names = [u'%s' % x for x in range(len(groups.loc[:, c].cat.categories))]
        groups.loc[:, c].cat.rename_categories(new_cat_names, inplace=True)

    if restrict_to is not None:
        groups.index = adata.obs[restrict_key].index
    else:
        groups.index = adata.obs_names

    # add column names
    groups.columns = ["%s_level_%d" % (key_added, level) for level in range(len(bs))]

    # remove any column with the same key
    keep_columns = [x for x in adata.obs.columns if not x.startswith('%s_level_' % key_added)]
    adata.obs = adata.obs.loc[:, keep_columns]
    # concatenate obs with new data
    adata.obs = pd.concat([adata.obs, groups], axis=1)

    # add some unstructured info

    adata.uns['nsbm'] = {}
    adata.uns['nsbm']['stats'] = dict(
        sweep_dS=s_dS,
        sweep_nattempts=s_nattempts,
        sweep_nmoves=s_nmoves,
        equlibrate_dS=e_dS,
        equlibrate_nattempts=e_nattempts,
        equlibrate_nmoves=e_nmoves,
        level_entropy=np.array([state.level_entropy(x) for x in range(len(state.levels))] )
    )
    if save_state:
        adata.uns['nsbm']['state'] = state

#    adata.uns['nsbm']['params'] = dict(
#        resolution=resolution,
#        random_state=random_state,
#        n_iterations=n_iterations,
#    )




    # now add marginal probabilities. The problem here is that
    # we have as many matrices as levels

    if collect_marginals:
        adata.uns['nsbm']['node_marginals'] = {}

        # get counts for the lowest levels, cells by groups. This will be summed in the
        # parent levels, according to groupings
        c0 = vertex_counts[0].get_2d_array(range(state.get_levels()[0].s.get_nonempty_B())).T
        adata.uns['nsbm']['node_marginals']['level_0'] = c0

        l0 = "%s_level_0" % key_added
        for level in groups.columns[1:]:
            cross_tab = pd.crosstab(groups.loc[:, l0], groups.loc[:, level])
            key_name = level.replace('%s_' % key_added, '')
            cl = np.zeros((c0.shape[0], cross_tab.shape[1]), dtype=c0.dtype)
            for x in range(c0.shape[1]):
                # sum counts of level_0 groups corresponding to
                # this group at current level
                cl[:, x] = c0[:, np.where(cross_tab.iloc[:, x] > 0)[0]].sum(axis=1)
            adata.uns['nsbm']['node_marginals'][key_name] = cl




    logg.info(
        '    finished',
        time=start,
        deep=(
            f'found {len(np.unique(groups))} clusters and added\n'
            f'    {key_added!r}, the cluster labels (adata.obs, categorical)'
        ),
    )
    return adata if copy else None
