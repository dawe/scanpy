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
    # Prepare find_partition arguments as a dictionary,
    # appending to whatever the user provided. It needs to be this way
    # as this allows for the accounting of a None resolution
    # (in the case of a partition variant that doesn't take it on input)
    if use_weights:
        partition_kwargs['weights'] = np.array(g.ep['edge_weight'].a).astype(np.float64)
    partition_kwargs['n_iterations'] = n_iterations
    partition_kwargs['seed'] = random_state
    if resolution is not None:
        partition_kwargs['resolution_parameter'] = resolution
    # clustering proper
    part = leidenalg.find_partition(g, partition_type, **partition_kwargs)
    # store output into adata.obs
    groups = np.array(part.membership)
    if restrict_to is not None:
        if key_added == 'louvain':
            key_added += '_R'
        groups = rename_groups(
            adata,
            key_added,
            restrict_key,
            restrict_categories,
            restrict_indices,
            groups,
        )
    adata.obs[key_added] = pd.Categorical(
        values=groups.astype('U'),
        categories=natsorted(np.unique(groups).astype('U')),
    )
    # store information on the clustering parameters
    adata.uns['leiden'] = {}
    adata.uns['leiden']['params'] = dict(
        resolution=resolution,
        random_state=random_state,
        n_iterations=n_iterations,
    )
    logg.info(
        '    finished',
        time=start,
        deep=(
            f'found {len(np.unique(groups))} clusters and added\n'
            f'    {key_added!r}, the cluster labels (adata.obs, categorical)'
        ),
    )
    return adata if copy else None
