import collections
import multiprocessing

from rdkit.Chem import AllChem


def _init_worker(data, distThresh, distFunc):
    global _data, _distThresh, _distFunc, _vectorize
    _data = data
    _distThresh = distThresh
    _distFunc = distFunc


def _calc(row):
    nbrLists = collections.defaultdict(list)
    distrow = _distFunc(_data[row], _data[:row])
    for j in range(row):
        if distrow[j] <= _distThresh:
            nbrLists[row].append(j)
            nbrLists[j].append(row)
    return nbrLists


def _getnbrLists(data, nPts, distThresh, distFunc):
    nbrLists = [[] for _ in range(nPts)]

    with multiprocessing.Pool(processes=None, initializer=_init_worker, initargs=[data, distThresh, distFunc]) as pool:
        rowList = pool.map(_calc, [row for row in range(nPts)])

    #now we simple merge the lists
    for row in rowList:
        for i, j in row.items():
            nbrLists[i].extend(j)
    return nbrLists


def BulkTanimotoSimilarity(fp, lfps):
    return AllChem.DataStructs.BulkTanimotoSimilarity(fp, lfps, returnDistance=1)


def ParallelClusterData(data, nPts, distThresh, distFunc=BulkTanimotoSimilarity,
                        reordering=False):
    """  clusters the data points passed in and returns the list of clusters

     **Arguments**

       - data: a list of items with the input data

       - nPts: the number of points to be used

       - distThresh: elements within this range of each other are considered
         to be neighbors

       - distFunc: a function to calculate distances between a point and a list of points.
            Receives 1 point and a list as arguments, should return a list of floats

       - reodering: if this toggle is set, the number of neighbors is updated
            for the unassigned molecules after a new cluster is created such
            that always the molecule with the largest number of unassigned
            neighbors is selected as the next cluster center.

     **Returns**

       - a tuple of tuples containing information about the clusters:
          ( (cluster1_elem1, cluster1_elem2, ...),
            (cluster2_elem1, cluster2_elem2, ...),
            ...
          )
          The first element for each cluster is its centroid.

    """

    nbrLists = _getnbrLists(data, nPts, distThresh, distFunc)

    # sort by the number of neighbors:
    tLists = [(len(y), x) for x, y in enumerate(nbrLists)]
    tLists.sort(reverse=True)

    res = []
    seen = [0] * nPts
    while tLists:
        _, idx = tLists.pop(0)
        if seen[idx]:
            continue
        tRes = [idx]
        for nbr in nbrLists[idx]:
            if not seen[nbr]:
                 tRes.append(nbr)
                 seen[nbr] = 1
        # update the number of neighbors:
        # remove all members of the new cluster from the list of
        # neighbors and reorder the tLists
        if reordering:
            # get the list of affected molecules, i.e. all molecules
            # which have at least one of the members of the new cluster
            # as a neighbor
            nbrNbr = [nbrLists[t] for t in tRes]
            nbrNbr = frozenset().union(*nbrNbr)
            # loop over all remaining molecules in tLists but only
            # consider unassigned and affected compounds
            for x, y in enumerate(tLists):
                y1 = y[1]
                if seen[y1] or (y1 not in nbrNbr):
                    continue
                # update the number of neighbors
                nbrLists[y1] = set(nbrLists[y1]).difference(tRes)
                tLists[x] = (len(nbrLists[y1]), y1)
            # now reorder the list
            tLists.sort(reverse=True)
        res.append(tuple(tRes))
    return tuple(res)