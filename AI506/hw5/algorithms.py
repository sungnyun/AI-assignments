from utils import Baskets
from typing import List, Tuple

### TODO: You may import any Python's standard library here (Do not import other external libraries) ###
from collections import defaultdict
### Import End ###

### TODO: You may declare any additional function here if needed ###

### Import End ###


def naive_with_matrix(baskets: Baskets, threshold: int) -> List[Tuple[int, int, int]]:
    # TODO: Implement the naive algorithm with matrix
    reorder_id, mat = [], []
    n = 0
    while True:
        items = baskets.readItems()
        if items is None:
            break
        for item in items:
            if item not in reorder_id:
                reorder_id.append(item)
                mat.append([0] * n)
                n += 1
        for idx, i in enumerate(items):
            for j in items[idx+1:]:
                id1, id2 = reorder_id.index(i), reorder_id.index(j)
                if id1 > id2:
                    id1, id2 = id2, id1
                mat[id2][id1] += 1
    
    ret = []
    for i in range(1, len(mat)):
        for j in range(len(mat[i])):
            if mat[i][j] >= threshold:
                id1, id2 = reorder_id[i], reorder_id[j]
                if id1 > id2:
                    id1, id2 = id2, id1
                ret.append((id1, id2, mat[i][j])) 
    ret.sort()

    return ret

def naive_with_triples(baskets: Baskets, threshold: int) -> List[Tuple[int, int, int]]:
    # TODO: Implement the naive algorithm with triples
    mat = defaultdict(lambda: 0)
    while True:
        items = baskets.readItems()
        if items is None:
            break
        for idx, i in enumerate(items):
            for j in items[idx+1:]:
                mat[(i,j)] += 1
    
    ret = []
    for pair in mat.keys():
        if mat[pair] >= threshold:
            ret.append((pair[0], pair[1], mat[pair]))
    ret.sort()

    return ret

def apriori_with_matrix(baskets: Baskets, threshold: int) -> List[Tuple[int, int, int]]:
    # TODO: Implement the apriori algorithm with matrix
    count_item = defaultdict(lambda: 0)
    valid_item = []
    while True:
        items = baskets.readItems()
        if items is None:
            break
        for item in items:
            count_item[item] += 1
    for item in count_item.keys():
        if count_item[item] >= threshold:
            valid_item.append(item)
    
    baskets.setBOF()
    reorder_id, mat = [], []
    n = 0
    while True:
        items = baskets.readItems()
        if items is None:
            break
        for item in items:
            if item not in reorder_id and item in valid_item:
                reorder_id.append(item)
                mat.append([0] * n)
                n += 1
        for idx, i in enumerate(items):
            for j in items[idx+1:]:
                if not (i in valid_item and j in valid_item):
                    continue
                id1, id2 = reorder_id.index(i), reorder_id.index(j)
                if id1 > id2:
                    id1, id2 = id2, id1
                mat[id2][id1] += 1
    
    ret = []
    for i in range(1, len(mat)):
        for j in range(len(mat[i])):
            if mat[i][j] >= threshold:
                id1, id2 = reorder_id[i], reorder_id[j]
                if id1 > id2:
                    id1, id2 = id2, id1
                ret.append((id1, id2, mat[i][j])) 
    ret.sort()

    return ret

def apriori_with_triples(baskets: Baskets, threshold: int) -> List[Tuple[int, int, int]]:
    # TODO: Implement the apriori algorithm with triples
    count_item = defaultdict(lambda: 0)
    valid_item = []
    while True:
        items = baskets.readItems()
        if items is None:
            break
        for item in items:
            count_item[item] += 1
    for item in count_item.keys():
        if count_item[item] >= threshold:
            valid_item.append(item)

    baskets.setBOF()
    mat = defaultdict(lambda: 0)
    while True:
        items = baskets.readItems()
        if items is None:
            break
        for idx, i in enumerate(items):
            for j in items[idx+1:]:
                if i in valid_item and j in valid_item:
                    mat[(i,j)] += 1 

    ret = []
    for pair in mat.keys():
        if mat[pair] >= threshold:
            ret.append((pair[0], pair[1], mat[pair]))
    ret.sort()

    return ret
