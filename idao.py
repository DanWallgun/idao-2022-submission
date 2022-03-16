import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from catboost import CatBoostRegressor
from pymatgen.core import Structure


def read_pymatgen_dict(file):
    with open(file, "r") as f:
        d = json.load(f)
    return Structure.from_dict(d)

# public data
def load_public_data():
    dataset_path = Path('data/dichalcogenides_public')
    targets = pd.read_csv(dataset_path / "targets.csv", index_col=0)
    struct = {item.name.strip(".json"): read_pymatgen_dict(item) for item in (dataset_path / "structures").iterdir()}
    data = pd.DataFrame(columns=["structures"], index=struct.keys())
    data = data.assign(structures=struct.values(), targets=targets)
    return data

public_data = load_public_data()
train, test = train_test_split(public_data, test_size=0.25, random_state=666)

# private data
def load_private_data():
    dataset_path = Path('data/dichalcogenides_private')
    struct = {item.name.strip('.json'): read_pymatgen_dict(item) for item in (dataset_path / 'structures').iterdir()}
    private_test = pd.DataFrame(columns=['id', 'structures'], index=struct.keys())
    private_test = private_test.assign(structures=struct.values())
    return private_test

private_test = load_private_data()

def prepare_box_coord(crystal):
    # Crystal must contain exactly 192 balls
    assert len(crystal) == 8 * 8 * 3
    lst = []
    for ind, site in enumerate(crystal):
        site_x, site_y, site_z = map(int, site.coords)
        lst.append((site_z, site_y, site_x, site.coords))
    to_box_coord = dict()
    real_coord = dict()
    lst = sorted(lst)
    for z in range(3):
        for y in range(8):
            for x in range(8):
                coor = lst[z * 64 + y * 8 + x]
                to_box_coord[(coor[2], coor[1], coor[0])] = (x, y, z)
                real_coord[(x, y, z)] = coor[3]
    return to_box_coord, real_coord

def get_box(crystal, to_box_coord):
    box = [[['E' for k in range(3)] for j in range(8)] for i in range(8)]
    for ind, site in enumerate(crystal):
        site_x, site_y, site_z = map(int, site.coords)
        box_x, box_y, box_z = to_box_coord[site_x, site_y, site_z]
        box[box_x][box_y][box_z] = site.species.formula
    return box

def get_interesting_points(box):
    usual = ['S1', 'Mo1', 'S1']
    points = []
    for z in range(3):
        for y in range(8):
            for x in range(8):
                if box[x][y][z] != usual[z]:
                    points.append((x, y, z, box[x][y][z]))
    return points

struct_with_192 = None
for idx, struct in enumerate(public_data['structures']):
    if len(struct.sites) == 192:
        struct_with_192 = struct
        break
assert struct_with_192 is not None
to_box_coord, real_coord = prepare_box_coord(struct_with_192)


class CrystalTransformerRelativeWithShift(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X):
        return self
    
    def transform(self, X):
        X_out = np.empty((len(X), 18), dtype=object)
        for it, i in enumerate(X):
            box = get_box(i, to_box_coord)
            interesting_points = get_interesting_points(box)
            lens = []
            while len(interesting_points) < 3:
                interesting_points += [interesting_points[0]]
            def get_perimeter(points):
                perim = 0
                for i in range(len(points)):
                    cur_point = points[i]
                    nxt_point = points[(i + 1) % len(points)]
                    perim += np.linalg.norm(real_coord[tuple(nxt_point[:3])] - real_coord[tuple(cur_point[:3])])
                return perim
            
            shifts = []
            for point_i in interesting_points + [(8, 8, 0, 'E')]:
                for point_j in interesting_points + [(8, 8, 0, 'E')]:
#                     print(point_j)
                    shift_x = 8 - point_i[0]
                    shift_y = 8 - point_j[1]
                    new_points = []
                    for cur_point in interesting_points:
                        new_points.append([*cur_point])
                    for ind in range(len(new_points)):
                        new_points[ind][0] += shift_x
                        new_points[ind][1] += shift_y
                        new_points[ind][0] %= 8
                        new_points[ind][1] %= 8
#                     print(new_points)
                    shifts.append([get_perimeter(new_points), shift_x, shift_y])
            shifts = sorted(shifts)
            final_shift = shifts[0][1:]
            for ind in range(len(interesting_points)):
                tpl = interesting_points[ind]
                tpl = list(tpl)
                tpl[0] += final_shift[0]
                tpl[1] += final_shift[1]
                tpl[0] %= 8
                tpl[1] %= 8
                interesting_points[ind] = tuple(tpl)
                

            for i in range(len(interesting_points)):
                cur_point = interesting_points[i]
                next_point = interesting_points[(i + 1) % len(interesting_points)]
                vec_len = np.linalg.norm(real_coord[next_point[:3]] - real_coord[cur_point[:3]])
                lens.append((vec_len, cur_point[3], i, (i + 1) % len(interesting_points)))
            lens = sorted(lens)
            cnt = dict()
            def add_to_cnt(x):
                cnt[x] = cnt.get(x, 0) + 1
            add_to_cnt(lens[0][2])
            add_to_cnt(lens[0][3])
            add_to_cnt(lens[1][2])
            add_to_cnt(lens[1][3])
            start = -1
            for k, v in cnt.items():
                if v == 2:
                    start = k
            assert lens[1][2] == start or lens[1][3] == start
            nxt = lens[1][2] + lens[1][3] - start
            last = 3 - start - nxt
            order = [start, nxt, last]
            vecs = []
            vec_lens = []
            angles = []
            for i in range(len(order)):
                cur_ind, nxt_ind = order[i], order[(i + 1) % len(order)]
                cur_vec = np.array(real_coord[interesting_points[nxt_ind][:3]])
                nxt_vec = np.array(real_coord[interesting_points[cur_ind][:3]])
                vecs.append(nxt_vec - cur_vec)

            def angle(v1, v2):
                if np.linalg.norm(v1) < 1e-6 or np.linalg.norm(v2) < 1e-6:
                    return 0
                res = np.arccos(np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1.0, 1.0))
#                 print(v1, v2, res)
                assert 0 <= res <= np.pi
                return res
            
            for i in range(len(vecs)):
                cur_vec = vecs[i]
                nxt_vec = vecs[(i + 1) % len(vecs)]
                angles.append(angle(cur_vec, nxt_vec))
                vec_lens.append(np.linalg.norm(cur_vec))
            
            cnt_E = 0
            for ind in order:
                if interesting_points[ind][3] == 'E':
                    cnt_E += 1
            X_out[it, 0] = interesting_points[order[0]][3]
            X_out[it, 1] = interesting_points[order[1]][3]
            X_out[it, 2] = interesting_points[order[2]][3]
            X_out[it, 3:6] = vec_lens
            X_out[it, 6:9] = angles
            X_out[it, 9] = interesting_points[order[1]][2] - interesting_points[order[0]][2]
            X_out[it, 10] = interesting_points[order[2]][2] - interesting_points[order[1]][2]
            X_out[it, 11] = interesting_points[order[0]][2] - interesting_points[order[2]][2]
            X_out[it, 12] = X_out[it, 0] + X_out[it, 1]
            X_out[it, 13] = X_out[it, 1] + X_out[it, 2]
            X_out[it, 14] = X_out[it, 2] + X_out[it, 0]
            X_out[it, 15] = X_out[it, 0] + X_out[it, 1] + X_out[it, 2]
            X_out[it, 16] = ''.join(sorted([X_out[it, 0], X_out[it, 1], X_out[it, 2]]))
            X_out[it, 17] = cnt_E
        return X_out

def to_X_y(dataset):
    return (dataset['structures'], dataset['targets'])


X_train, y_train = to_X_y(train)
X_test, y_test = to_X_y(test)
X_public, y_public = to_X_y(public_data)
X_private_test = private_test['structures']


transformer = CrystalTransformerRelativeWithShift()
X_train = transformer.fit_transform(X_train)
X_test = transformer.fit_transform(X_test)
X_public = transformer.fit_transform(X_public)
X_private_test = transformer.fit_transform(X_private_test)

def write_out_preds(preds):
    df_out = pd.DataFrame(data=preds, index=private_test.index, columns=['predictions']).reset_index().rename(columns={'index': 'id'})
    df_out.to_csv('submission.csv', index=False)

preds = None
cats = 10
random_seeds = [
    5197, 94889, 110359, 386989, 974107,
    608273, 315011, 361217, 925163, 651347
]
for run_idx in range(cats):
    catboost_best = CatBoostRegressor(
        cat_features=[0, 1, 2, 9, 10, 11, 12, 13, 14, 15, 16, 17],
        depth=8,
        l2_leaf_reg=1.85824300175498,
        learning_rate=0.08905944785869672,
        verbose=0,
        random_seed=random_seeds[run_idx],
    )
    catboost_best.fit(X_public, y_public)
    cur_preds = catboost_best.predict(X_private_test)
    if preds is None:
        preds = cur_preds
    else:
        preds += cur_preds
preds /= cats

write_out_preds(preds)
