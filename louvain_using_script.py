from collections import Counter
import numpy as np
from tqdm import tqdm
import bct
from sklearn.metrics.cluster import adjusted_rand_score
import math
import pickle as pkl

def find_resolution_range(W, min_community_num, max_community_num, min_gamma, max_gamma, inc, B='modularity', sample_size=1):
    x_axis = []
    y_axis = []
    good_res = []
    max_range = math.ceil((max_gamma - min_gamma) / inc)
    for r in tqdm(range(0, max_range)):
        res = min_gamma + r * inc
        x_axis.append(res)
        c, q = bct.community_louvain(W, gamma=res, B=B)
        avg_num = len(Counter(c))
        if avg_num >= min_community_num and avg_num <= max_community_num:
            for i in range(0,sample_size):
                c, q = bct.community_louvain(W, gamma=res, B=B)
                avg_num += len(Counter(c))
            avg_num = avg_num/sample_size
            y_axis.append(avg_num)
            if avg_num >= min_community_num and avg_num <= max_community_num:
                good_res.append((res, avg_num))
        else:
            y_axis.append(len(Counter(c)))
    return good_res, x_axis, y_axis


def community_by_modularity_stability(W, res_lower_range, res_upper_range, inc, B='modularity', rep=2):
    count = math.comb(rep, 2)
    r_upper = math.ceil((res_upper_range - res_lower_range)/inc)
    partition_by_gamma = {r: [] for r in range(0, r_upper)}
    partition_list_by_gamma = {r: [] for r in range(0, r_upper)}
    stability_by_gamma = {r: 0.0 for r in range(0, r_upper)}
    modularity_by_gamma = {r: 0.0 for r in range(0, r_upper)}
    modularity_by_gamma_per_partition = {r: [] for r in range(0, r_upper)}
    for i in tqdm(range(0, rep)):
        for r in range(0, r_upper):
            res = res_lower_range + inc * r
            partition, q = bct.community_louvain(W, gamma=res, B=B)
            partition_list = [p for p in partition]
            if len(partition_by_gamma[r]) >= 1:
                # Compute a z-rand score
                for plist in partition_list_by_gamma[r]:
                    stability_by_gamma[r] += adjusted_rand_score(plist, partition_list)
            partition_by_gamma[r].append(partition)
            partition_list_by_gamma[r].append(partition_list)
            modularity_by_gamma_per_partition[r].append(q)
            # Compute modularity value  ****** is q returned modularity value? Or something q-statistc??
            modularity_by_gamma[r] += q
    # Calculate average z-rand score per gamma
    avg_stab_by_gamma = np.array([z / count for r, z in stability_by_gamma.items()])
    # Calculate average modularity score per gamma
    avg_modularity_by_gamma = np.array([mod / rep for r, mod in modularity_by_gamma.items()])
    # Calculate weighted avg z-rand = <mod>*<z-rand> per gamma
    weighted_avg_zscore = avg_modularity_by_gamma * avg_stab_by_gamma
    # Find max(weighted avg z-rand) --> gamma_max
    index_max = np.argmax(weighted_avg_zscore)
    if weighted_avg_zscore[index_max] != max(weighted_avg_zscore):
        print("Max gamma incorrect - something wrong.")
    return res_lower_range + inc*index_max, index_max, avg_modularity_by_gamma, avg_stab_by_gamma, partition_by_gamma, modularity_by_gamma_per_partition


def main(path, data_path, phase):
    if phase=='1':
        phase = "EF"
    if phase=='2':
        phase = "LF"
    if phase=='3':
        phase = "ML"

    pickle_file = data_path + '/' + f'averaged-{phase}-2022-04-27.pkl'
    if os.path.exists(pickle_file):
        with open(pickle_file, 'rb') as f:
            avg_connectome = pkl.load(f)
            print(f'path = {path} \n data_path = {data_path} \n phase = {phase}')
            print("opened connectome: ", avg_connectome)

    res_range = {'EF': [1.077, 1.112], 'LF': [1.07, 1.125], 'ML': [1.061, 1.102]}
    _, _, _, _, partition_by_gamma, modularity_by_gamma_per_partition = community_by_modularity_stability(avg_connectome,
                                                                                      res_range[phase][0],
                                                                                      res_range[phase][1],
                                                                                      inc=0.001,
                                                                                      B='negative_asym',
                                                                                      rep=1000)
    good_count = 0
    continue_search = True
    i = 0
    while continue_search:
        for partition_list in partition_by_gamma[i]:
            if len(Counter(partition_list)) == 7:
                good_count += 1
        if good_count >= 700:
            continue_search = False
            print("phase\t gamma \t good_count \t index i")
            print(phase, res_range[phase][0] + 0.001 * i, good_count, i)
        else:
            i += 1
    max_mod = max(modularity_by_gamma_per_partition[i])
    max_mod_idx = modularity_by_gamma_per_partition[i].index(max_mod)
    best_partition = partition_by_gamma[i][max_mod_idx]

    file_path = os.path.join(path, f'best_partition_{phase}.pkl')
    with open(file_path, 'wb') as f:
        pkl.dump(best_partition, f)
    f.close()
    print(f"Saved to {file_path}.")


if __name__ == '__main__':
    import sys, os

    path = sys.argv[1]
    data_path = sys.argv[2]
    phase = sys.argv[3]
    if not os.path.exists(path):
        print('Directory:', path, '\ndoes not exist, creating new one.')
        os.makedirs(path)
    else:
        print('Saving to', path)
    main(path, data_path, phase)