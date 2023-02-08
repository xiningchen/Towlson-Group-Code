import bct
import pickle as pkl
import math

def community_by_modularity_stability(W, gamma, B='modularity', rep=2):
    partitions = []
    modularity = []
    for i in range(rep):
        partition, q = bct.community_louvain(W, gamma=gamma, B=B)
        partitions.append(partition)
        modularity.append(q)
    return partitions, modularity

def main(out_path, data_path, fname_i):
    #skip = {3: [0,1], 7: [0,1], 12: [0,1,2], 17: []}
    
    fname_i = int(fname_i)
    with open('./raw/lf_input_params.pkl', 'rb') as f:
        input_params = pkl.load(f)[fname_i]

    fname = input_params[0]
    idx = input_params[1]
    res_range = input_params[2]
    out_path = out_path + f'/{idx}/'
    if not os.path.exists(out_path):
        os.makedirs(out_path)
 
    with open(data_path+f'/{idx}/{fname}', 'rb') as f:
        avg_connectome = pkl.load(f)

    inc = 0.001
    for i in range(len(res_range)):
        # if i in skip[fname_i]:
        #     continue

        data = {'partitions_by_gamma': [], 'modularity_by_gamma_per_partition': []}
        gamma = []
        g = round(res_range[i][0], 3)
        max_g = round(res_range[i][1], 3)
        while not math.isclose(g, max_g):
            gamma.append(g)
            p_list, m_list = community_by_modularity_stability(avg_connectome, gamma=g, B='negative_asym', rep=1000)
            data['partitions_by_gamma'].append(p_list)
            data['modularity_by_gamma_per_partition'].append(m_list)
            g += inc
        data['gamma'] = gamma
        with open(os.path.join(out_path, fname[:len(fname)-4]+f'_auditory_{i}.pkl'), 'wb') as f:
            pkl.dump(data, f)
    print(fname, " --- COMPLETE ---")


if __name__ == '__main__':
    import sys, os
    out_path = sys.argv[1]
    data_path = sys.argv[2]
    fname_i = sys.argv[3]

    if not os.path.exists(out_path):
        print('Directory:', out_path, '\ndoes not exist, creating new one.')
        os.makedirs(out_path)
    else:
        print('Saving to', out_path)
    main(out_path, data_path, fname_i)
