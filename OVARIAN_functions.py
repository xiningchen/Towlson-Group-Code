import pandas as pd
from sklearn.metrics.cluster import normalized_mutual_info_score
from collections import Counter
import importlib
import Towlson_group_code.functional_brain_community as brain_community
import Towlson_group_code.data_io as myFunc
import Towlson_group_code.fsleyes_customizer as fsleyes
import pickle as pkl
import os
import random
import numpy as np
import math
importlib.reload(myFunc)
importlib.reload(brain_community)
importlib.reload(fsleyes)

AVG_EF_FN_AUDITORY = {0: 'DMN', 1: 'Somamotor', 2: 'Visual', 3: 'Frontoparietal', 4: 'Dorsal Attention',
                      5: 'Ventral Attention', 6: 'Limbic', 7: 'Auditory', 8: 'Unknown'}
AVG_EF_FN = {2: 'visual', 1: 'somamotor', 6: 'dorsal attention?', 3: 'ventral attention/salience',
                 5: 'limbic', 0: 'DMN', 4: 'frontoparietal control', 7: 'Unknown'}
AVG_ML_FN = {0: 'DMN', 1: 'somamotor', 2: 'visual', 3: 'ventral attention/salience',
                 4: 'frontoparietal control', 5: 'Unknown', 6: 'limbic', 7: 'dorsal attention?'}

def get_pid_from_idx(idx):
    path = f'../Ovarian_hormone/ARC/subject_level/Results2_Rep/{idx}'
    for root, dirs, files in os.walk(path):
        for file in files:
            if not file.endswith('.pkl'):
                continue
            return file[3:6]
    return ""

def get_cycles():
    oam_induction = pd.read_excel('../Ovarian_hormone/data/OAM Inductions.xlsx', sheet_name='Sheet2', header=0, index_col=None, usecols='A:C')
    oam_induction['PID'] = oam_induction['Scan ID'].str[3:6]
    oam_induction['date'] = oam_induction['Scan ID'].str[6:]
    unique_id_list = set(oam_induction['PID'])
    cycles = {id: [] for id in unique_id_list}
    for id in unique_id_list:
        cycle = []
        subject_induction = oam_induction[oam_induction['PID'] == id]
        for _, row in subject_induction.iterrows():
            cycle.append(f"{row['Menstrual Phase']}-{row['PID']}-{row['date']}")
            if len(cycle) == 3:
                cycles[id].append(sorted(cycle, key=lambda x: x[0]))
                cycle = []
    return cycles

def get_node_list():
    node_pd = pd.read_excel('../Ovarian_hormone/data/node_cog.xlsx', index_col=0, header=0, sheet_name='Sheet_name_1', usecols='A,B')
    return node_pd

def similarity(set1, set2):
    # Simplest definition is the number of overlapping elements between the two sets
    sim_n = len(set1.intersection(set2))
    return sim_n, sim_n/len(set1), sim_n/len(set2)

# def get_highest_similarity_partition(target_partition, partitions):
#     nmis = []
#     for i in range(len(partitions)):
#         nmis.append(normalized_mutual_info_score(target_partition, partitions[i]))
#     return nmis

def get_8_modules(partition):
    """
    Gets top 7 largest modules + an 8th module that includes all remaining nodes
    :param partition: List of integers representing a module the node belongs to
    :return: Sorted list of modules from largest to smallest 7-th module, with an 8th module.
    """
    node_pd = get_node_list()
    top_7_modules = [m for m, ms in Counter(partition).most_common(7)]
    partition_8 = [p if p in top_7_modules else 1000 for p in partition]
    keys = [m for m in top_7_modules] + [1000]
    reindex_map = dict(zip(keys, range(8)))
    partition_norm = [reindex_map[m] for m in partition_8]
    sets = []
    for m in range(8):
        elements = [node_pd.iloc[i]['region_name'] for i, c in enumerate(partition_norm) if c == m]
        sets.append(set(elements))
    return sets, partition_norm

def get_all_modules(partition):
    """
    Get all modules of a partition.
    :param partition: List of integers representing a module the node belongs to
    :return: Sorted list of modules from largest to smallest.
    """
    node_pd = get_node_list()
    top_modules = [m for m, ms in Counter(partition).most_common()]
    reindex_map = dict(zip(top_modules, range(len(top_modules))))
    partition_norm = [reindex_map[m] for m in partition]
    sets = []
    for m in range(len(top_modules)):
        elements = [node_pd.iloc[i]['region_name'] for i, c in enumerate(partition_norm) if c == m]
        sets.append(set(elements))
    return sets, partition_norm

def get_modules(partition, module_definition):
    node_pd = get_node_list()
    modules = []
    for m, mname in module_definition.items():
        elements = [node_pd.iloc[i]['region_name'] for i, c in enumerate(partition) if c == m]
        modules.append(set(elements))
    return modules

def get_stable_matching(men_prefs, women_prefs):
    """
    Deferred Acceptance Algorithm.
    :param men_prefs: Dictionary of men and a list of their highest preference to lowest.
    :param women_prefs: Dictionary of women and a list of their highest preference to lowest.
    :return: A list of pairs, representing matchings.
    """
    pairs = {w: None for w in women_prefs.keys()}
    single_men = set(men_prefs.keys())
    while single_men:
        man = single_men.pop()
        for woman, score in men_prefs[man].items():
            # print(f'{man} proposing to {woman}.')
            partner = pairs[woman]
            if partner is None:
                # print(f'\t{woman} is single. Engaged to {man}. ')
                pairs[woman] = man
                break
            elif women_prefs[woman][partner] < women_prefs[woman][man]:
                # print(f'\t{woman} prefers {man} over partner {partner} ({women_prefs[woman][partner]} < {women_prefs[woman][man]}). ')
                pairs[woman] = man
                single_men.add(partner)
                break
    return pairs

def get_poly_matching(men_prefs, women_prefs, poly_threshold=1):
    """
    Modified Deferred Acceptance Algorithm where an individual can have more than 1 partner.
    This could result in unmatched individuals, which will be grouped into a final "loner" module.
    The condition for allowing more than 1 partner can be toggled by the poly_threshold parameter.
    Women will only upgrade partners that are not 100% "committed" (preference).
    :param men_prefs: Dictionary of men and a list of their highest preference to lowest.
    :param women_prefs: Dictionary of women and a list of their highest preference to lowest. "Target"
    :param poly_threshold: Threshold for allowing multiple partner matching. By default, threshold is 1, meaning that a
    poly relationship will occur if the m_i's preference is 100% for w_i.
    :return: A list of pairs, representing matchings.
    """
    pairs = {w: None for w in women_prefs.keys()}
    committed_pairs = {w: [] for w in women_prefs.keys()}
    single_men = set(men_prefs.keys())
    loners = set()
    while single_men:
        man = single_men.pop()
        for woman, score in men_prefs[man].items():
            hope = True
            # print(f'{man} proposing to {woman} with a score of {score}.')
            if math.isclose(score, 0):
                # print(f'\t{man} does not want to be with this woman. Next.')
                hope = False
                continue
            # -- If man's preference for woman is 100%, automatically pair.
            if math.isclose(score, 1):
                # print(f'\t{man} is very committed to {woman}.')
                committed_pairs[woman].append(man)
                break
            if pairs[woman] is None:
                # print(f'\t{woman} is single. Engaged to {man}. ')
                pairs[woman] = {man}
                break
            else:
                partners = [p for p in pairs[woman]]
                hope = False
                # print(f"\t{woman} is engaged to ", partners)
                for partner in partners:
                    if abs(women_prefs[woman][man] - women_prefs[woman][partner]) <= 0.1 or score >= poly_threshold:
                        # print(f"\t{woman} can't decide or man will try really hard. Enter poly relationship.")
                        # --- Woman cannot decide. So choose poly relationship.
                        pairs[woman].add(man)
                        hope = True
                        break
                    elif women_prefs[woman][man] > women_prefs[woman][partner]:
                        # print(f"\t{woman}'s pref. for man {man} vs partner {partner} is {women_prefs[woman][man]:0.4f} "
                        #     f"vs. {women_prefs[woman][partner]:0.4f}. Replacing partner with man.")
                        # --- Replace this partner with man.
                        pairs[woman].add(man)
                        pairs[woman].remove(partner)
                        single_men.add(partner)
                        hope = True
                    # -- Woman prefers this partner over man.
                    # print(f"\t{woman} prefers partner over man. Do nothing.")
                if hope:
                    break
        if not hope:
            # print(f"{man} is a loner.")
            loners.add(man)
    # -- Combine pairs, committed pairs, and loners
    for w, m in pairs.items():
        if m is not None:
            committed_pairs[w] += m
    committed_pairs['loners'] = list(loners)
    #
    # print("Committed pairs", committed_pairs)
    poly_matching = {}
    for y, l in committed_pairs.items():
        if y == 'loners':
            continue
        poly_matching[AVG_EF_FN_AUDITORY[y]] = l
    if 'Unknown' in poly_matching.keys():
        poly_matching['Unknown'] += committed_pairs['loners']
    else:
        poly_matching['Unknown'] = committed_pairs['loners']
    return poly_matching

def merge_poly_modules(partition, poly_matching):
    # print("Poly match: ", poly_matching)
    icns = {}
    for fn, match_list in poly_matching.items():
        for l in match_list:
            icns[l] = fn
    avg_FN_rev = {v: k for k, v in AVG_EF_FN_AUDITORY.items()}
    remod_avg = []
    for x in partition:
        remod_avg.append(avg_FN_rev[icns[x]])
    return remod_avg

def get_preference_list(group1, candidates):
    preference_dict = {}
    for p_num, p in enumerate(group1):
        pref = []
        for j_num, j in enumerate(candidates):
            if len(p) == 0 or len(j) == 0:
                pref.append((j_num, 0))
                continue
            _, p_pref, _ = similarity(p, j)
            pref.append((j_num, p_pref))
        sorted_pref = sorted(pref, key=lambda x: x[1], reverse=True)
        preference_dict[p_num] = {k: v for k, v in sorted_pref}
    return preference_dict

def get_module_across_phases(ef_partition, lf_partition, ml_partition):
    """
    Use Stable Matching between EF - LF, and LF - ML to find best matching modules.
    Compare EF -> LF -> ML modules for a cycle and get the best matching module tracking.
    :param ef_modules: List of sets of nodes representing modules for EF phase of a cycle.
    :param lf_modules: List of sets of nodes representing modules for LF phase of a cycle.
    :param ml_modules: List of sets of nodes representing modules for ML phase of a cycle.
    :return: List of lists containing the matched modules across phases
    """
    ef_modules, ef_partition_norm = get_all_modules(ef_partition)
    lf_modules, lf_partition_norm = get_all_modules(lf_partition)
    # Create preferences between EF and LF
    ef_preferences = get_preference_list(ef_modules, lf_modules)
    lf_preferences = get_preference_list(lf_modules, ef_modules)

    match1 = get_poly_matching(lf_preferences, ef_preferences, poly_threshold=0.6)
    lf_merged_partition = merge_poly_modules(lf_partition_norm, match1)
    print("Merged LF partition.")
    print(Counter(lf_merged_partition))
        # print("EF preferences")
        # print(ef_preferences)
        # print("LF preferences")
        # print(lf_preferences)
        # print(match1)
        # print("\n")
    # Create preferences between ML and LF
    lf_modules, lf_merged_partition_norm = get_all_modules(lf_merged_partition)
    ml_modules, ml_partition_norm = get_all_modules(ml_partition)
    lf_preferences = get_preference_list(lf_modules, ml_modules)
    ml_preferences = get_preference_list(ml_modules, lf_modules)
    match2 = get_poly_matching(ml_preferences, lf_preferences, poly_threshold=0.6)
    ml_merged_partition = merge_poly_modules(ml_partition_norm, match2)
        # print("lF preferences")
        # print(lf_preferences)
        # print("ml preferences")
        # print(ml_preferences)
        # print(match2)
    # Connect the two matchings
    print(match1)
    print(match2)
    # module_across_phases = [[ef, lf, match2[lf]] for ef, lf in match1.items()]
    # return module_across_phases

def get_FN_map_for_cycle(ef_partition, lf_partition, ml_partition, avg_ef_partition, ef_FN):
    node_pd = get_node_list()
    print("EF: ", set(ef_partition))
    print("LF: ", set(lf_partition))
    print("ML: ", set(ml_partition))
    # --- Old code
    # ef_modules, _ = get_8_modules(ef_partition)
    # lf_modules, _ = get_8_modules(lf_partition)
    # ml_modules, _ = get_8_modules(ml_partition)
    # --- New code
    get_module_across_phases(ef_partition, lf_partition, ml_partition)

    # Custom get_modules function for processing averaged EF partition using Joe's assignments
    # top_7 = [1,2,3,4,5,7,8]
    # partition_8 = [p if p in top_7 else 1000 for p in avg_ef_partition]
    # keys = [m for m in top_7] + [1000]
    # reindex_map = dict(zip(keys, range(8)))
    # partition_norm = [reindex_map[m] for m in partition_8]
    # avg_ef_modules = []
    # for m in range(8):
    #     elements = [node_pd.iloc[i]['region_name'] for i, c in enumerate(partition_norm) if c == m]
    #     avg_ef_modules.append(set(elements))
    #
    # avg_ef_prefs = get_preference_list(avg_ef_modules, ef_modules)
    # ef_prefs = get_preference_list(ef_modules, avg_ef_modules)
    # avg_ef_match = get_stable_matching(avg_ef_prefs, ef_prefs)
    #
    # # Compare EF module to average EF FN assignments from Joe
    # fn_cycle = {ef_FN[avg_ef_match[match[0]]]: match for match in matched_modules}
    # return fn_cycle


def load_existing_data(idx):
    path = f'../Ovarian_hormone/ARC/subject_level/Results2_Rep/{idx}'
    path2 = f'../Ovarian_hormone/ARC/subject_level/Results2_Redo_Rep/{idx}/'
    data_dict = {}
    for root, dirs, files in os.walk(path):
        for file in files:
            if not file.endswith('.pkl'):
                continue
            # Check if there's was a redo run for this file. If yes, redo take priority.
            if os.path.exists(path2+file):
                with open(path2+file, 'rb') as f:
                    data_dict[file] = pkl.load(f)
            else:
                with open(root + '/' + file, 'rb') as f:
                    data_dict[file] = pkl.load(f)
    return data_dict

def group_partitions_by_similarity(list_of_partitions, threshold, sample_size=10):
    """
    Given a grouping threshold, group the list of partitions into subgroups based on NMI similarity value by
    comparing a partition with 10 randomly selected members of a group. If the average similarity value passes
    the threshold, then add partition to this group.
    :param list_of_partitions: a list of pairs of the form: (<partition>, <gamma value>)
    :param threshold: threshold value for defining groups
    :return: The best representative partition for each group, the mean gamma for each group, and size of each group
    """
    partition_groups = []
    partition_groups_gamma = []
    for partition, gamma in list_of_partitions:
        if len(partition_groups) == 0:
            partition_groups.append([partition])
            partition_groups_gamma.append([gamma])
            continue
        merged = False
        for gi, group in enumerate(partition_groups):
            repeat = 0
            s = 0
            group_size = len(group)-1
            while repeat < sample_size:
                i = random.randint(0, group_size)
                s += normalized_mutual_info_score(group[i], partition)
                repeat += 1
            if s/sample_size > threshold:
                group.append(partition)
                partition_groups_gamma[gi].append(gamma)
                merged = True
                break
        if merged == False:
            partition_groups.append([partition])
            partition_groups_gamma.append([gamma])
    reppg = []
    group_size = []
    # avg_gamma = []
    for gi, group in enumerate(partition_groups):
        gs = len(group)
        group_size.append(gs)
        # avg_gamma.append(partition_groups_gamma[gi]/gs)
        reppg.append(brain_community.get_best_partition(group))
    return reppg, partition_groups_gamma, group_size

def export_for_fsleyes(project, partition, fname, btype, reindex="top8"):
    DATA_PATH = f'../{project}/data/'
    FILE = fname + '.txt'
    # ---- PART 1
    if reindex == "top8":
        x = [p[0] for p in Counter(partition).most_common()]
        # RE-INDEX Community # from 1 = largest community, to smaller communities
        reindex_map = dict(zip(x, np.arange(1, len(Counter(partition)) + 1)))
        reindex_partition = [reindex_map[c] for c in partition]
    if reindex == "+1":
        reindex_partition = [c+1 for c in partition]
    else:
        reindex_partition = partition
    # ---- PART 2
    node_cog_df = myFunc.import_XLSX(DATA_PATH, 'node_cog.xlsx')
    node_list = list(node_cog_df['region_name'])
    node_list_2 = [n.replace("_", "-") for n in node_list]

    data_formatted = dict(zip(node_list_2, reindex_partition))
    # Export to txt in format described above
    buffer = ""
    for n, c in data_formatted.items():
        buffer += n + " " + str(c) + "\n"
    with open(f'../{project}/Brain_Atlas/data_to_plot/' + FILE, 'w') as f:
        f.write(buffer)

    # ---- PART 3
    if btype == 'both':
        types = ['cortical', 'subcortical']
    else:
        types = [btype]
    for btype in types:
        # Check file path
        os.path.abspath(f"../{project}/Brain_Atlas/data_to_plot/" + FILE)
        absolute_path = f"/Users/shine/Documents/MSc/Neuro Research/{project}/Brain_Atlas/"
        data_file_path = absolute_path + 'data_to_plot/'
        output_path = absolute_path + 'fsleyes_custom/'

        if btype == 'cortical':
            lut_file = absolute_path + 'Cortical.nii.txt'
            nii_file = absolute_path + 'Schaefer2018_1000Parcels_7Networks_order_FSLMNI152_2mm.nii.gz'
        if btype == 'subcortical':
            lut_file = absolute_path + 'Subcortical.nii.txt'
            nii_file = absolute_path + 'Tian_Subcortex_S4_3T_1mm.nii.gz'

        # In my case I need to call formatter.
        data_txt_file = absolute_path + 'data_to_plot/' + FILE
        formatted_data = fsleyes.format_data(data_txt_file, lut_file)
        fsleyes.run_customizer(output_path, lut_file, nii_file, fname=f'{FILE[:len(FILE) - 4]}_{btype}',
                               data_values=formatted_data)

def create_cortical_lut(partition, fname):
    color_rgb = {0: [255, 51, 51], 1: [102, 179, 255], 2: [179, 102, 255], 3: [255, 179, 102], 4: [0, 153, 77],
                 5: [255, 204, 255], 6: [245, 211, 20], 7: [201, 0, 117], 8: [128, 128, 128], -1: [0,0,0]}
    lut_f = '../Ovarian_hormone/Brain_Atlas/Schaefer2018_1000Parcels_7Networks_order.lut'
    f = open(lut_f, "r")
    file_content = f.readlines()
    f.close()
    my_file_content = ""
    for l, line in enumerate(file_content):
        vec = line.split(" ")
        for i in range(1, 4):
            vec[i] = str(round(color_rgb[partition[l]][i - 1] / 255, 5))
        my_file_content += ' '.join(vec)
    with open(f"../Ovarian_hormone/Brain_Atlas/custom_lut/{fname}.lut", 'w') as output_file:
        output_file.write(my_file_content)



def create_subcortical_lut(partition, fname):
    color_rgb = {0: [255, 51, 51], 1: [102, 179, 255], 2: [179, 102, 255], 3: [255, 179, 102], 4: [0, 153, 77],
                 5: [255, 204, 255], 6: [245, 211, 20], 7: [201, 0, 117], 8: [128, 128, 128], -1: [0,0,0]}
    lut_f = '../Ovarian_hormone/Brain_Atlas/Subcortical.nii.txt'
    f = open(lut_f, "r")
    file_content = f.readlines()
    f.close()
    my_file_content = ""
    for l, line in enumerate(file_content):
        vec = line.split(" ")[:-1]
        my_file_content += f"{vec[0]} {round(color_rgb[partition[1000 + l]][0] / 255, 5)} " \
                           f"{round(color_rgb[partition[1000 + l]][1] / 255, 5)} " \
                           f"{round(color_rgb[partition[1000 + l]][2] / 255, 5)} " \
                           f"{vec[1]}\n"

    with open(f"../Ovarian_hormone/Brain_Atlas/custom_lut/{fname}_subcortex.lut", 'w') as output_file:
        output_file.write(my_file_content)
