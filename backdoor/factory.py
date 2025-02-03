from .heteroBA import HeteroBA
from .UGBA import UGBA
from .CGBA import CGBA

def construct_backdoor_model(args, features, device, hg, hete_adjs, clean_labels, poison_labels, num_classes, posion_trainset_index, posion_testset_index, \
        clean_trainset_index, clean_testset_index, trainset_index, testset_index, primary_type, metapaths,text_attribute, edge_template, homo_g, node_mapping):
    backdoor_type = args.backdoor_type

    backdoors = {'HeteroBA': HeteroBA,
                 'UGBA': UGBA,
                 'CGBA': CGBA,}

    if backdoor_type not in backdoors:
        raise TypeError(f"Unknown backdoor type: {backdoor_type}. Available types are {list(backdoors.keys())}")

    return backdoors[backdoor_type](args, features, device, hg, hete_adjs, clean_labels, poison_labels, num_classes, posion_trainset_index, posion_testset_index, \
        clean_trainset_index, clean_testset_index, trainset_index, testset_index, primary_type, metapaths,text_attribute, edge_template, homo_g, node_mapping)
