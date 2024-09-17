cifar100 = {
    0:  'apple',
    1:  'aquarium_fish',
    2:  'baby',
    3:  'bear',
    4:  'beaver',
    5:  'bed',
    6:  'bee',
    7:  'beetle',
    8:  'bicycle',
    9:  'bottle',
    10: 'bowl',
    11: 'boy',
    12: 'bridge',
    13: 'bus',
    14: 'butterfly',
    15: 'camel',
    16: 'can',
    17: 'castle',
    18: 'caterpillar',
    19: 'cattle',
    20: 'chair',
    21: 'chimpanzee',
    22: 'clock',
    23: 'cloud',
    24: 'cockroach',
    25: 'couch',
    26: 'crab',
    27: 'crocodile',
    28: 'cup',
    29: 'dinosaur',
    30: 'dolphin',
    31: 'elephant',
    32: 'flatfish',
    33: 'forest',
    34: 'fox',
    35: 'girl',
    36: 'hamster',
    37: 'house',
    38: 'kangaroo',
    39: 'keyboard',
    40: 'lamp',
    41: 'lawn_mower',
    42: 'leopard',
    43: 'lion',
    44: 'lizard',
    45: 'lobster',
    46: 'man',
    47: 'maple_tree',
    48: 'motorcycle',
    49: 'mountain',
    50: 'mouse',
    51: 'mushroom',
    52: 'oak_tree',
    53: 'orange',
    54: 'orchid',
    55: 'otter',
    56: 'palm_tree',
    57: 'pear',
    58: 'pickup_truck',
    59: 'pine_tree',
    60: 'plain',
    61: 'plate',
    62: 'poppy',
    63: 'porcupine',
    64: 'possum',
    65: 'rabbit',
    66: 'raccoon',
    67: 'ray',
    68: 'road',
    69: 'rocket',
    70: 'rose',
    71: 'sea',
    72: 'seal',
    73: 'shark',
    74: 'shrew',
    75: 'skunk',
    76: 'skyscraper',
    77: 'snail',
    78: 'snake',
    79: 'spider',
    80: 'squirrel',
    81: 'streetcar',
    82: 'sunflower',
    83: 'sweet_pepper',
    84: 'table',
    85: 'tank',
    86: 'telephone',
    87: 'television',
    88: 'tiger',
    89: 'tractor',
    90: 'train',
    91: 'trout',
    92: 'tulip',
    93: 'turtle',
    94: 'wardrobe',
    95: 'whale',
    96: 'willow_tree',
    97: 'wolf',
    98: 'woman',
    99: 'worm'
}


def extract_leaf_groups(taxonomy, parent_category=None):
    """
    Recursively traverses the taxonomy and extracts sub-groups at the leaf level.
    
    :param taxonomy: The taxonomy dictionary.
    :param parent_category: The parent category (used for maintaining the hierarchy).
    :return: A list of dictionaries representing leaf-level sub-groups.
    """
    leaf_groups = []

    # Traverse the taxonomy
    for category, subcategory in taxonomy.items():
        # If the subcategory is a dictionary, recurse further
        if isinstance(subcategory, dict):
            leaf_groups.extend(extract_leaf_groups(subcategory, category))
        # If the subcategory is a list (i.e., leaf nodes), collect the current category
        elif isinstance(subcategory, list):
            leaf_groups.append({
                'group': category,
                'parent': parent_category,
                'labels': subcategory
            })

    return leaf_groups


def ID_to_name(ID_2D_list, map):
    label_2D_list = [[map[name] for name in sublist] for sublist in ID_2D_list]
    return label_2D_list

def name_to_ID(names_2D_list, dataset):
    label_to_ID = {label: ID for ID, label in dataset.items()}
    ids_2D_list = [[label_to_ID[name] for name in sublist] for sublist in names_2D_list]
    return ids_2D_list

def leaf_group_to_llist(taxonomy, dataset=cifar100):
    # Extract leaf-level sub-groups
    leaf_groups = extract_leaf_groups(taxonomy)
    # Reverse the dictionary
    
    llist = []

    # Example of how to use the extracted leaf groups
    for group in leaf_groups:
        # print(f"Group: {group['group']} (Parent: {group['parent']})")
        # print(f"Labels: {group['labels']}\n")
        llist.append(group['labels'])
        
    llist = name_to_ID(llist, dataset)
        
    return llist

if __name__ == '__main__':
    ID = [[42, 41, 91, 9, 65, 50, 1, 70, 15, 78], 
          [73, 10, 55, 56, 72, 45, 48, 92, 76, 37],
          [30, 21, 32, 96, 80, 49, 83, 26, 87, 33],
          [8, 47, 59, 63, 74, 44, 98, 52, 85, 12],
          [36, 23, 39, 40, 18, 66, 61, 60, 7, 34],
          [99, 46, 2, 51, 16, 38, 58, 68, 22, 62],
          [24, 5, 6, 67, 82, 19, 79, 43, 90, 20],
          [0, 95, 57, 93, 53, 89, 25, 71, 84, 77],
          [64, 29, 27, 88, 97, 4, 54, 75, 11, 69],
          [86, 13, 17, 28, 31, 35, 94, 3, 14, 81]]
    kk = ID_to_name(ID, cifar100)
    print(kk)

    # # Example taxonomy
    # taxonomy = {
    #     "Natural": {
    #         "Plants": {
    #             "Fruits": ["apple", "orange", "pear", "sweet_pepper"],
    #             "Flowers": ["orchid", "poppy", "rose", "sunflower", "tulip"],
    #             "Trees": ["maple_tree", "oak_tree", "palm_tree", "pine_tree", "willow_tree"]
    #         },
    #         "Animals": {
    #             "Mammals": {
    #                 "Four-legged": ["bear", "beaver", "camel", "cattle", "chimpanzee", "elephant",
    #                                 "fox", "hamster", "kangaroo", "leopard", "lion", "mouse", "otter",
    #                                 "porcupine", "possum", "rabbit", "raccoon", "seal", "skunk", "squirrel",
    #                                 "tiger", "wolf"],
    #                 "Two-legged": ["baby", "boy", "girl", "man", "woman"]
    #             },
    #             "Insects": ["bee", "beetle", "butterfly", "caterpillar", "cockroach", "spider"],
    #             "Aquatic": ["aquarium_fish", "crocodile", "dolphin", "flatfish", "lobster", "ray", "shark", "trout", "whale"],
    #             "Reptiles": ["dinosaur", "lizard", "snake", "turtle"],
    #             "Others": ["snail", "worm"]
    #         }
    #     },
    #     "Man-Made": {
    #         "Vehicles": {
    #             "Wheeled": ["bicycle", "bus", "motorcycle", "pickup_truck", "tractor", "train", "streetcar"],
    #             "Air": ["rocket"],
    #             "Water": []
    #         },
    #         "Structures": {
    #             "Buildings": ["castle", "house", "skyscraper"],
    #             "Bridges": ["bridge"],
    #             "Others": ["road"]
    #         },
    #         "Objects": {
    #             "Furniture": ["bed", "chair", "couch", "table", "wardrobe"],
    #             "Appliances": ["television"],
    #             "Containers": ["bottle", "bowl", "can", "cup", "plate"],
    #             "Instruments": ["clock", "keyboard", "lamp", "telephone"],
    #             "Tools": ["lawn_mower", "tank"]
    #         }
    #     },
    #     "Environment": {
    #         "Natural Features": ["cloud", "forest", "mountain", "plain", "sea"]
    #     }
    # }

    # # Extract leaf-level sub-groups
    # leaf_groups = extract_leaf_groups(taxonomy)

    # # Example of how to use the extracted leaf groups
    # for group in leaf_groups:
    #     print(f"Group: {group['group']} (Parent: {group['parent']})")
    #     print(f"Labels: {group['labels']}\n")