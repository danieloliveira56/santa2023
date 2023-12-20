PUZZLE_TYPES = [
    'cube_2/2/2',
    'cube_3/3/3',
    'cube_4/4/4',
    'cube_5/5/5',
    'cube_6/6/6',
    'cube_7/7/7',
    'cube_8/8/8',
    'cube_9/9/9',
    'cube_10/10/10',
    'cube_19/19/19',
    'cube_33/33/33',
    'wreath_6/6',
    'wreath_7/7',
    'wreath_12/12',
    'wreath_21/21',
    'wreath_33/33',
    'wreath_100/100',
    'globe_1/8',
    'globe_1/16',
    'globe_2/6',
    'globe_3/4',
    'globe_6/4',
    'globe_6/8',
    'globe_6/10',
    'globe_3/33',
    'globe_33/3',
    'globe_8/25',
]
def get_inverse(permutation):
    return [
        i
        for i, _ in sorted(
            [(i, v) for i, v in enumerate(permutation)], key=lambda x: x[1]
        )
    ]


def read_solution(filename):
    with open(filename, "r") as f:
        lines = f.readlines()
    return [line.split(",")[1].strip().split(".") for line in lines[1:]]


def remove_identity(permutation):
    return permutation
    for i in range(len(permutation) - 1, 0):
        if (
            permutation[i] == f"-{permutation[i + 1]}"
            or permutation[i + 1] == f"-{permutation[i]}"
        ):
            permutation.pop(i)
            permutation.pop(i + 1)
    return permutation
