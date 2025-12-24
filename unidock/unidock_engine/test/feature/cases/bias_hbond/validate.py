"""
bias line on NA (No.6 from 0) is [1.782, 66.634, 6.337, -15, 1.0]
"""

import os
import json


def check_two_floats(a, b, lim = 0.2):
    if abs(a - b) > lim:
        print("{} != {}".format(a, b))
        exit(3)



if __name__ == "__main__":
    output_file = "input_1.json"
    coord_expected = [1.782, 66.634, 6.337]

    if not os.path.exists(output_file):
        exit(2)

    with open(output_file) as f:
        data = json.load(f)

    for pose in list(data.values())[0]:
        coord = pose["coords"][6 * 3: 6 * 3 + 3]
        check_two_floats(coord[0], coord_expected[0])
        check_two_floats(coord[1], coord_expected[1])
        check_two_floats(coord[2], coord_expected[2])

