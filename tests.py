#!.env/bin/python3

import numpy as np
import matplotlib.pyplot as plt

#
# results = [{'positive_ratings': 141, 'f_score': 0.0}, {'positive_ratings': 128, 'f_score': 0.7142857142857142}, {'positive_ratings': 90, 'f_score': 0.5}, {'positive_ratings': 78, 'f_score': 0.9166666666666666}, {'positive_ratings': 65, 'f_score': 0.8333333333333333}, {'positive_ratings': 62, 'f_score': 0.4411764705882354}, {'positive_ratings': 49, 'f_score': 0.0}, {'positive_ratings': 40, 'f_score': 0.7142857142857142}, {'positive_ratings': 40, 'f_score': 1.0}, {'positive_ratings': 35, 'f_score': 1.0}, {'positive_ratings': 34, 'f_score': 0.0}, {'positive_ratings': 28, 'f_score': 0.9615384615384615}, {'positive_ratings': 27, 'f_score': 0.0}, {'positive_ratings': 25, 'f_score': 0.8823529411764707}, {'positive_ratings': 21, 'f_score': 0.4545454545454546}, {'positive_ratings': 21, 'f_score': 1.0}, {'positive_ratings': 20, 'f_score': 0.0}, {'positive_ratings': 17, 'f_score': 0.0}, {'positive_ratings': 16, 'f_score': 0.0}, {'positive_ratings': 15, 'f_score': 0.0}, {'positive_ratings': 15, 'f_score': 0.0}, {'positive_ratings': 15, 'f_score': 0.6666666666666666}, {'positive_ratings': 14, 'f_score': 0.0}, {'positive_ratings': 13, 'f_score': 1.0}, {'positive_ratings': 13, 'f_score': 0.0}, {'positive_ratings': 13, 'f_score': 0.0}, {'positive_ratings': 12, 'f_score': 0.0}, {'positive_ratings': 9, 'f_score': 0.0}, {'positive_ratings': 9, 'f_score': 0.0}, {'positive_ratings': 8, 'f_score': 1.0}, {'positive_ratings': 8, 'f_score': 0.0}, {'positive_ratings': 8, 'f_score': 0.38461538461538464}, {'positive_ratings': 7, 'f_score': 0.8823529411764707}, {'positive_ratings': 7, 'f_score': 0.0}, {'positive_ratings': 7, 'f_score': 0.0}, {'positive_ratings': 6, 'f_score': 0.0}, {'positive_ratings': 6, 'f_score': 0.0}, {'positive_ratings': 6, 'f_score': 0.0}, {'positive_ratings': 5, 'f_score': 0.0}, {'positive_ratings': 5, 'f_score': 0.0}, {'positive_ratings': 5, 'f_score': 0.0}, {'positive_ratings': 4, 'f_score': 0.0}, {'positive_ratings': 4, 'f_score': 0.0}, {'positive_ratings': 4, 'f_score': 0.0}, {'positive_ratings': 4, 'f_score': 0.0}, {'positive_ratings': 4, 'f_score': 0.0}, {'positive_ratings': 4, 'f_score': 0.0}, {'positive_ratings': 3, 'f_score': 0.0}, {'positive_ratings': 3, 'f_score': 0.0}, {'positive_ratings': 3, 'f_score': 0.0}, {'positive_ratings': 3, 'f_score': 0.0}, {'positive_ratings': 3, 'f_score': 0.0}, {'positive_ratings': 3, 'f_score': 0.0}, {'positive_ratings': 3, 'f_score': 0.0}, {'positive_ratings': 3, 'f_score': 0.0}, {'positive_ratings': 3, 'f_score': 0.0}, {'positive_ratings': 3, 'f_score': 0.0}, {'positive_ratings': 3, 'f_score': 0.0}, {'positive_ratings': 2, 'f_score': 0.0}, {'positive_ratings': 2, 'f_score': 0.0}, {'positive_ratings': 2, 'f_score': 0.0}, {'positive_ratings': 2, 'f_score': 0.0}, {'positive_ratings': 2, 'f_score': 0.0}, {'positive_ratings': 2, 'f_score': 0.0}, {'positive_ratings': 2, 'f_score': 0.0}, {'positive_ratings': 2, 'f_score': 0.7142857142857144}, {'positive_ratings': 2, 'f_score': 0.0}, {'positive_ratings': 2, 'f_score': 0.0}, {'positive_ratings': 2, 'f_score': 0.0}, {'positive_ratings': 2, 'f_score': 0.0}, {'positive_ratings': 2, 'f_score': 0.0}, {'positive_ratings': 2, 'f_score': 0.0}, {'positive_ratings': 2, 'f_score': 0.0}, {'positive_ratings': 2, 'f_score': 0.0}, {'positive_ratings': 2, 'f_score': 0.0}, {'positive_ratings': 2, 'f_score': 0.0}, {'positive_ratings': 2, 'f_score': 0.0}, {'positive_ratings': 2, 'f_score': 0.0}, {'positive_ratings': 2, 'f_score': 0.0}, {'positive_ratings': 2, 'f_score': 0.0}, {'positive_ratings': 1, 'f_score': 0.0}, {'positive_ratings': 1, 'f_score': 0.0}, {'positive_ratings': 1, 'f_score': 0.0}, {'positive_ratings': 1, 'f_score': 0.8333333333333333}, {'positive_ratings': 1, 'f_score': 0.0}, {'positive_ratings': 1, 'f_score': 0.0}, {'positive_ratings': 1, 'f_score': 0.0}, {'positive_ratings': 1, 'f_score': 0.0}, {'positive_ratings': 1, 'f_score': 0.0}, {'positive_ratings': 1, 'f_score': 0.0}, {'positive_ratings': 1, 'f_score': 0.0}, {'positive_ratings': 1, 'f_score': 0.0}, {'positive_ratings': 1, 'f_score': 0.0}, {'positive_ratings': 1, 'f_score': 0.0}, {'positive_ratings': 1, 'f_score': 0.0}, {'positive_ratings': 1, 'f_score': 0.0}, {'positive_ratings': 1, 'f_score': 0.0}, {'positive_ratings': 1, 'f_score': 0.0}, {'positive_ratings': 1, 'f_score': 0.0}, {'positive_ratings': 1, 'f_score': 0.0}, {'positive_ratings': 1, 'f_score': 0.0}, {'positive_ratings': 1, 'f_score': 0.0}, {'positive_ratings': 1, 'f_score': 0.0}, {'positive_ratings': 1, 'f_score': 0.0}, {'positive_ratings': 1, 'f_score': 0.0}, {'positive_ratings': 1, 'f_score': 0.0}, {'positive_ratings': 1, 'f_score': 0.0}, {'positive_ratings': 1, 'f_score': 0.0}, {'positive_ratings': 1, 'f_score': 0.0}, {'positive_ratings': 1, 'f_score': 0.0}, {'positive_ratings': 1, 'f_score': 0.0}, {'positive_ratings': 1, 'f_score': 0.0}, {'positive_ratings': 1, 'f_score': 0.0}, {'positive_ratings': 1, 'f_score': 0.0}, {'positive_ratings': 1, 'f_score': 0.0}, {'positive_ratings': 1, 'f_score': 0.0}, {'positive_ratings': 1, 'f_score': 0.0}, {'positive_ratings': 1, 'f_score': 0.0}, {'positive_ratings': 1, 'f_score': 0.0}, {'positive_ratings': 1, 'f_score': 0.0}, {'positive_ratings': 1, 'f_score': 0.0}, {'positive_ratings': 1, 'f_score': 0.0}, {'positive_ratings': 1, 'f_score': 0.0}, {'positive_ratings': 1, 'f_score': 0.0}, {'positive_ratings': 1, 'f_score': 0.0}, {'positive_ratings': 1, 'f_score': 0.0}, {'positive_ratings': 1, 'f_score': 0.0}, {'positive_ratings': 1, 'f_score': 0.0}, {'positive_ratings': 1, 'f_score': 0.0}, {'positive_ratings': 1, 'f_score': 0.0}, {'positive_ratings': 1, 'f_score': 0.0}, {'positive_ratings': 1, 'f_score': 0.0}, {'positive_ratings': 1, 'f_score': 0.0}, {'positive_ratings': 1, 'f_score': 0.0}, {'positive_ratings': 1, 'f_score': 0.0}, {'positive_ratings': 1, 'f_score': 0.0}, {'positive_ratings': 1, 'f_score': 0.0}, {'positive_ratings': 1, 'f_score': 0.0}, {'positive_ratings': 1, 'f_score': 0.0}, {'positive_ratings': 1, 'f_score': 0.0}, {'positive_ratings': 1, 'f_score': 0.0}, {'positive_ratings': 1, 'f_score': 0.0}, {'positive_ratings': 1, 'f_score': 0.0}, {'positive_ratings': 1, 'f_score': 0.0}, {'positive_ratings': 1, 'f_score': 0.0}, {'positive_ratings': 1, 'f_score': 0.0}, {'positive_ratings': 1, 'f_score': 0.0}, {'positive_ratings': 1, 'f_score': 0.0}, {'positive_ratings': 1, 'f_score': 0.0}, {'positive_ratings': 1, 'f_score': 0.0}, {'positive_ratings': 1, 'f_score': 0.0}, {'positive_ratings': 1, 'f_score': 0.0}, {'positive_ratings': 1, 'f_score': 0.0}, {'positive_ratings': 1, 'f_score': 0.0}, {'positive_ratings': 1, 'f_score': 0.0}, {'positive_ratings': 1, 'f_score': 0.0}, {'positive_ratings': 1, 'f_score': 0.0}, {'positive_ratings': 1, 'f_score': 0.0}, {'positive_ratings': 1, 'f_score': 0.0}, {'positive_ratings': 1, 'f_score': 0.0}, {'positive_ratings': 1, 'f_score': 0.0}, {'positive_ratings': 1, 'f_score': 0.0}, {'positive_ratings': 1, 'f_score': 0.0}, {'positive_ratings': 1, 'f_score': 0.0}]
#
# f_scores = {}
# for result in results:
#     try:
#         f_scores[result['positive_ratings']].append(result['f_score'])
#     except KeyError:
#         f_scores[result['positive_ratings']] = [result['f_score']]
#
# f_scores = {k: np.average(v) for k, v in f_scores.items()}
#
# print(np.std(list(f_scores.values())))
#
# fig, ax = plt.subplots()
# ax.bar(list(f_scores.keys()), list(f_scores.values()), width=1.0)
#
# ax.set_xlabel('Positive ratings')
# ax.set_ylabel('F-score avg')
# fig.tight_layout()
# plt.show()

#
final_model = {'[30-50]': {'alternative_parameters': [{'merge_method': 'union', 'user_profile_v': 1, 'alpha': 0.25, 'cat_score_method': 1, 'cat_key_weight': (80, 20)}, {'merge_method': 'union', 'user_profile_v': 1, 'alpha': 0.25, 'cat_score_method': 2, 'cat_key_weight': (70, 30)}, {'merge_method': 'union', 'user_profile_v': 1, 'alpha': 0.25, 'cat_score_method': 2, 'cat_key_weight': (80, 20)}], 'f_score': 0.6829640947288006, 'parameters': {'merge_method': 'union', 'user_profile_v': 1, 'alpha': 0.25, 'cat_score_method': 1, 'cat_key_weight': (70, 30)}}, '[20-30]': {'alternative_parameters': [{'merge_method': 'union', 'user_profile_v': 1, 'alpha': 0, 'cat_score_method': 1, 'cat_key_weight': (70, 30)}, {'merge_method': 'union', 'user_profile_v': 1, 'alpha': 0, 'cat_score_method': 1, 'cat_key_weight': (80, 20)}, {'merge_method': 'union', 'user_profile_v': 1, 'alpha': 0, 'cat_score_method': 2, 'cat_key_weight': (60, 40)}, {'merge_method': 'union', 'user_profile_v': 1, 'alpha': 0, 'cat_score_method': 2, 'cat_key_weight': (70, 30)}, {'merge_method': 'union', 'user_profile_v': 1, 'alpha': 0, 'cat_score_method': 2, 'cat_key_weight': (80, 20)}], 'f_score': 0.51587301587301593, 'parameters': {'merge_method': 'union', 'user_profile_v': 1, 'alpha': 0, 'cat_score_method': 1, 'cat_key_weight': (60, 40)}}, '[5-10]': {'alternative_parameters': [{'merge_method': 'union', 'user_profile_v': 1, 'alpha': 0, 'cat_score_method': 1, 'cat_key_weight': (70, 30)}, {'merge_method': 'union', 'user_profile_v': 1, 'alpha': 0, 'cat_score_method': 1, 'cat_key_weight': (80, 20)}, {'merge_method': 'union', 'user_profile_v': 1, 'alpha': 0, 'cat_score_method': 2, 'cat_key_weight': (60, 40)}, {'merge_method': 'union', 'user_profile_v': 1, 'alpha': 0, 'cat_score_method': 2, 'cat_key_weight': (70, 30)}, {'merge_method': 'union', 'user_profile_v': 1, 'alpha': 0, 'cat_score_method': 2, 'cat_key_weight': (80, 20)}], 'f_score': 0.66666666666666663, 'parameters': {'merge_method': 'union', 'user_profile_v': 1, 'alpha': 0, 'cat_score_method': 1, 'cat_key_weight': (60, 40)}}, '[10-20]': {'alternative_parameters': [{'merge_method': 'intersection', 'user_profile_v': 2, 'alpha': 0, 'cat_score_method': 2, 'cat_key_weight': (80, 20)}], 'f_score': 0.43548085901027078, 'parameters': {'merge_method': 'intersection', 'user_profile_v': 2, 'alpha': 0, 'cat_score_method': 1, 'cat_key_weight': (80, 20)}}, '[100-150]': {'alternative_parameters': [{'merge_method': 'union', 'user_profile_v': 1, 'alpha': 0.5, 'cat_score_method': 1, 'cat_key_weight': (70, 30)}, {'merge_method': 'union', 'user_profile_v': 1, 'alpha': 0.5, 'cat_score_method': 1, 'cat_key_weight': (80, 20)}, {'merge_method': 'union', 'user_profile_v': 1, 'alpha': 0.5, 'cat_score_method': 2, 'cat_key_weight': (60, 40)}, {'merge_method': 'union', 'user_profile_v': 1, 'alpha': 0.5, 'cat_score_method': 2, 'cat_key_weight': (70, 30)}, {'merge_method': 'union', 'user_profile_v': 1, 'alpha': 0.5, 'cat_score_method': 2, 'cat_key_weight': (80, 20)}, {'merge_method': 'union', 'user_profile_v': 1, 'alpha': 0.75, 'cat_score_method': 1, 'cat_key_weight': (60, 40)}, {'merge_method': 'union', 'user_profile_v': 1, 'alpha': 0.75, 'cat_score_method': 1, 'cat_key_weight': (70, 30)}, {'merge_method': 'union', 'user_profile_v': 1, 'alpha': 0.75, 'cat_score_method': 1, 'cat_key_weight': (80, 20)}, {'merge_method': 'union', 'user_profile_v': 1, 'alpha': 0.75, 'cat_score_method': 2, 'cat_key_weight': (60, 40)}, {'merge_method': 'union', 'user_profile_v': 1, 'alpha': 0.75, 'cat_score_method': 2, 'cat_key_weight': (70, 30)}, {'merge_method': 'union', 'user_profile_v': 1, 'alpha': 0.75, 'cat_score_method': 2, 'cat_key_weight': (80, 20)}], 'f_score': 0.53713527851458887, 'parameters': {'merge_method': 'union', 'user_profile_v': 1, 'alpha': 0.5, 'cat_score_method': 1, 'cat_key_weight': (60, 40)}}, '[70-100]': {'alternative_parameters': [{'merge_method': 'union', 'user_profile_v': 1, 'alpha': 0.5, 'cat_score_method': 1, 'cat_key_weight': (70, 30)}, {'merge_method': 'union', 'user_profile_v': 1, 'alpha': 0.5, 'cat_score_method': 1, 'cat_key_weight': (80, 20)}, {'merge_method': 'union', 'user_profile_v': 1, 'alpha': 0.5, 'cat_score_method': 2, 'cat_key_weight': (60, 40)}, {'merge_method': 'union', 'user_profile_v': 1, 'alpha': 0.5, 'cat_score_method': 2, 'cat_key_weight': (70, 30)}, {'merge_method': 'union', 'user_profile_v': 1, 'alpha': 0.5, 'cat_score_method': 2, 'cat_key_weight': (80, 20)}], 'f_score': 0.87121212121212122, 'parameters': {'merge_method': 'union', 'user_profile_v': 1, 'alpha': 0.5, 'cat_score_method': 1, 'cat_key_weight': (60, 40)}}, '[1-5]': {'alternative_parameters': [{'merge_method': 'union', 'user_profile_v': 2, 'alpha': 0, 'cat_score_method': 2, 'cat_key_weight': (60, 40)}], 'f_score': 0.044901065449010652, 'parameters': {'merge_method': 'union', 'user_profile_v': 2, 'alpha': 0, 'cat_score_method': 1, 'cat_key_weight': (60, 40)}}, '[50-70]': {'alternative_parameters': [{'merge_method': 'union', 'user_profile_v': 1, 'alpha': 0.75, 'cat_score_method': 1, 'cat_key_weight': (60, 40)}], 'f_score': 0.63586956521739135, 'parameters': {'merge_method': 'union', 'user_profile_v': 1, 'alpha': 0.5, 'cat_score_method': 1, 'cat_key_weight': (60, 40)}}, '[150+]': {'alternative_parameters': [{'merge_method': 'indifferent', 'user_profile_v': 1, 'alpha': 0, 'cat_score_method': 1, 'cat_key_weight': (70, 30)}, {'merge_method': 'indifferent', 'user_profile_v': 1, 'alpha': 0, 'cat_score_method': 1, 'cat_key_weight': (80, 20)}, {'merge_method': 'indifferent', 'user_profile_v': 1, 'alpha': 0, 'cat_score_method': 2, 'cat_key_weight': (60, 40)}, {'merge_method': 'indifferent', 'user_profile_v': 1, 'alpha': 0, 'cat_score_method': 2, 'cat_key_weight': (70, 30)}, {'merge_method': 'indifferent', 'user_profile_v': 1, 'alpha': 0, 'cat_score_method': 2, 'cat_key_weight': (80, 20)}, {'merge_method': 'indifferent', 'user_profile_v': 2, 'alpha': 0, 'cat_score_method': 1, 'cat_key_weight': (60, 40)}, {'merge_method': 'indifferent', 'user_profile_v': 2, 'alpha': 0, 'cat_score_method': 1, 'cat_key_weight': (70, 30)}, {'merge_method': 'indifferent', 'user_profile_v': 2, 'alpha': 0, 'cat_score_method': 1, 'cat_key_weight': (80, 20)}, {'merge_method': 'indifferent', 'user_profile_v': 2, 'alpha': 0, 'cat_score_method': 2, 'cat_key_weight': (60, 40)}, {'merge_method': 'indifferent', 'user_profile_v': 2, 'alpha': 0, 'cat_score_method': 2, 'cat_key_weight': (70, 30)}, {'merge_method': 'indifferent', 'user_profile_v': 2, 'alpha': 0, 'cat_score_method': 2, 'cat_key_weight': (80, 20)}, {'merge_method': 'indifferent', 'user_profile_v': 1, 'alpha': 0.25, 'cat_score_method': 1, 'cat_key_weight': (60, 40)}, {'merge_method': 'indifferent', 'user_profile_v': 1, 'alpha': 0.25, 'cat_score_method': 1, 'cat_key_weight': (70, 30)}, {'merge_method': 'indifferent', 'user_profile_v': 1, 'alpha': 0.25, 'cat_score_method': 1, 'cat_key_weight': (80, 20)}, {'merge_method': 'indifferent', 'user_profile_v': 1, 'alpha': 0.25, 'cat_score_method': 2, 'cat_key_weight': (60, 40)}, {'merge_method': 'indifferent', 'user_profile_v': 1, 'alpha': 0.25, 'cat_score_method': 2, 'cat_key_weight': (70, 30)}, {'merge_method': 'indifferent', 'user_profile_v': 1, 'alpha': 0.25, 'cat_score_method': 2, 'cat_key_weight': (80, 20)}, {'merge_method': 'indifferent', 'user_profile_v': 2, 'alpha': 0.25, 'cat_score_method': 1, 'cat_key_weight': (60, 40)}, {'merge_method': 'indifferent', 'user_profile_v': 2, 'alpha': 0.25, 'cat_score_method': 1, 'cat_key_weight': (70, 30)}, {'merge_method': 'indifferent', 'user_profile_v': 2, 'alpha': 0.25, 'cat_score_method': 1, 'cat_key_weight': (80, 20)}, {'merge_method': 'indifferent', 'user_profile_v': 2, 'alpha': 0.25, 'cat_score_method': 2, 'cat_key_weight': (60, 40)}, {'merge_method': 'indifferent', 'user_profile_v': 2, 'alpha': 0.25, 'cat_score_method': 2, 'cat_key_weight': (70, 30)}, {'merge_method': 'indifferent', 'user_profile_v': 2, 'alpha': 0.25, 'cat_score_method': 2, 'cat_key_weight': (80, 20)}, {'merge_method': 'indifferent', 'user_profile_v': 1, 'alpha': 0.5, 'cat_score_method': 1, 'cat_key_weight': (60, 40)}, {'merge_method': 'indifferent', 'user_profile_v': 1, 'alpha': 0.5, 'cat_score_method': 1, 'cat_key_weight': (70, 30)}, {'merge_method': 'indifferent', 'user_profile_v': 1, 'alpha': 0.5, 'cat_score_method': 1, 'cat_key_weight': (80, 20)}, {'merge_method': 'indifferent', 'user_profile_v': 1, 'alpha': 0.5, 'cat_score_method': 2, 'cat_key_weight': (60, 40)}, {'merge_method': 'indifferent', 'user_profile_v': 1, 'alpha': 0.5, 'cat_score_method': 2, 'cat_key_weight': (70, 30)}, {'merge_method': 'indifferent', 'user_profile_v': 1, 'alpha': 0.5, 'cat_score_method': 2, 'cat_key_weight': (80, 20)}, {'merge_method': 'indifferent', 'user_profile_v': 2, 'alpha': 0.5, 'cat_score_method': 1, 'cat_key_weight': (60, 40)}, {'merge_method': 'indifferent', 'user_profile_v': 2, 'alpha': 0.5, 'cat_score_method': 1, 'cat_key_weight': (70, 30)}, {'merge_method': 'indifferent', 'user_profile_v': 2, 'alpha': 0.5, 'cat_score_method': 1, 'cat_key_weight': (80, 20)}, {'merge_method': 'indifferent', 'user_profile_v': 2, 'alpha': 0.5, 'cat_score_method': 2, 'cat_key_weight': (60, 40)}, {'merge_method': 'indifferent', 'user_profile_v': 2, 'alpha': 0.5, 'cat_score_method': 2, 'cat_key_weight': (70, 30)}, {'merge_method': 'indifferent', 'user_profile_v': 2, 'alpha': 0.5, 'cat_score_method': 2, 'cat_key_weight': (80, 20)}, {'merge_method': 'indifferent', 'user_profile_v': 1, 'alpha': 0.75, 'cat_score_method': 1, 'cat_key_weight': (60, 40)}, {'merge_method': 'indifferent', 'user_profile_v': 1, 'alpha': 0.75, 'cat_score_method': 1, 'cat_key_weight': (70, 30)}, {'merge_method': 'indifferent', 'user_profile_v': 1, 'alpha': 0.75, 'cat_score_method': 1, 'cat_key_weight': (80, 20)}, {'merge_method': 'indifferent', 'user_profile_v': 1, 'alpha': 0.75, 'cat_score_method': 2, 'cat_key_weight': (60, 40)}, {'merge_method': 'indifferent', 'user_profile_v': 1, 'alpha': 0.75, 'cat_score_method': 2, 'cat_key_weight': (70, 30)}, {'merge_method': 'indifferent', 'user_profile_v': 1, 'alpha': 0.75, 'cat_score_method': 2, 'cat_key_weight': (80, 20)}, {'merge_method': 'indifferent', 'user_profile_v': 2, 'alpha': 0.75, 'cat_score_method': 1, 'cat_key_weight': (60, 40)}, {'merge_method': 'indifferent', 'user_profile_v': 2, 'alpha': 0.75, 'cat_score_method': 1, 'cat_key_weight': (70, 30)}, {'merge_method': 'indifferent', 'user_profile_v': 2, 'alpha': 0.75, 'cat_score_method': 1, 'cat_key_weight': (80, 20)}, {'merge_method': 'indifferent', 'user_profile_v': 2, 'alpha': 0.75, 'cat_score_method': 2, 'cat_key_weight': (60, 40)}, {'merge_method': 'indifferent', 'user_profile_v': 2, 'alpha': 0.75, 'cat_score_method': 2, 'cat_key_weight': (70, 30)}, {'merge_method': 'indifferent', 'user_profile_v': 2, 'alpha': 0.75, 'cat_score_method': 2, 'cat_key_weight': (80, 20)}, {'merge_method': 'indifferent', 'user_profile_v': 1, 'alpha': 1.0, 'cat_score_method': 1, 'cat_key_weight': (60, 40)}, {'merge_method': 'indifferent', 'user_profile_v': 1, 'alpha': 1.0, 'cat_score_method': 1, 'cat_key_weight': (70, 30)}, {'merge_method': 'indifferent', 'user_profile_v': 1, 'alpha': 1.0, 'cat_score_method': 1, 'cat_key_weight': (80, 20)}, {'merge_method': 'indifferent', 'user_profile_v': 1, 'alpha': 1.0, 'cat_score_method': 2, 'cat_key_weight': (60, 40)}, {'merge_method': 'indifferent', 'user_profile_v': 1, 'alpha': 1.0, 'cat_score_method': 2, 'cat_key_weight': (70, 30)}, {'merge_method': 'indifferent', 'user_profile_v': 1, 'alpha': 1.0, 'cat_score_method': 2, 'cat_key_weight': (80, 20)}, {'merge_method': 'indifferent', 'user_profile_v': 2, 'alpha': 1.0, 'cat_score_method': 1, 'cat_key_weight': (60, 40)}, {'merge_method': 'indifferent', 'user_profile_v': 2, 'alpha': 1.0, 'cat_score_method': 1, 'cat_key_weight': (70, 30)}, {'merge_method': 'indifferent', 'user_profile_v': 2, 'alpha': 1.0, 'cat_score_method': 1, 'cat_key_weight': (80, 20)}, {'merge_method': 'indifferent', 'user_profile_v': 2, 'alpha': 1.0, 'cat_score_method': 2, 'cat_key_weight': (60, 40)}, {'merge_method': 'indifferent', 'user_profile_v': 2, 'alpha': 1.0, 'cat_score_method': 2, 'cat_key_weight': (70, 30)}, {'merge_method': 'indifferent', 'user_profile_v': 2, 'alpha': 1.0, 'cat_score_method': 2, 'cat_key_weight': (80, 20)}], 'f_score': 0, 'parameters': {'merge_method': 'indifferent', 'user_profile_v': 1, 'alpha': 0, 'cat_score_method': 1, 'cat_key_weight': (60, 40)}}}
#
for bin, model in final_model.items():
    print(bin, model['f_score'])
    print(model['parameters'])

    for param in model['alternative_parameters']:
        print(param)
    print("============")

# cf_model = {'[10-20]': {'f_score': 0.42194211017740429, 'parameters': {'alpha': 0}}, '[1-5]': {'f_score': 0.041095890410958902, 'parameters': {'alpha': 0}}, '[70-100]': {'f_score': 0.87121212121212122, 'parameters': {'alpha': 0.5}}, '[5-10]': {'f_score': 0.66666666666666663, 'parameters': {'alpha': 0}}, '[20-30]': {'f_score': 0.51587301587301593, 'parameters': {'alpha': 0}}, '[30-50]': {'f_score': 0.6829640947288006, 'parameters': {'alpha': 0.25}}, '[150+]': {'f_score': 0, 'parameters': {'alpha': 0}}, '[100-150]': {'f_score': 0.53713527851458887, 'parameters': {'alpha': 0.5}}, '[50-70]': {'f_score': 0.59868421052631582, 'parameters': {'alpha': 0.5}}}
# for bin, model in cf_model.items():
#     print("%s -> CF: %s , Hybrid: %s" % (bin, model['f_score'], final_model[bin]['f_score']))
#     print("alpha -> CF: %s , Hybrid: %s" % (model['parameters']['alpha'], final_model[bin]['parameters']['alpha']))


# cb_model = {'[50-70]': {'f_score': 0.22846215780998388, 'parameters': {'cat_key_weight': (80, 20), 'user_profile_v': 2, 'cat_score_method': 1}, 'alternative_parameters': []}, '[10-20]': {'f_score': 0.12817985837877144, 'parameters': {'cat_key_weight': (80, 20), 'user_profile_v': 2, 'cat_score_method': 1}, 'alternative_parameters': [{'user_profile_v': 1, 'cat_key_weight': (80, 20), 'cat_score_method': 1}, {'user_profile_v': 1, 'cat_key_weight': (60, 40), 'cat_score_method': 2}, {'user_profile_v': 1, 'cat_key_weight': (80, 20), 'cat_score_method': 2}, {'user_profile_v': 2, 'cat_key_weight': (80, 20), 'cat_score_method': 2}]}, '[5-10]': {'f_score': 0.08382066276803118, 'parameters': {'cat_key_weight': (80, 20), 'user_profile_v': 2, 'cat_score_method': 1}, 'alternative_parameters': [{'user_profile_v': 1, 'cat_key_weight': (80, 20), 'cat_score_method': 1}, {'user_profile_v': 1, 'cat_key_weight': (60, 40), 'cat_score_method': 2}, {'user_profile_v': 1, 'cat_key_weight': (80, 20), 'cat_score_method': 2}, {'user_profile_v': 2, 'cat_key_weight': (60, 40), 'cat_score_method': 1}, {'user_profile_v': 2, 'cat_key_weight': (80, 20), 'cat_score_method': 2}]}, '[30-50]': {'f_score': 0.183889174586849, 'parameters': {'cat_key_weight': (80, 20), 'user_profile_v': 2, 'cat_score_method': 1}, 'alternative_parameters': [{'user_profile_v': 1, 'cat_key_weight': (80, 20), 'cat_score_method': 1}, {'user_profile_v': 1, 'cat_key_weight': (60, 40), 'cat_score_method': 2}, {'user_profile_v': 1, 'cat_key_weight': (80, 20), 'cat_score_method': 2}, {'user_profile_v': 2, 'cat_key_weight': (80, 20), 'cat_score_method': 2}]}, '[1-5]': {'f_score': 0.00929549902152642, 'parameters': {'cat_key_weight': (80, 20), 'user_profile_v': 2, 'cat_score_method': 1}, 'alternative_parameters': [{'user_profile_v': 1, 'cat_key_weight': (80, 20), 'cat_score_method': 1}, {'user_profile_v': 1, 'cat_key_weight': (60, 40), 'cat_score_method': 2}, {'user_profile_v': 1, 'cat_key_weight': (80, 20), 'cat_score_method': 2}, {'user_profile_v': 2, 'cat_key_weight': (80, 20), 'cat_score_method': 2}]}, '[150+]': {'f_score': 0, 'parameters': {'cat_key_weight': (60, 40), 'user_profile_v': 1, 'cat_score_method': 1}, 'alternative_parameters': [{'user_profile_v': 1, 'cat_key_weight': (80, 20), 'cat_score_method': 1}, {'user_profile_v': 1, 'cat_key_weight': (60, 40), 'cat_score_method': 2}, {'user_profile_v': 1, 'cat_key_weight': (80, 20), 'cat_score_method': 2}, {'user_profile_v': 2, 'cat_key_weight': (60, 40), 'cat_score_method': 1}, {'user_profile_v': 2, 'cat_key_weight': (80, 20), 'cat_score_method': 1}, {'user_profile_v': 2, 'cat_key_weight': (60, 40), 'cat_score_method': 2}, {'user_profile_v': 2, 'cat_key_weight': (80, 20), 'cat_score_method': 2}]}, '[20-30]': {'f_score': 0.2158119658119658, 'parameters': {'cat_key_weight': (80, 20), 'user_profile_v': 2, 'cat_score_method': 1}, 'alternative_parameters': [{'user_profile_v': 1, 'cat_key_weight': (80, 20), 'cat_score_method': 1}, {'user_profile_v': 1, 'cat_key_weight': (60, 40), 'cat_score_method': 2}, {'user_profile_v': 1, 'cat_key_weight': (80, 20), 'cat_score_method': 2}, {'user_profile_v': 2, 'cat_key_weight': (80, 20), 'cat_score_method': 2}]}, '[70-100]': {'f_score': 0.2337288745055735, 'parameters': {'cat_key_weight': (80, 20), 'user_profile_v': 2, 'cat_score_method': 2}, 'alternative_parameters': [{'user_profile_v': 1, 'cat_key_weight': (80, 20), 'cat_score_method': 1}, {'user_profile_v': 1, 'cat_key_weight': (60, 40), 'cat_score_method': 2}, {'user_profile_v': 1, 'cat_key_weight': (80, 20), 'cat_score_method': 2}]}, '[100-150]': {'f_score': 0.03424657534246575, 'parameters': {'cat_key_weight': (60, 40), 'user_profile_v': 2, 'cat_score_method': 1}, 'alternative_parameters': [{'user_profile_v': 1, 'cat_key_weight': (80, 20), 'cat_score_method': 1}, {'user_profile_v': 1, 'cat_key_weight': (60, 40), 'cat_score_method': 2}, {'user_profile_v': 1, 'cat_key_weight': (80, 20), 'cat_score_method': 2}, {'user_profile_v': 2, 'cat_key_weight': (60, 40), 'cat_score_method': 2}]}}
# for bin, model in cb_model.items():
#     print(bin, model['f_score'])
#     print(model['parameters'])
#
#     for param in model['alternative_parameters']:
#         print(param)
#     print("============")

# from pymongo import MongoClient
#
#
# client = MongoClient('localhost', 28017)
# hubchat = client.hubchat
#
# users_training = list(hubchat.ratings_training.aggregate([
#     {
#         "$group": {
#             "_id": "$user"
#         }
#     }
# ]))
#
# print("users in training set: %s" % len(users_training))
#
#
# users_validate = list(hubchat.ratings_validate.aggregate([
#     {
#         "$group": {
#             "_id": "$user"
#         }
#     }
# ]))
#
# print("users in validate set: %s" % len(users_validate))
#
#
# users_testing = list(hubchat.ratings_testing.aggregate([
#     {
#         "$group": {
#             "_id": "$user"
#         }
#     }
# ]))
#
# print("users in testing set: %s" % len(users_testing))
#
# print("each user:")
# for user in hubchat.ratings.aggregate([
#     {
#         "$match": {
#             "rate": {"$gt": 1}
#         }
#     },
#     {
#         "$group": {
#             "_id": "$user",
#             "count": {"$sum": 1}
#         }
#     }
# ]):
#     user_training = list(hubchat.ratings_training.aggregate([
#         {
#             "$match": {
#                 "user": user['_id']
#             }
#         },
#         {
#             "$group": {
#                 "_id": "$user",
#                 "count": {"$sum": 1}
#             }
#         }
#     ]))
#
#     user_training_n = user_training[0]['count'] if user_training else 0
#
#     user_validate = list(hubchat.ratings_validate.aggregate([
#         {
#             "$match": {
#                 "user": user['_id']
#             }
#         },
#         {
#             "$group": {
#                 "_id": "$user",
#                 "count": {"$sum": 1}
#             }
#         }
#     ]))
#
#     user_validate_n = user_validate[0]['count'] if user_validate else 0
#
#     user_testing = list(hubchat.ratings_testing.aggregate([
#         {
#             "$match": {
#                 "user": user['_id']
#             }
#         },
#         {
#             "$group": {
#                 "_id": "$user",
#                 "count": {"$sum": 1}
#             }
#         }
#     ]))
#
#     user_testing_n = user_testing[0]['count'] if user_testing else 0
#
#     print(user_training_n, user_validate_n, user_testing_n)