#!.env/bin/python3

from collaborative_filtering import *


users_with_positive_ratings_order_by_count = [
    {
        "$match": {
            "rate": {"$gt": 1}
        }
    },
    {
        "$group": {
            "_id": "$user",
            "count": {"$sum": 1}
        }
    },
    {
        "$sort": {
            "count": -1
        }
    }
]

split_ratings_training_testing()

# PHASE 1
update_similarities(phase=1)

split_ratings_training_validate()

# alphas = [0, 0.25, 0.5, 0.75, 1.0]
# chosen_alpha = None
# max_precision = 0
#
# for alpha in alphas:
#
#     true_positive = 0.0
#     false_positive = 0.0
#     false_negative = 0.0
#     true_negative = 0.0
#
#     recommendation_rate = {}  # How many recommendations over how many positive rates
#
#     for user in hubchat.ratings_training.aggregate(users_with_positive_ratings_order_by_count):
#
#         user_id = str(user['_id'])
#         recommendation_list = get_recommendations(user_id, alpha)
#
#         tp, fp, fn, tn = get_confusion_matrix(user_id, get_recommendations, phase=2)
#
#         true_positive += tp
#         false_positive += fp
#         false_negative += fn
#         true_negative += tn
#
#         positive_ratings = user['count']
#         if positive_ratings <= 1000:
#             try:
#                 recommendation_rate[positive_ratings].append(tp)
#             except KeyError:
#                 recommendation_rate[positive_ratings] = [tp]
#
#     # Calculate total performances metrics
#     precision = true_positive / (true_positive + false_positive)
#     recall = true_positive / (true_positive + false_negative)
#     accuracy = (true_positive + true_negative) / (true_positive + false_positive + false_negative + true_negative)
#
#     if precision > max_precision:
#         max_precision = precision
#         chosen_alpha = alpha
#
#     # Average recommendations per positive ratings
#     recommendation_rate = {k: np.average(v) for k, v in recommendation_rate.items()}
#     print(sorted(recommendation_rate.items(), key=lambda t: t[0]))
#
#     print("[Validation][Alpha=%s] Precision: %s , Recall: %s , Accuracy: %s" % (alpha, precision, recall, accuracy))
#     print("Correct recommendations: %s" % true_positive)

# PHASE 2

chosen_alpha = 0.5
update_similarities(phase=2)

merge_ratings_training_validate()

true_positive = 0.0
false_positive = 0.0
false_negative = 0.0
true_negative = 0.0

for user in hubchat.ratings_training.aggregate(users_with_positive_ratings_order_by_count):
    user_id = str(user['_id'])
    recommendation_list = get_recommendations(user_id, chosen_alpha)

    tp, fp, fn, tn = get_confusion_matrix(user_id, recommendation_list, phase=2)

    true_positive += tp
    false_positive += fp
    false_negative += fn
    true_negative += tn

# Calculate total performances metrics
precision = true_positive / (true_positive + false_positive)
recall = true_positive / (true_positive + false_negative)
accuracy = (true_positive + true_negative) / (true_positive + false_positive + false_negative + true_negative)

print("[Testing][Alpha=%s] Precision: %s , Recall: %s , Accuracy: %s" % (chosen_alpha, precision, recall, accuracy))