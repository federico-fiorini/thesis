#!.env/bin/python3

from collaborative_filtering import *
from utils import f_score, BinnedUsers


def get_f_score(tp, fp, fn, tn):
    try:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
    except ZeroDivisionError:
        precision = 0.0
        recall = 0.0

    return f_score(precision, recall)


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

# split_ratings_training_testing()
#
# # PHASE 1
# update_similarities(phase=1)
#
# split_ratings_training_validate()


# Bin users
binned_users = BinnedUsers()
for user in hubchat.ratings_training.aggregate(users_with_positive_ratings_order_by_count):
    positive_ratings = user['count']
    binned_users.add(positive_ratings, user["_id"])

alphas = [0, 0.25, 0.5, 0.75, 1.0]

model = {
    '[1-5]': {
        'f_score': -1,
        'parameters': {
            'alpha': None
        }
    },
    '[5-10]': {
        'f_score': -1,
        'parameters': {
            'alpha': None
        }
    },
    '[10-20]': {
        'f_score': -1,
        'parameters': {
            'alpha': None
        }
    },
    '[20-30]': {
        'f_score': -1,
        'parameters': {
            'alpha': None
        }
    },
    '[30-50]': {
        'f_score': -1,
        'parameters': {
            'alpha': None
        }
    },
    '[50-70]': {
        'f_score': -1,
        'parameters': {
            'alpha': None
        }
    },
    '[70-100]': {
        'f_score': -1,
        'parameters': {
            'alpha': None
        }
    },
    '[100-150]': {
        'f_score': -1,
        'parameters': {
            'alpha': None
        }
    },
    '[150+]': {
        'f_score': -1,
        'parameters': {
            'alpha': None
        }
    }
}

for alpha in alphas:

    for bin, users in binned_users.user_bins.items():

        f_scores = []

        for user in users:

            user_id = str(user)

            recommendation_list = get_recommendations(user_id, alpha)

            tp, fp, fn, tn = get_confusion_matrix(user_id, recommendation_list, phase=1)
            f_score_value = get_f_score(tp, fp, fn, tn)

            f_scores.append(f_score_value)

        f_score_avg = np.average(f_scores) if f_scores else 0

        if f_score_avg > model[bin]['f_score']:
            model[bin]['f_score'] = f_score_avg
            model[bin]['parameters']['alpha'] = alpha

        print("%s[alpha=%s] F-score: %s" % (bin, alpha, f_score_avg))

print(model)
# PHASE 2
exit()
#
# chosen_alpha = 0.5
# update_similarities(phase=2)
#
# merge_ratings_training_validate()
#
# true_positive = 0.0
# false_positive = 0.0
# false_negative = 0.0
# true_negative = 0.0
#
# for user in hubchat.ratings_training.aggregate(users_with_positive_ratings_order_by_count):
#     user_id = str(user['_id'])
#     recommendation_list = get_recommendations(user_id, chosen_alpha)
#
#     tp, fp, fn, tn = get_confusion_matrix(user_id, recommendation_list, phase=2)
#
#     true_positive += tp
#     false_positive += fp
#     false_negative += fn
#     true_negative += tn
#
# # Calculate total performances metrics
# precision = true_positive / (true_positive + false_positive)
# recall = true_positive / (true_positive + false_negative)
# accuracy = (true_positive + true_negative) / (true_positive + false_positive + false_negative + true_negative)
#
# print("[Testing][Alpha=%s] Precision: %s , Recall: %s , Accuracy: %s" % (chosen_alpha, precision, recall, accuracy))