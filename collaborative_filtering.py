from pymongo import MongoClient
from bson.objectid import ObjectId
from scipy.linalg import norm
import numpy as np


client = MongoClient('localhost', 28017)
hubchat = client.hubchat


def split_ratings_training_testing():

    hubchat.ratings_training.delete_many({})
    hubchat.ratings_testing.delete_many({})

    for user in hubchat.ratings.aggregate([
        {
            "$group": {
                "_id": "$user",
                "count": {"$sum": 1}
            }
        }
    ]):
        ratings = list(hubchat.ratings.find({'user': user['_id']}).sort("createdAt"))

        training_len = round(len(ratings) * 90 / 100)
        training_set = ratings[:training_len]
        testing_set = ratings[training_len:]

        for rate in training_set:
            hubchat.ratings_training.insert_one(
                {
                    "user": rate['user'],
                    "post": rate['post'],
                    "rate": rate['rate'],
                    "createdAt": rate['createdAt']
                }
            )

        for rate in testing_set:
            hubchat.ratings_testing.insert_one(
                {
                    "user": rate['user'],
                    "post": rate['post'],
                    "rate": rate['rate'],
                    "createdAt": rate['createdAt']
                }
            )


def split_ratings_training_validate():

    hubchat.ratings_validate.delete_many({})

    for user in hubchat.ratings_training.aggregate([
        {
            "$group": {
                "_id": "$user",
                "count": {"$sum": 1}
            }
        }
    ]):

        ratings = list(hubchat.ratings_training.find({'user': user['_id']}).sort("createdAt"))

        training_len = round(len(ratings) * 90 / 100)
        validate_set = ratings[training_len:]

        for rate in validate_set:
            hubchat.ratings_validate.insert_one(
                {
                    "user": rate['user'],
                    "post": rate['post'],
                    "rate": rate['rate'],
                    "createdAt": rate['createdAt']
                }
            )

            hubchat.ratings_training.delete_many(
                {
                    "user": rate['user'],
                    "post": rate['post']
                }
            )


def merge_ratings_training_validate():
    for rate in hubchat.ratings_validate.find({}):
        hubchat.ratings_training.insert_one(
            {
                "user": rate['user'],
                "post": rate['post'],
                "rate": rate['rate'],
                "createdAt": rate['createdAt']
            }
        )


def get_rated_posts_except_user(user_id):

    for post in hubchat.ratings_training.aggregate([
        {
            "$match": {
                "rate": {
                    "$gt": 1
                },
                "user": {"$ne": ObjectId(user_id)}
            }
        },
        {
            "$group": {
                "_id": "$post",
                "users": {"$push": {"user": "$user", "rate": "$rate"}}
            }
        }
    ]):
        yield {'_id': post['_id'], 'rates': {u['user']: u['rate'] for u in post['users']}}


def get_rated_posts(phase):

    if phase == 1:
        for post in hubchat.ratings_training.aggregate([
            {
                "$match": {
                    "rate": {
                        "$gt": 1
                    }
                }
            },
            {
                "$group": {
                    "_id": "$post",
                    "users": {"$push": {"user": "$user", "rate": "$rate"}}
                }
            }
        ]):
            yield {'_id': post['_id'], 'rates': {u['user']: u['rate'] for u in post['users']}}

    elif phase == 2:
        for post in hubchat.ratings.aggregate([
            {
                "$match": {
                    "rate": {
                        "$gt": 1
                    }
                }
            },
            {
                "$group": {
                    "_id": "$post",
                    "users": {"$push": {"user": "$user", "rate": "$rate"}}
                }
            }
        ]):
            yield {'_id': post['_id'], 'rates': {u['user']: u['rate'] for u in post['users']}}


def get_cosine_similarity(p1, p2):

    if len(p2) < len(p1):
        p1, p2 = p2, p1

    res = 0
    for key, p1_value in p1.items():
        res += p1_value * p2.get(key, 0)
    if res == 0:
        return 0

    try:
        res = res / norm(list(p1.values())) / norm(list(p2.values()))
    except ZeroDivisionError:
        res = 0
    return res


def update_similarities(phase):

    # Delete all documents
    hubchat.postsimilarity.delete_many({})

    posts = list(get_rated_posts(phase))
    posts2 = posts.copy()

    skip = 0
    analysed = set()

    for post1 in posts:
        skip += 1
        for post2 in posts2[skip:]:

            # Post 1 to be the smaller
            post1_id, post2_id = sorted([str(post1['_id']), str(post2['_id'])])

            # Skip if they are the same
            if post1_id == post2_id:
                continue

            key = ''.join([post1_id, post2_id])
            if key in analysed:
                continue

            analysed.add(key)

            # Get similarity
            sim = get_cosine_similarity(post1['rates'], post2['rates'])

            if sim > 0:
                # Insert on db
                hubchat.postsimilarity.insert_one(
                    {
                        "post1": ObjectId(post1_id),
                        "post2": ObjectId(post2_id),
                        "similarity": sim
                    }
                )


def get_users_with_positive_ratings():
    return hubchat.ratings_training.aggregate([
        {
            "$match": {
                "rate": {
                    "$gt": 1
                }
            }
        },
        {
            "$group": {
                "_id": "$user"
            }
        }
    ])


def update_similarities_per_user():

    # Delete all documents
    hubchat.postsimilarity.delete_many({})

    for user in get_users_with_positive_ratings():
        user_id = user['_id']

        posts = list(get_rated_posts_except_user(user_id))
        posts2 = posts.copy()

        skip = 0
        analysed = set()

        for post1 in posts:
            skip += 1
            for post2 in posts2[skip:]:

                # Post 1 to be the smaller
                post1_id, post2_id = sorted([str(post1['_id']), str(post2['_id'])])

                # Skip if they are the same
                if post1_id == post2_id:
                    continue

                key = ''.join([post1_id, post2_id])
                if key in analysed:
                    continue

                analysed.add(key)

                # Get similarity
                sim = get_cosine_similarity(post1['rates'], post2['rates'])

                if sim > 0:
                    # Insert on db
                    hubchat.postsimilarity.insert_one(
                        {
                            "user": user_id,
                            "post1": ObjectId(post1_id),
                            "post2": ObjectId(post2_id),
                            "similarity": sim
                        }
                    )


def get_similar_posts(post_id, threshold):

    sim_posts = []

    for post in hubchat.postsimilarity.find({
        "$or": [
            {"post1": ObjectId(post_id)},
            {"post2": ObjectId(post_id)}
        ],
        "similarity": {
            "$gte": threshold
        }
    }):
        sim_post = str(post['post2']) if str(post['post1']) == str(post_id) else str(post['post1'])
        sim_posts.append(sim_post)

    return sim_posts


def get_high_rated_posts(user_id):

    for rate in hubchat.ratings_training.find({
        "user": ObjectId(user_id),
        "rate": {"$gte": 3}
    }):
        yield str(rate['post'])


def get_recommendations(user_id, alpha):

    threshold = define_threshold(alpha)

    # Get posts similar to high rated posts
    similars = set()
    for post_id in get_high_rated_posts(user_id):
        similars |= set(get_similar_posts(post_id, threshold))

    # Remove posts already seen
    for rate in hubchat.ratings_training.find({"user": ObjectId(user_id)}):
        post_id = str(rate["post"])
        similars.discard(post_id)

    return similars


def get_recommendations_all(user_id, alpha):

    threshold = define_threshold(alpha)

    # Get posts similar to high rated posts
    similars = set()
    for post_id in get_high_rated_posts(user_id):
        similars |= set(get_similar_posts(post_id, threshold))

    # Remove posts already seen
    for rate in hubchat.ratings.find({"user": ObjectId(user_id)}):
        post_id = str(rate["post"])
        similars.discard(post_id)

    return similars

def define_threshold(alpha):

    similarities = []

    for post in hubchat.postsimilarity.find({}):
        similarities.append(float(post['similarity']))

    np_arr = np.array(similarities)

    return np.average(np_arr) + (alpha * np.std(np_arr))


def get_confusion_matrix(user_id, recommendation_list, phase):
    """
    Build confusion matrix for user
    :param user_id:
    :param recommendation_list:
    :param phase:
    :return:
    """
    tp = 0  # True positive
    fp = 0  # False positive
    fn = 0  # False negative
    tn = 0  # True negative

    if recommendation_list is None:
        recommendation_list = set()

    for post_id in recommendation_list:

        query = {
            "user": ObjectId(user_id),
            "post": ObjectId(post_id)
        }

        if phase == 1:
            rate = hubchat.ratings_validate.find_one(query)
        elif phase == 2:
            rate = hubchat.ratings_testing.find_one(query)

        if rate is None:
            continue  # Ignore

        if 3 <= rate['rate'] <= 4:
            tp += 1  # In recommendation list and in testing set with rate >= 3
        elif 1 <= rate['rate'] <= 2:
            fp += 1  # In recommendation list and in testing set with rate < 3

    if phase == 1:
        testing_rates = hubchat.ratings_validate.find({"user": ObjectId(user_id)})
    elif phase == 2:
        testing_rates = hubchat.ratings_testing.find({"user": ObjectId(user_id)})

    for rate in testing_rates:

        if str(rate['post']) in recommendation_list:
            continue  # Already analysed

        if 3 <= rate['rate'] <= 4:
            fn += 1  # In testing set with rate >= 3 but not in recommendation list
        elif 1 <= rate['rate'] <= 2:
            tn += 1  # In testing set with rate < 3 and not in recommendation list

    return tp, fp, fn, tn

#
# split_ratings_training_testing()
#
# correct_rates = []
# # correct_recommendations = {}
# for user in hubchat.ratings_training.aggregate(users_with_positive_ratings_order_by_count):
#     user_id = str(user['_id'])
#     recommendation_list = get_recommendations(user_id)
#
#     tp = 0  # True positive
#     fp = 0  # False positive
#
#     for post_id in recommendation_list:
#         rate = hubchat.ratings_testing.find_one({
#             "user": ObjectId(user_id),
#             "post": ObjectId(post_id)
#         })
#
#         if rate is None:
#             # Ignore
#             continue
#
#         if 3 <= rate['rate'] <= 4:
#             tp += 1
#         elif 1 <= rate['rate'] <= 2:
#             fp += 1
#
#     fn = 0  # False negative
#     tn = 0  # True negative
#     # try:
    #     correct_rate = count_correct / float(count)
    # except ZeroDivisionError:
    #     correct_rate = None
    #
    # if correct_rate is not None:
    #     correct_rates.append(correct_rate)
    #
    # try:
    #     correct_recommendations[user['count']].append(count_correct)
    # except KeyError:
    #     correct_recommendations[user['count']] = [count_correct]
    #
    # print("Positive ratings: %s , Recommendations unknown tried: %s , Recommendations known tried: %s , Correct: %s , correct rate: %s" % (user['count'], count_1, count, count_correct, correct_rate))

# print("Correct rates avg with a=0.75 -> threshold=0.857705843546: ", np.average(correct_rates))

# correct_recommendations = {k: np.average(v) for k, v in correct_recommendations.items()}
# print(correct_recommendations)
#
# import matplotlib.pyplot as plt
#
# fig, ax = plt.subplots()
# ax.bar(list(correct_recommendations.keys()), list(correct_recommendations.values()), width=1.0)
#
# ax.set_xlabel('Positive ratings')
# ax.set_ylabel('Correct recommendations')
# fig.tight_layout()
# plt.savefig("figures/cf-threshold-926009457076.pdf", format='pdf')