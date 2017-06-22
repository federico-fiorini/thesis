#!.env/bin/python3

from pymongo import MongoClient, ASCENDING
from bson.objectid import ObjectId
from scipy.linalg import norm
import numpy as np
import random


client = MongoClient('localhost', 28017)
hubchat = client.hubchat


def split_ratings_training_testing():

    ratings = list(hubchat.ratings.find({}).sort("createdAt"))

    training_len = int(len(ratings) * 80 / 100)
    training_set = ratings[:training_len]
    testing_set = ratings[training_len:]

    hubchat.ratings_training.delete_many({})
    for rate in training_set:
        hubchat.ratings_training.insert_one(
            {
                "user": rate['user'],
                "post": rate['post'],
                "rate": rate['rate']
            }
        )

    hubchat.ratings_testing.delete_many({})
    for rate in testing_set:
        hubchat.ratings_testing.insert_one(
            {
                "user": rate['user'],
                "post": rate['post'],
                "rate": rate['rate']
            }
        )


def get_rated_posts():
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


def update_similarities():

    # Delete all documents
    hubchat.postsimilarity.delete_many({})

    posts = list(get_rated_posts())
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


def get_similar_posts(post_id):

    threshold = 0.857705843546
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


def get_recommendations(user_id):

    # Get posts similar to high rated posts
    similars = set()
    for post_id in get_high_rated_posts(user_id):
        similars |= set(get_similar_posts(post_id))

    # Remove posts already seen
    for rate in hubchat.ratings_training.find({"user": ObjectId(user_id)}):
        post_id = str(rate["post"])
        similars.discard(post_id)

    return similars






# similarities = []
#
# for post in hubchat.postsimilarityv2.find({}):
#     similarities.append(float(post['similarity']))
#
# a = 1.0
# np_arr = np.array(similarities)
# avg = np.average(np_arr)
# std = np.std(np_arr)
# threshold = np.average(np_arr) + (a * np.std(np_arr))
#
# print("similarity avg: ", avg)
# print("similarity standard deviation: ", std)
# print("threshold with a=%s : " % a, threshold)
#
# exit()


# update_similarities()

split_ratings_training_testing()

correct_rates = []
correct_recommendations = {}
for user in hubchat.ratings_training.aggregate([
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
]):
    user_id = str(user['_id'])
    similar_posts = get_recommendations(user_id)

    # if len(similar_posts) == 0:
    #     print("No similar remained")
    #     continue

    count = 0
    count_1 = 0
    count_correct = 0
    for post_id in similar_posts:
        rate = hubchat.ratings_testing.find_one({
            "user": ObjectId(user_id),
            "post": ObjectId(post_id)
        })

        count_1 += 1
        if rate is None:
            # print("Rate not found: post not seen")
            continue

        count += 1
        if 3 <= rate['rate'] <= 4:
            count_correct += 1

    try:
        correct_rate = count_correct / float(count)
    except ZeroDivisionError:
        correct_rate = None

    if correct_rate is not None:
        correct_rates.append(correct_rate)

    try:
        correct_recommendations[user['count']].append(count_correct)
    except KeyError:
        correct_recommendations[user['count']] = [count_correct]

    print("Positive ratings: %s , Recommendations unknown tried: %s , Recommendations known tried: %s , Correct: %s , correct rate: %s" % (user['count'], count_1, count, count_correct, correct_rate))

print("Correct rates avg with a=0.75 -> threshold=0.857705843546: ", np.average(correct_rates))

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