#!.env/bin/python3

from pymongo import MongoClient
from bson.objectid import ObjectId
from scipy.linalg import norm


client = MongoClient('localhost', 28017)
hubchat = client.hubchat


def get_rated_posts():
    for post in hubchat.ratings.aggregate([
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
