#!.env/bin/python3

from pymongo import MongoClient
from bson.objectid import ObjectId

client = MongoClient('localhost', 28017)
hubchat = client.hubchat


def did_like(postID, userID):
    """
    Checks whether the user liked or not the post
    :param postID:
    :param userID:
    :return:
    """
    for _ in hubchat.commentactions.aggregate([
        {
            "$match": {
                "post": ObjectId(postID),
                "createdBy": ObjectId(userID),
                "actionType": "voteUp",
                "deletedAt": {
                    "$exists": "false"
                }
            }
        },
        {
            "$project": {
                "post": 1,
                "comment": 1,
                "cmp": {
                    "$cmp": ["$post", "$comment"]
                }
            }
        },
        {
            "$match": {"cmp": 0}
        }
    ]):
        # There is one "voteUp" action
        return True
    # There is no "voteUp" action
    return False


def calculate_rating(postID, userID):
    like = did_like(postID, userID)


for post in hubchat.commentseens.aggregate([
    {
        "$lookup": {
            "from": "comments",
            "localField": "post",
            "foreignField": "_id",
            "as": "the_post"
        }
    }
]):
    # Filter out results without the post
    if not post['the_post']:
        continue

    # Filter out comments
    if post['the_post'][0]['type'] != "POST":
        c += 1
        continue

    p += 1

    postID = post['post']
    userID = post['user']

    score = calculate_rating(postID, userID)
