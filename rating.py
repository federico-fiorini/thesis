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


def get_comments(postUUID, userID):
    """
    Returns how many comments the user left on the post
    :param postID:
    :param userID:
    :return:
    """
    count = 0

    for _ in hubchat.comments.find(
        {
            "postUUID": postUUID,
            "createdBy": ObjectId(userID),
        }
    ):
        count += 1

    return count


def calculate_rating(postID, postUUID, userID):
    liked = did_like(postID, userID)
    comments = get_comments(postUUID, userID)

    if liked:
        if comments == 0:
            return 3
        elif 1 <= comments <= 2:
            return 4
        elif comments >= 3:
            return 5
    else:
        if comments == 0:
            return 1
        elif comments == 1:
            return 2
        elif 2 <= comments <= 3:
            return 3
        elif 4 <= comments <= 5:
            return 4
        elif comments >= 6:
            return 5


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
        continue

    postID = post['post']
    postUUID = post['postUUID']
    userID = post['user']

    score = calculate_rating(postID, postUUID, userID)
    print(score)
