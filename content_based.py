
def category_distance(cat1, cat2):
    """
    Calculate distance between categories and subcategories
    Return -1 if categories have different roots
    :param cat1:
    :param cat2:
    :return:
    """
    if cat1 == cat2:
        return 0

    cat1_arr = cat1.strip('/').split('/')
    cat2_arr = cat2.strip('/').split('/')

    min_len = min(len(cat1_arr), len(cat2_arr))
    matches = 0

    for i in range(min_len):
        if cat1_arr[i] == cat2_arr[i]:
            matches += 1

    if matches == 0:
        return -1

    cat1_arr = cat1_arr[matches:]
    cat2_arr = cat2_arr[matches:]

    return len(cat1_arr) + len(cat2_arr)


def get_root(category):
    return category.strip('/').split('/')[0]


def split_by_root_category(categories):

    root_categories = {}

    for category, score in categories.items():
        root = get_root(category)
        try:
            root_categories[root][category] = score
        except KeyError:
            root_categories[root] = {}
            root_categories[root][category] = score

    return root_categories


def calculate_keywords_score(user_keywords, post_keywords):
    MISSING_KEYWORD_FACTOR = 0
    keywords_score = 0
    relevance_sum = 0

    for keyword, score in user_keywords.items():
        try:
            to_sum = post_keywords[keyword] * float(score)
            relevance_sum += post_keywords[keyword]
        except KeyError:
            to_sum = MISSING_KEYWORD_FACTOR * float(score)

        keywords_score += to_sum

    # Weighted avg
    if relevance_sum > 0:
        keywords_score /= relevance_sum

    return keywords_score


def calculate_categories_score(user_categories, post_categories):
    categories_score = 0
    relevance_sum = 0

    root_categories = split_by_root_category(user_categories)

    # For each root group
    for root_category, categories in root_categories.items():

        # For each item category
        for post_category, relevance in post_categories.items():

            # Skip if different root category
            if root_category != get_root(post_category):
                continue

            # For each category in root group, keep the one with min distance
            min_dist = None
            user_score = None
            for user_category, score in categories.items():
                distance = category_distance(user_category, post_category)
                if min_dist is None or distance < min_dist:
                    min_dist = distance
                    user_score = score

            categories_score += float(user_score) * relevance * (1 / (2.0 ** min_dist))
            relevance_sum += relevance * (1 / (2.0 ** min_dist))

    # Weighted avg
    if relevance_sum > 0:
        categories_score /= relevance_sum

    return categories_score


def predict_score(user_profile, item_profile):

    # Split item profile
    post_categories = {}
    post_keywords = {}
    for profile in item_profile:
        if profile['type'] == 'keyword':
            post_keywords[profile['text']] = profile['relevance']
        elif profile['type'] == 'category':
            post_categories[profile['text']] = profile['relevance']

    # Get keywords score
    keywords_score = calculate_keywords_score(user_profile['keywords'], post_keywords) if post_keywords != {} else None

    # Get score from categories
    category_score = calculate_categories_score(user_profile['categories'], post_categories) if post_categories != {} else None

    if keywords_score is None and category_score is None:
        return 0
    elif keywords_score is None:
        return category_score
    elif category_score is None:
        return keywords_score
    else:
        return (keywords_score + category_score) / 2.0
