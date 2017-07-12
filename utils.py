from html.parser import HTMLParser
import re
import numpy as np


class MLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self.reset()
        self.strict = False
        self.convert_charrefs= True
        self.fed = []

    def handle_data(self, d):
        self.fed.append(d)

    def get_data(self):
        return ''.join(self.fed)


def strip_html_tags(text):
    s = MLStripper()
    s.feed(text)
    return s.get_data()


def extract_urls(text):
    return re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)


def f_score(p, r, b=0.5):
    """
    Calculate F score
    :param p: Precision
    :param r: Recall
    :param b:
    :return:
    """
    return (1 + (b ** 2)) * ((p * r) / (((b ** 2) * p) + r))


class RecommendationsBinAvg():
    def __init__(self):
        self.recommendations_avg = {
            '[0-10]': [],
            '[10-20]': [],
            '[20-30]': [],
            '[30-50]': [],
            '[50-70]': [],
            '[70-100]': [],
            '[100-150]': [],
            '[150+]': []
        }

    def __str__(self):
        return str(self.recommendations_avg)

    def add(self, number_of_rates, number_of_recommendations):

        if 0 <= number_of_rates <= 10:
            key = '[0-10]'
        elif 10 < number_of_rates <= 20:
            key = '[10-20]'
        elif 20 < number_of_rates <= 30:
            key = '[20-30]'
        elif 30 < number_of_rates <= 50:
            key = '[30-50]'
        elif 50 < number_of_rates <= 70:
            key = '[50-70]'
        elif 70 < number_of_rates <= 100:
            key = '[70-100]'
        elif 100 < number_of_rates <= 150:
            key = '[100-150]'
        else:
            key = '[150+]'

        self.recommendations_avg[key].append(number_of_recommendations)

    def avg(self):
        recommendations_avg = self.recommendations_avg.copy()

        for k, v in recommendations_avg.items():
            self.recommendations_avg[k] = np.average(v) if v else None
