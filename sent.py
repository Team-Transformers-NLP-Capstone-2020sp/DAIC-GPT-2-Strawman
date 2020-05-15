import pandas as pd, numpy as np
import re, os
import json
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer


sid = SentimentIntensityAnalyzer()
def polarity(sentence):
    score = sid.polarity_scores(sentence)["compound"]
    return score
"""
    if score <= -0.05:
        return "negative"
    elif score >= 0.05:
        return "positive"
    else:
        return "neutral"
        """

def parse_conversation(df):
    drop_rows = []
    for i in range(1, len(df)):
        # check for consecutive rows with same speaker
        df.loc[i, "value"] = str(df.loc[i, "value"])
        if "(" in df.loc[i, "value"]:
            idx = df.loc[i, "value"].find("(")
            df.loc[i, "value"] = df.loc[i, "value"][idx+1:-1]
        if df.loc[i, "speaker"] == df.loc[i - 1, "speaker"]:
            df.loc[i, "value"] = df.loc[i - 1, "value"] + " " + df.loc[i, "value"]
            drop_rows.append(i - 1)
    df = df.drop(drop_rows)
    df['value'] = df['value'] + " . "
    df['value'] = df['value'].str.lower()
    df = df[df != ""].reset_index(drop=True)
    df.loc[0, "value"] = "hi, how are you ?"

    avg = []
    for index, row in df.iterrows():
        if row["speaker"] == "Participant":
            avg.append(polarity(row["value"]))

    avg = np.array(avg)
    return len(avg[avg < -0.3]) / len(avg)


def get_personality(num):
    personality = []
    with open("personalities/" + str(num) + "_personality.txt") as f:
        for p in f.readlines():
            personality.append(p.strip().lower() + " .")

    f.close()
    return personality


convs = []
personalities = []
with open('scores.txt', 'w') as f:
    for i in range(300,500):
        try:
            personalities.append(get_personality(i))
        except:
            pass

        try:
            df = pd.read_csv('DAIC/' + str(i) + '_TRANSCRIPT.csv', delimiter="\t")
            f.write(str(i) + " " + str(parse_conversation(df)) + "\n")
        except:
            pass
