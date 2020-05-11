import pandas as pd, numpy as np
import re, os
import json

def parse_conversation(df):
    drop_rows = []
    for i in range(1, len(df)):
        # check for consecutive rows with same speaker
        if "(" in df.loc[i, "value"]:
            idx = df.loc[i, "value"].find("(")
            df.loc[i, "value"] = df.loc[i, "value"][idx+1:-1]
        if df.loc[i, "speaker"] == df.loc[i - 1, "speaker"]:
            df.loc[i, "value"] = str(df.loc[i - 1, "value"]) + " " + str(df.loc[i, "value"])
            drop_rows.append(i - 1)
    df = df.drop(drop_rows)
    df['value'] = df['value'] + " . "
    df['value'] = df['value'].str.lower()
    df = df[df != ""]
    return df

def get_personality(num):
    personality = []
    with open("personalities/" + str(num) + "_personality.txt") as f:
        for p in f.readlines():
            personality.append(p.strip().lower() + " .")

    f.close()
    return personality


convs = []
personalities = []
for i in range(399,400):
    try:
        df = pd.read_csv('DAIC/' + str(i) + '_TRANSCRIPT.csv', delimiter="\t")
        convs.append(parse_conversation(df))
    except:
        pass

    try:
        personalities.append(get_personality(i))
    except:
        personalities.append(["i am depressed ."])
convs_full = pd.concat(convs).reset_index(drop=True)

transcripts = []
data = {}
for conv_num, conv in enumerate(convs):
    transcript = {}
    conversation = []
    for _, row in conv.iterrows():
        conversation.append(row['value'])
    conversation = [sent[:-1] for sent in conversation[:-20]]
    transcript['personality'] = personalities[conv_num]
    depressed_convo = conversation[1::2]
    examples = []
    for p_num, i in enumerate(range(1, len(conversation), 2)):
        example = {}
        example['candidates'] = [depressed_convo[j] for j in range(len(depressed_convo)) if j != p_num] + [depressed_convo[p_num]]
        example['history'] = conversation[:i]
        examples.append(example)
    transcript['utterances'] = examples
    transcripts.append(transcript)

data['train'] = transcripts[:-2]
data['valid'] = transcripts[-2:]

with open('transcript_data2.json', 'w') as f:
    f.write(json.dumps(data))
