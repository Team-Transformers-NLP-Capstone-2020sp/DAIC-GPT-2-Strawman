import pandas as pd, numpy as np
import re, os
import json

def parse_conversation(df):
    drop_rows = []
    for i in range(1, len(df)):
        # check for consecutive rows with same speaker
        if df.loc[i, "speaker"] == df.loc[i - 1, "speaker"]:
            df.loc[i, "value"] = str(df.loc[i - 1, "value"]) + " " + str(df.loc[i, "value"])
            drop_rows.append(i - 1)
    df = df.drop(drop_rows)
    df['value'] = df['value'] + " . "
    df['value'] = df['value'].str.lower()
    df = df[df != ""]
    return df

convs = []
for i in range(300,350):
    try:
        df = pd.read_csv('DAIC/' + str(i) + '_TRANSCRIPT.csv', delimiter="\t")
        convs.append(parse_conversation(df))
    except:
        pass
convs_full = pd.concat(convs).reset_index(drop=True)
#print(convs_full)

transcripts = list()
data = dict()
for i in convs:
    transcript = dict()
    conversation = list()
    for _, row in i.iterrows():
        conversation.append(row['value'])
    conversation = [sent[:-1] for sent in conversation[:-20]]
    transcript['personality'] = "I am depressed"
    depressed_convo = conversation[1::2]
    examples = list()
    for p_num, i in enumerate(range(1, len(conversation), 2)):
        example = dict()
        example['candidates'] = [depressed_convo[j] for j in range(len(depressed_convo)) if j != p_num] + [depressed_convo[p_num]]
        example['history'] = conversation[:i]
        examples.append(example)
    transcript['utterances'] = examples
    transcripts.append(transcript)

data['train'] = transcripts
data['valid'] = transcripts

with open('transcript_data2.json', 'w') as f:
    f.write(json.dumps(data))


