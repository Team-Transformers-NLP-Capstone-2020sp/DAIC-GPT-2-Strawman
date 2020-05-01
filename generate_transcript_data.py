import csv
import json
import os

data = dict()
nums = set(range(300, 301))
#nums.remove(342)
transcripts = list()
for transcript_num in nums:
    transcript = dict()
    conversation = list()
    with open('transcripts/' + str(transcript_num) + '_TRANSCRIPT.csv') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        prior_speaker = None
        for row in list(reader)[1:]:
            if len(row) < 4:
                continue
            current_speaker = row[2]
            sentence = row[3]
            if prior_speaker == current_speaker:
                conversation[-1] += sentence + ' . '
            else:
                conversation.append(sentence + '. ')
            prior_speaker = current_speaker
        conversation = [sent[:-1] for sent in conversation[:-20]]
        transcript['personality'] = conversation[1::2][:10]
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

with open('transcript_data1.json', 'w') as f:
    f.write(json.dumps(data))
