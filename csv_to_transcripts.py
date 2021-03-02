import sys
import pandas as pd

PUNCT = set(".,'?!")

def reconcatenate(tokens):
    def gen():
        for i, token in enumerate(tokens):
            if i != 0 and not any(c in PUNCT for c in token):
                yield " "
            yield token
    return "".join(gen())

def capitalize_first(s):
    first, *rest = s
    return "".join([first.upper()] + rest)

def main(filename):
    input = pd.read_csv(filename)
    df = input[['dialogue', 'sentence_id', 'speaker', 'orth']].groupby(['dialogue', 'sentence_id', 'speaker']).aggregate(reconcatenate).reset_index()
    df['orth'] = df['orth'].map(capitalize_first)
    for i, row in df.iterrows():
        dialogue = row['dialogue']
        print(" ".join(map(str, row[['dialogue', 'sentence_id', 'speaker']])), ": ", row['orth'], sep="")

if __name__ == '__main__':
    main(*sys.argv[1:])
