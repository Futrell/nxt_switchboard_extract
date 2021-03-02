import sys
import csv

EMPTY = "O"

def process(lines, word_field, tag_field, sentence_fields):
    curr_sentence = None
    for d in lines:
        new_sentence = tuple(d[field] for field in sentence_fields)
        if curr_sentence != new_sentence:
            if curr_sentence is not None:
                yield ""
            curr_sentence = new_sentence
        tag = d[tag_field]
        if not tag:
            tag = EMPTY
        yield d[word_field], tag

def main(filename, word_field, tag_field, *sentence_fields):
    with open(filename, "rt") as infile:
        reader = csv.DictReader(infile)
        for line in process(reader, word_field, tag_field, sentence_fields):
            print(" ".join(line))

if __name__ == '__main__':
    main(*sys.argv[1:])
