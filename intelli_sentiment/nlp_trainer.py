import os
import random
import csv
import logging
from pathlib import Path
import urllib.request

import spacy
from spacy.util import minibatch, compounding

logging.basicConfig(
    format='%(asctime)s %(levelname)s: %(message)s', level=logging.DEBUG)

DATASET_URL = os.environ.get('DATASET_URL')

N_ITER = 100

model_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '../data/model')


class CatDataset:
    def __init__(self, records, cats):
        self.records = records
        self.cats = cats

    def prepare(self, split=0.8):
        records = []
        cat_records = []
        for record in self.records:
            if any(record[1]['cats'].values()):
                cat_records.append(record)
            else:
                records.append(record)

        random.shuffle(records)
        random.shuffle(cat_records)
        records = records[:1000] # reduce training size
        split = int(len(records) * split)
        cat_split = int(len(cat_records) * split)

        return (records[:split] + cat_records[:cat_split],
                records[split:] + cat_records[cat_split:])

    @classmethod
    def load(cls):
        cats = ['NOT_ENOUGH_PAY']
        data = []
        with urllib.request.urlopen(DATASET_URL) as response:
            response = response.read().decode('utf-8')
            reader = csv.DictReader(response.splitlines(), delimiter='\t')
            for row in reader:
                cats = {c: row[c] == '1' for c in cats}
                data.append((row['TEXT'], dict(cats=cats)))

        return cls(data, cats)


def main():
    nlp = spacy.load(model_path)  # load existing spaCy model

    if 'textcat' not in nlp.pipe_names:
        textcat = nlp.create_pipe('textcat')
        nlp.add_pipe(textcat, last=True)
    else:
        textcat = nlp.get_pipe('textcat')

    # add label to text classifier
    cat_dataset = CatDataset.load()
    for tag in cat_dataset.cats:
        textcat.add_label(tag)

    # load dataset
    logging.info('Loading sentiment cats data...')
    train_data, test_data = cat_dataset.prepare()
    logging.info(f'Using {len(train_data) + len(test_data)} examples ' +
                 f'({len(train_data)} training, {len(test_data)} evaluation)')

    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'textcat']
    with nlp.disable_pipes(*other_pipes):  # only train textcat
        optimizer = nlp.begin_training()
        logging.info('Training the model...')
        logging.info(
            '{:^5}\t{:^5}\t{:^5}\t{:^5}'.format('LOSS', 'P', 'R', 'F'))
        for i in range(N_ITER):
            losses = {}
            # batch up the examples using spaCy's minibatch
            batches = minibatch(train_data, size=compounding(10, 500, 1.01))
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(
                    texts, annotations, sgd=optimizer, drop=0.2, losses=losses)
            with textcat.model.use_params(optimizer.averages):
                # evaluate on the dev data split off in load_data()
                scores = evaluate(nlp.tokenizer, textcat, test_data)
            logging.info(
                '{0:.3f}\t{1:.3f}\t{2:.3f}\t{3:.3f}'  # print a simple table
                .format(losses['textcat'], scores['textcat_p'], scores[
                    'textcat_r'], scores['textcat_f']))

    # test the trained model
    import csv
    from tests.util import _tokenizer
    _file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        '../tests/sentiment_dataset_f5mgroup.tsv')
    with open(_file) as dataset_file:
        reader = csv.reader(dataset_file, delimiter='\t')
        count = 0
        for row in reader:
            for sent in _tokenizer(row[1]):
                count += 1
                cats = nlp(sent).cats
                if cats['NOT_ENOUGH_PAY'] > 0.5:
                    print(sent)

    # test_text = "This movie sucked"
    # doc = nlp(test_text)
    # print(test_text, doc.cats)
    #
    # if output_dir is not None:
    #     output_dir = Path(output_dir)
    #     if not output_dir.exists():
    #         output_dir.mkdir()
    #     nlp.to_disk(output_dir)
    #     print("Saved model to", output_dir)
    #
    #     # test the saved model
    #     print("Loading from", output_dir)
    #     nlp2 = spacy.load(output_dir)
    #     doc2 = nlp2(test_text)
    #     print(test_text, doc2.cats)


def evaluate(tokenizer, textcat, test_data):
    docs = (tokenizer(d[0]) for d in test_data)
    tp = 1e-8  # True positives
    fp = 1e-8  # False positives
    fn = 1e-8  # False negatives
    tn = 1e-8  # True negatives
    for i, doc in enumerate(textcat.pipe(docs)):
        gold = test_data[i][1]['cats']
        for label, score in doc.cats.items():
            if label not in gold:
                continue
            if score >= 0.5 and gold[label] >= 0.5:
                tp += 1.
            elif score >= 0.5 and gold[label] < 0.5:
                fp += 1.
            elif score < 0.5 and gold[label] < 0.5:
                tn += 1
            elif score < 0.5 and gold[label] >= 0.5:
                fn += 1
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f_score = 2 * (precision * recall) / (precision + recall)
    return {'textcat_p': precision, 'textcat_r': recall, 'textcat_f': f_score}


if __name__ == '__main__':
    main()
