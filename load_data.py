from datasets import load_dataset
from transformers import BertTokenizer

def filter_texts(examples):
    examples["labels"] = examples["label"].copy()
    examples.pop('label',None)
    return examples


def make_labels(samples):
    samples['labels'] = samples['input_ids'].copy()
    return samples


def mrpc():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    datasets = load_dataset('glue', 'mrpc')
    mrpc_tok = datasets.map(lambda samples: tokenizer(samples['sentence1'], samples['sentence2'],padding='max_length',max_length=512),
                       remove_columns=['sentence1', 'sentence2','idx'],
                       load_from_cache_file=False
                      )
    mrpc = mrpc_tok.map(
        filter_texts,
        batched=True,
        num_proc=4
    )
    return mrpc

def wiki():
    datasets = load_dataset('wikitext', 'wikitext-2-raw-v1')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenized_dataset = datasets.map(lambda samples: tokenizer(samples['text'], max_length=512, truncation=True, padding= 'max_length'), batched=True, num_proc=4, remove_columns = ["text"])
    block_size = 128
    lm_datasets = tokenized_dataset.map(
        make_labels,
        batched=True,
        num_proc=4
    )

    return lm_datasets

