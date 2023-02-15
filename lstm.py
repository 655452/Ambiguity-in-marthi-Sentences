# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 10:47:25 2021

@author: ss10224
"""

import keras
from keras_self_attention import SeqSelfAttention
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from opfunu.cec_basic.cec2014_nobias import *
import RainOptim


import warnings 
warnings.filterwarnings('ignore')

import argparse
import re

import torch
# from tabulate import tabulate
from torch.nn.functional import softmax
from tqdm import tqdm
from transformers import BertTokenizer

from utils.dataset import GlossSelectionRecord, _create_features_from_records
from utils.model import BertWSD, forward_gloss_selection
from utils.wordnet import get_glosses
import ambiguous

from keras.optimizers import RMSprop
opt = RMSprop(lr=0.01)


MAX_SEQ_LENGTH = 128
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_predictions(model, tokenizer, sentence):
    re_result = re.search(r"(.*)", sentence)
    if re_result is None:
        print("\nIncorrect input format. Please try again.")
        return

    ambiguous_word = re_result.group(1).strip()
    sense_keys = []
    definitions = []
    for sense_key, definition in get_glosses(ambiguous_word, None).items():
    #for sense_key, definition in get_example_sentences(ambiguous_word, None).items():
        sense_keys.append(sense_key)
        definitions.append(definition)

    record = GlossSelectionRecord("test", sentence, sense_keys, definitions, [-1])
    features = _create_features_from_records([record], MAX_SEQ_LENGTH, tokenizer,
                                             cls_token=tokenizer.cls_token,
                                             sep_token=tokenizer.sep_token,
                                             cls_token_segment_id=1,
                                             pad_token_segment_id=0,
                                             disable_progress_bar=True)[0]
    
    with torch.no_grad():
        logits = torch.zeros(len(definitions), dtype=torch.double).to(DEVICE)
        for i, bert_input in tqdm(list(enumerate(features)), desc="Progress"):
            logits[i] = model.ranking_linear(
                model.bert(
                    input_ids=torch.tensor(bert_input.input_ids, dtype=torch.long).unsqueeze(0).to(DEVICE),
                    attention_mask=torch.tensor(bert_input.input_mask, dtype=torch.long).unsqueeze(0).to(DEVICE),
                    token_type_ids=torch.tensor(bert_input.segment_ids, dtype=torch.long).unsqueeze(0).to(DEVICE)
                )[1]
            )
        scores = softmax(logits, dim=0)

    # return sorted(zip(sense_keys, definitions, scores), key=lambda x: x[-1], reverse=True)
    return definitions

def model(length,classes):
    obj_func = F5
    lb = [-100]
    ub = [100]
    problem_size = 100
    batch_size = 25
    verbose = False
    epoch = 10
    vocab_size =2000
    model = keras.models.Sequential()
    model.add(Embedding(vocab_size, 8, input_length=2))
    model.add(keras.layers.LSTM(units=64,return_sequences=True))
    model.add(SeqSelfAttention(attention_activation='sigmoid'))
    model.add(Flatten())
    model.add(Dense(classes+1, activation='sigmoid'))
    optim= RainOptim.Rain(obj_func, lb, ub, problem_size, batch_size, verbose, epoch)
    Model = optim.train(),model
    model.compile(
        optimizer=opt,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model

