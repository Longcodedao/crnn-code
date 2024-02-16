#!/usr/bin/env python
# coding: utf-8

from collections import defaultdict

import numpy as np
import math
import torch
from scipy.special import logsumexp



NINF = -1 * float('inf')
DEFAULT_EMISSION_THRESHOLD = np.log(0.01)

class Hypothesis:
    def __init__(self, sequence, log_prob):
        self.sequence = sequence
        self.log_prob = log_prob



def reconstruct(labels, blank = 0):
    new_labels = []
    # Merge duplicate chars        
    previous = None
    for char in labels:
        if char != previous:
            new_labels.append(char)
            previous = char
    # Delete blank
    new_labels = [char for char in new_labels if char != blank]
    return new_labels


def greedy_decode(log_prob, blank = 0, **kwargs):
    labels = np.argmax(log_prob, axis = -1)
    labels = reconstruct(labels, blank = blank)
    return labels


def beam_search_decode(log_prob, blank = 0, beam_size = 10,
                        threshold = DEFAULT_EMISSION_THRESHOLD):

    initial_hypothesis = Hypothesis(sequence = [], log_prob = 0.0)
    beam = [initial_hypothesis]
    
    seq_length, class_counts = log_prob.shape

    for t in range(seq_length):
        new_beam = []
        
        for hypothesis in beam:
            for c in range(class_counts):
                log = log_prob[t, c]
                if log < threshold:
                    continue
                extended_sequences = hypothesis.sequence + [c]
                log_prob_extend = hypothesis.log_prob + log

                new_hypothesis = Hypothesis(sequence = extended_sequences,
                                            log_prob = log_prob_extend)
                new_beam.append(new_hypothesis)
            # beam = sorted(new_beam, key = lambda x: x.log_prob, reverse = True)[:beam_width]
            beam = sorted(new_beam, key=lambda x: x.log_prob, reverse=True)[:beam_size]

    total_accu_log_prob = {}
    # Sum up beams to produce labels
    for hypothesis in beam:
        labels, prob_log = hypothesis.sequence, hypothesis.log_prob
        # print(prob_log)
        labels = tuple(reconstruct(hypothesis.sequence, blank = blank))
        # print(labels)
        total_accu_log_prob[labels] = logsumexp([prob_log, total_accu_log_prob.get(labels, NINF)])
    
    label_beams = [(list(labels), accu_prob) for labels, accu_prob in total_accu_log_prob.items()] 
    label_beams.sort(key = lambda x: x[1], reverse = True)

    # print(label_beams)
    return label_beams[0][0]


def ctc_decoder(log_probs, label2char = None, blank = 0, method = 'beam_search', beam_size = 10):
    log_probs = np.transpose(log_probs.cpu().numpy(), (1, 0, 2))

    decoders = {
        'greedy': greedy_decode,
        'beam_search': beam_search_decode
    }

    decoder = decoders[method]

    decoded_list = []
    for log_prob in log_probs:
        labels = decoder(log_prob, blank = blank, beam_size = beam_size)
        # print(labels)
        if label2char:
            labels = [label2char[l] for l in labels]
        decoded_list.append(labels)
        
    return decoded_list




