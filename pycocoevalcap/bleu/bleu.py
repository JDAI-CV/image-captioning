#!/usr/bin/env python
#
# File Name : bleu.py
#
# Description : Wrapper for BLEU scorer.
#
# Creation Date : 06-01-2015
# Last Modified : Thu 19 Mar 2015 09:13:28 PM PDT
# Authors : Hao Fang <hfang@uw.edu> and Tsung-Yi Lin <tl483@cornell.edu>

# Last modified : Wed 22 May 2019 08:10:00 PM EDT
# By Sabarish Sivanath
# To support Python 3

from .bleu_scorer import BleuScorer


class Bleu:
    def __init__(self, n=4):
        # default compute Blue score up to 4
        self._n = n
        self._hypo_for_image = {}
        self.ref_for_image = {}

    def compute_score(self, gts, res, score_option = 'closest', verbose = 1):
        '''
        Inputs:
            gts - ground truths
            res - predictions
            score_option - {shortest, closest, average}
            verbose - 1 or 0
        Outputs:
            Blue scores
        '''
        assert(gts.keys() == res.keys())
        imgIds = gts.keys()

        bleu_scorer = BleuScorer(n=self._n)
        for id in imgIds:
            hypo = res[id]
            ref = gts[id]

            # Sanity check.
            assert(type(hypo) is list)
            assert(len(hypo) == 1)
            assert(type(ref) is list)
            #assert(len(ref) >= 1)

            bleu_scorer += (hypo[0], ref)

        score, scores = bleu_scorer.compute_score(option = score_option, verbose =verbose)

        # return (bleu, bleu_info)
        return score, scores

    def method(self):
        return "Bleu"
