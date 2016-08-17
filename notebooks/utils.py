# -*- coding: utf-8 -*-
"""
Created on Sat Aug  6 13:22:56 2016

@author: rakhunzy
"""


# normalize word if it contains two languages inside.
from alphabet_detector import AlphabetDetector
_alphabet_detector = AlphabetDetector()

def normalize2cyr(word):
    alphabets = _alphabet_detector.detect_alphabet(word)
    if 'CYRILLIC' in alphabets and len(alphabets) > 1:
        return word.translate(str.maketrans("qwertyuiopasdfghjklzxcvbnm", 
                                            "квертиуиопасдфгхжклзхсвбнм"))
    return word