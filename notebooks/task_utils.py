# -*- coding: utf-8 -*-
"""
Created on Sat Aug  6 13:22:56 2016

@author: rakhunzy
"""


# In[]: normalize word if it contains two languages inside.
from alphabet_detector import AlphabetDetector
_alphabet_detector = AlphabetDetector()

def normalize2cyr(word, alphabets = None):
    if not alphabets:
        alphabets = _alphabet_detector.detect_alphabet(word)
    if 'CYRILLIC' in alphabets and len(alphabets) > 1:
        return word.translate(str.maketrans("qwertyuiopasdfghjklzxcvbnm", 
                                            "квертиуиопасдфгхжклзхсвбнм"))
    return word
    
def short_latin_word(word, alphabets = None):
    if not alphabets:    
        alphabets = _alphabet_detector.detect_alphabet(word)
    return 'LATIN' in alphabets and len(alphabets) == 1 and len(word) <= 2
    
def short_rus_word(word, alphabets = None):
    if not alphabets:
        alphabets = _alphabet_detector.detect_alphabet(word)
    return 'CYRILLIC' in alphabets and len(word) == 2 and word[0] == word[1]

_stop_words=['abex','act','amp','amp','app','club','http','php','post',
             'questions','vkontakte','clickunder','pop','card','uid','esquire',
             'oss','soso','video','audio','graffiti','imm']
def is_stop_word(word):
    return word in _stop_words
    
# In[]:     
import re

_coaelse_detect_re = re.compile(r"(.)\1+")
def coalesce_char_sequences(s):
    return re.sub(_coaelse_detect_re, r"\1\1", s)

# In[]: remove digits
from string import digits

_remove_digits = str.maketrans('', '', digits)
def remove_digits(inpt_string):
    return inpt_string.translate(_remove_digits)

# In[] Lang detect
#from langdetect import detect as lang_detect
##import threading
##_lang_detect_lock = threading.Lock()
#
#def is_ru(sentence):
#    return lang_detect(sentence) == 'ru'
import langid
langid.set_languages(['ru', 'uk', 'en'])
def is_ru(sentence):
    return langid.classify(sentence)[0] == 'ru'
