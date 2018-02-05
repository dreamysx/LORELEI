import nltk
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize

# nltk.download('punkt')
class MTokenizer(object):

    @staticmethod
    def tokenize_string(string):
       """
       I designed this method to be used independently of an obj/field. If this is the case, call _tokenize_field.
       It's more robust.
       :param string: e.g. 'salt lake city'
       :return: list of tokens
       """
       list_of_sentences = list()
       tmp = list()
       tmp.append(string)
       k = list()
       k.append(tmp)
       # print k
       list_of_sentences += k  # we are assuming this is a unicode/string

       word_tokens = list()
       for sentences in list_of_sentences:
           # print sentences
           for sentence in sentences:
               for s in sent_tokenize(sentence):
                   word_tokens += word_tokenize(s)

       return word_tokens