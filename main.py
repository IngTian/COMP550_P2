from cky.CKYParser import CKYParser
from nltk import CFG

if __name__ == '__main__':
    grammar: CFG = CFG.fromstring(open('./cfg.txt', 'r').read())
    parser = CKYParser(grammar)
