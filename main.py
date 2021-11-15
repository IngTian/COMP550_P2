from cky.CKYParser import CKYParser
from nltk import CFG
from nltk.tree import Tree


def testing_cnf_conversion():
    g2 = CFG.fromstring("""
    S -> VP NP
    VP -> 'eat' NP
    VP -> V 'noodles' 'with' NP PP PP
    V -> B
    B -> C
    NP -> C E
    C -> D
    D -> E
    B -> E
    E -> 'e'
    """)
    parser = CKYParser(g2)
    print(parser.cnf_grammar.productions())

    g3 = CFG.fromstring("""
    A -> C C 'c'
    C -> 'a'
    """)
    parser = CKYParser(g3)
    print(parser.cnf_grammar.productions())


if __name__ == '__main__':
    testing_cnf_conversion()
