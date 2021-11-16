from cky.CKYParser import CKYParser
from nltk import CFG
from nltk.tree import Tree
from typing import List


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


def testing_parse_sentences():
    g2 = CFG.fromstring("""
    S -> VP NP
    VP -> 'eat' NP
    VP -> V 'noodles' 'with' NP
    V -> B
    B -> C
    NP -> C E
    C -> D
    D -> E
    B -> E
    E -> 'e'
    """)
    parser = CKYParser(g2)
    parser.parse("e noodles with e e e e")


if __name__ == '__main__':
    testing_cnf_conversion()
    testing_parse_sentences()
