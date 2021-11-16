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


def parse_sentences(parser: CKYParser, sentences: List[str]):
    all = []
    all_true = True
    for s in sentences:
        trees = parser.parse(s)
        all.append(trees)
        print(f"Sentence: '{s}'")
        for t in trees:
            t = Tree.fromstring(str(t))
            t.pretty_print(unicodelines=True, nodedist=2)
            print(str(t) + '\n')
        if len(trees) == 0:
            print("The sentence is not accepted.\n")
            all_true = False
    if all_true:
        print("All accepted.")
    return all


if __name__ == '__main__':
    with open('zzj-french-grammar.txt') as f:
        lines = f.readlines()
    grammar = CFG.fromstring(lines)
    parser = CKYParser(grammar)
    s_accept = [
        # 'je regarde la television',
        # 'tu regardes la television',
        # 'il regarde la television',
        # 'nous regardons la television',
        # 'vous regardez la television',
        # 'ils regardent la television',
        # 'tu ne regardes pas la television',
        # 'il la regarde',
        # 'Jonathan aime le petit chat',
        # 'Jonathan aime les chats noirs',
        # 'je aime le Canada',
        # 'le beau chat le mange',
        'les aides aiment Montreal',
    ]

    s_reject = [
        # 'je mangent le poisson',
        # 'les noirs chats mangent le poisson',
        # 'la poisson mangent les chats',
        # 'je mange les',
    ]
    print("The following sentences should be accepted")
    parse_sentences(parser, s_accept)
    print("\nThe following sentences should be rejected")
    parse_sentences(parser, s_reject)
