import random
import re
from itertools import product
from nltk import CFG
from nltk.grammar import Production, Nonterminal
from nltk.tree import Tree
from typing import List, Dict, Union, Tuple, Set
from enum import Enum


class RuleType(Enum):
    TERMINAL = 1
    UNARY = 2
    BINARY = 3
    BINARY_PLUS = 4


class CKYTableEntry:
    cnf: Production
    left_child: "CKYTableEntry"
    right_child: "CKYTableEntry"

    def __init__(self, cnf: Production, left: Union["CKYTableEntry", None], right: Union["CKYTableEntry", None]):
        self.cnf = cnf
        self.left_child = left
        self.right_child = right

    def is_terminal(self):
        return self.left_child is None and self.right_child is None


class CKYParser:

    def __init__(self, grammar: CFG):
        self.__original_grammar = grammar
        self.__organized_cfg_grammar = CKYParser.__organize_grammar(grammar.productions())

        # This table records all possible unary paths, given the start point and the end point.
        # The key follows the format: <Non-terminal> -> <EndPoint1>,<EndPoint2>,<EndPoint3>.
        # e,g. A -> B,C / A -> 'ne',B,C
        self.__unary_chain_table: Dict[str, List[List[Production]]] = dict()
        self.__cnf_grammar: CFG
        self.__initialize_cnf()

    def parse(self, sentence: str) -> List[Tree]:
        dp_table: List[List[List[CKYTableEntry]]] = self.__cky(sentence)
        start_symbol = self.__original_grammar.start()
        start_nodes = list(filter(lambda entry: entry.cnf.lhs() == start_symbol, dp_table[0][-1]))
        possible_trees: List[Tree] = list(map(lambda node: self.__parse_tree(node), start_nodes))
        converted_trees: List[Tree] = list()
        for possible_tree in possible_trees:
            converted_trees += self.__translate_to_cgf(possible_tree)
        return converted_trees

    def __translate_to_cgf(self, tree: Tree) -> List[Tree]:
        return self.__translate_unary(self.__translate_binary_plus(tree))

    def __translate_binary_plus(self, tree: Tree) -> Tree:
        """
        Eliminate all added nodes due to binary plus.
        :param tree:
        :return:
        """
        # Base case, if terminal.
        if type(tree) != tree:
            return tree

        purified_children: List[Union[Tree, str]] = list()
        for child in tree:
            child = self.__translate_binary_plus(child)
            if type(child) == Tree and CKYParser.__is_cnf_specific_rule(str(child.label())):
                purified_children += list(child)
            else:
                purified_children.append(child)

        return Tree(tree.label(), purified_children)

    def __translate_unary(self, tree: Tree) -> List[Tree]:
        if type(tree) != Tree:
            return [tree]

        temp_unary_tree = self.__build_chain_trees(tree.label(), list(map(lambda t: t.label(), list(tree))))
        possible_child_trees = [self.__translate_unary(child) for child in tree]
        cartesian_child_trees = list(product(*possible_child_trees))

        result: List[Tree] = list()

        for unary_temp in temp_unary_tree:
            for child in cartesian_child_trees:
                top_node, bottom_node = unary_temp[0], unary_temp[1]
                for c in child:
                    bottom_node.append(c)
                result.append(top_node)

        return result

    @staticmethod
    def __build_tree_from_unary_chain(chain: List[Production]) -> Tuple[Tree, Tree]:
        """
        Build intermediate trees based on unary chains.
        :param chain:
        :return:
        """
        head_rule = chain.pop()
        head_node: Tree = Tree(head_rule.lhs())
        node = head_node
        while len(chain) != 0:
            cur = chain.pop()
            new_node = Tree(cur.lhs())
            node.append(new_node)
            node = new_node
        return head_node, node

    def __find_chains(self, lhs: Nonterminal, rhs: List[Nonterminal, str]) -> List[List[Production]]:
        """
        Find possible unary chains based
        on starting and end points.
        :param lhs:
        :param rhs:
        :return:
        """

        def stringify_list_of_nodes(node: Union[Nonterminal, str]) -> str:
            return f"'{node}'" if type(node) == str else str(node)

        identifier = f"{str(lhs)} -> {','.join(map(stringify_list_of_nodes, rhs))}"

        return self.__unary_chain_table.get(identifier)

    def __build_chain_trees(self, lhs: Nonterminal, rhs: List[Nonterminal, str]) -> List[Tuple[Tree, Tree]]:
        """
        Build temp unary trees from starting and end points.
        :param lhs:
        :param rhs:
        :return:
        """
        chains: List[List[Production]] = self.__find_chains(lhs, rhs)
        if chains:
            return list(map(lambda chain: CKYParser.__build_tree_from_unary_chain(chain), chains))
        else:
            t = Tree(lhs)
            return [(t, t)]

    def __parse_tree(self, node: CKYTableEntry) -> Tree:
        """
        Backtrack and produce a tree.
        :param node: A CKYTable entry.
        :return: A Tree
        """
        if node.is_terminal():
            return Tree(node.cnf.lhs(), node.cnf.rhs())
        left_sub_tree, right_sub_tree = self.__parse_tree(node.left_child), self.__parse_tree(node.right_child)
        return Tree(node.cnf.lhs(), [left_sub_tree, right_sub_tree])

    def __cky(self, sentence: str) -> List[List[List[CKYTableEntry]]]:
        """
        Classic CKY algorithm to compute all possible
        parse trees.
        :param sentence: A Sentence.
        :return: The result DP table.
        """
        if sentence[-1] == '.':
            sentence = sentence[:, -1]
        sentence = sentence.split(" ")
        sentence_length = len(sentence)
        dp = [[list() for j in range(sentence_length)] for i in range(sentence_length)]

        # Fill the table diagonally.
        for idx in range(sentence_length):
            list_of_terminals = self.__find_cnf_by_rhs(sentence[idx], "", True)
            list_of_terminals = list(map(lambda terminal: CKYTableEntry(terminal, None, None), list_of_terminals))
            dp[idx][idx] = list_of_terminals

        def find_cnf(first_rhs: List[CKYTableEntry], second_rhs: List[CKYTableEntry]) -> List[CKYTableEntry]:
            result = list()
            for rhs1 in first_rhs:
                for rhs2 in second_rhs:
                    rhs1_node, rhs2_node = rhs1.cnf.lhs(), rhs2.cnf.lhs()
                    result += list(map(lambda rule: CKYTableEntry(rule, rhs1, rhs2),
                                       self.__find_cnf_by_rhs(rhs1_node, rhs2_node, False)))
            return result

        # Fill the rest DP table.
        for right_bound in range(1, sentence_length):
            for left_bound in range(right_bound - 1):
                # The DP entry to fill is dp[left_bound][right_bound]
                possible_candidate = list()
                for split_point in range(left_bound, right_bound - 1):
                    possible_candidate += find_cnf(dp[left_bound][split_point], dp[split_point + 1][right_bound])

        return dp

    def __find_cnf_by_rhs(
            self,
            rhs1: Union[Nonterminal, str],
            rhs2: Union[Nonterminal, str],
            is_terminal: bool
    ) -> List[Production]:
        """
        Find all corresponding CNF rules with
        the specified rhs.
        :param rhs1:
        :param rhs2:
        :param is_terminal:
        :return:
        """

        result = list()

        for rule in self.__cnf_grammar.productions():
            assert isinstance(rule, Production)
            if is_terminal and rule.rhs()[0] == rhs1:
                result.append(rule)
            elif not is_terminal and rule.rhs()[0] == rhs1 and rule.rhs()[1] == rhs2:
                result.append(rule)

        return result

    def __initialize_cnf(self) -> None:
        cnf_rules: List[Production] = list()
        for cgf_rule in self.__original_grammar.productions():
            cnf_rules += CKYParser.__build_binary_plus_cnf(cgf_rule)
        organized_cnf = CKYParser.__organize_grammar(cnf_rules)
        node_reachable_table: Dict[Nonterminal, List[List[Production]]] = dict()
        for starting_point in organized_cnf.keys():
            CKYParser.__parse_unary_rules(starting_point, node_reachable_table, organized_cnf)

        def stringify_list_of_nodes(node: Union[Nonterminal, str]) -> str:
            return f"'{node}'" if type(node) == str else str(node)

        for starting_point in node_reachable_table.keys():
            for paths in node_reachable_table.get(starting_point):
                new_unary_rules: Set[Production] = set()
                for path in paths:
                    # Non-unary
                    if len(path) <= 1:
                        continue
                    # Unary
                    path_identifier = f"{str(path[0].lhs())} -> {','.join(map(stringify_list_of_nodes, path[-1].rhs()))}"
                    if path_identifier in self.__unary_chain_table.keys():
                        self.__unary_chain_table.get(path_identifier).append(path)
                    else:
                        self.__unary_chain_table[path_identifier] = [path]
                    new_unary_rules = new_unary_rules.union([Production(path[0].lhs(), path[-1].rhs())])

        self.__cnf_grammar = CFG(self.__original_grammar.start(), cnf_rules)

    @staticmethod
    def __build_binary_plus_cnf(rule: Production) -> List[Production]:
        """
        Convert a Binary+ rule into CNF.
        :param rule: A binary plus rule.
        :return: A list of CNF.
        """
        lhs, rhs = rule.lhs(), rule.rhs()
        results: List[Production]

        # If terminal or binary, simply return.
        if CKYParser.__is_binary(rule) or CKYParser.__is_terminal(rule) or CKYParser.__is_unary(rule):
            results = [Production(lhs, rhs)]
        # If the rhs is binary_plus, do the following.
        else:
            rhs_length = len(rhs)
            rhs_first_half = rhs[:rhs_length - 1]
            rhs_first_half_symbol = Nonterminal(CKYParser.__get_cnf_non_terminal_id(rhs_first_half))
            first_half_temp_cnf = Production(rhs_first_half_symbol, rhs_first_half)
            first_half_rules = CKYParser.__build_binary_plus_cnf(first_half_temp_cnf)
            if type(rhs[-1]) == Nonterminal:
                results = [Production(lhs, [rhs_first_half_symbol, rhs[-1]])] + first_half_rules
            else:
                second_half_rule = Production(Nonterminal(CKYParser.__get_cnf_non_terminal_id([rhs[-1]])), rhs[-1])
                return [Production(lhs, [rhs_first_half_symbol, second_half_rule.lhs()])] + first_half_rules + [
                    second_half_rule]

        return results

    @staticmethod
    def __is_cnf_specific_rule(label: str) -> bool:
        return re.match(r'^(CNF-Symbol).+', label) is not None

    @staticmethod
    def __get_cnf_non_terminal_id(rhs: List[Union[Nonterminal, str]]) -> str:
        return f'CNF-Symbol-Replace-{"-".join(map(lambda s: str(s), rhs))}-{random.randint(0, 10000)}'

    @staticmethod
    def __parse_unary_rules(
            node: Nonterminal,
            node_reachable_table: Dict[Nonterminal, List[List[Production]]],
            organized_grammar: Dict[Nonterminal, Dict[RuleType, List[Production]]]
    ) -> List[List[Production]]:
        """
        Parse unary rules.
        :param node: The starting non-terminal node.
        :param node_reachable_table: A table.
        :param organized_grammar: Organized grammar.
        :return: List of possible paths from the starting node.
        """

        if node in node_reachable_table.keys():
            return node_reachable_table.get(node)

        # Concatenate all terminal, binary, and binary-plus productions.
        result: List[List[Production]] = list()
        node_cfg_record = organized_grammar.get(node)
        for terminal_rule in node_cfg_record.get(RuleType.TERMINAL):
            result.append([terminal_rule])

        for binary in node_cfg_record.get(RuleType.BINARY):
            result.append([binary])

        for binary_plus_rule in node_cfg_record.get(RuleType.BINARY_PLUS):
            result.append([binary_plus_rule])

        unary_paths: List[List[Production]] = list()
        for unary_rule in node_cfg_record.get(RuleType.UNARY):
            target_node = unary_rule.rhs()[0]
            paths = CKYParser.__parse_unary_rules(target_node, node_reachable_table, organized_grammar)
            for path in paths:
                cloned_path = path.copy()
                cloned_path.insert(0, unary_rule)
                unary_paths.append(cloned_path)

        result += unary_paths

        node_reachable_table[node] = result

        return result

    @staticmethod
    def __organize_grammar(rules: List[Production]) -> Dict[Nonterminal, Dict[RuleType, List[Production]]]:
        """
        Organize each CFG rule into 4 categories.
        Terminal:    A -> 'word'
        UNARY:       A -> B
        BINARY:      A -> B C
        BINARY_PLUS: A -> B C D / A -> B 'ne' D / A -> 'ne' B
        :return: Organized CGF rules.
        """
        result: Dict[Nonterminal, Dict[RuleType, List[Production]]] = dict()
        for rule in rules:
            assert isinstance(rule, Production)
            lhs, rhs = rule.lhs(), rule.rhs()
            if lhs not in result.keys():
                result[lhs] = {
                    RuleType.TERMINAL: [],
                    RuleType.UNARY: [],
                    RuleType.BINARY: [],
                    RuleType.BINARY_PLUS: [],
                }

            if CKYParser.__is_terminal(rule):
                result[lhs][RuleType.TERMINAL].append(rule)
            elif CKYParser.__is_unary(rule):
                result[lhs][RuleType.UNARY].append(rule)
            elif CKYParser.__is_binary(rule):
                result[lhs][RuleType.BINARY].append(rule)
            elif CKYParser.__is_binary_plus(rule):
                result[lhs][RuleType.BINARY_PLUS].append(rule)

        return result

    @staticmethod
    def __is_terminal(rule: Production) -> bool:
        return rule.is_lexical() and len(rule.rhs()) == 1

    @staticmethod
    def __is_unary(rule: Production) -> bool:
        return rule.is_nonlexical() and len(rule.rhs()) == 1

    @staticmethod
    def __is_binary(rule: Production) -> bool:
        return rule.is_nonlexical() and len(rule.rhs()) == 2

    @staticmethod
    def __is_binary_plus(rule: Production) -> bool:
        return len(rule.rhs()) >= 2 and not CKYParser.__is_binary(rule)
