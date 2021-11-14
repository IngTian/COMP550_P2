import random

from nltk import CFG
from nltk.grammar import Production, Nonterminal
from typing import List, Dict, Union, Tuple


class CGFStatement:
    rhs: Tuple[Union[Nonterminal, str]]
    lhs: Nonterminal
    is_terminal: bool


class CNFStatement:
    rhs: Tuple[Union[Nonterminal, str]]
    lhs: Nonterminal
    is_terminal: bool
    map: "CGFToCNFMap"


class CGFToCNFMap:
    cgf: List[List[CGFStatement]]
    cnf: List[CNFStatement]


class CKYDPBlockEntry:
    cnf_statement: CNFStatement
    is_terminal: bool
    descendents: List["CKYDPBlockEntry"]


class CNFContainer:

    def __init__(self, cnf_map: Dict[Nonterminal, Dict[str, List[CGFToCNFMap]]]):
        pass


def clone_map(m: CGFToCNFMap) -> CGFToCNFMap:
    chain_of_cnf: List[CNFStatement] = list()
    list_of_chain_of_cgf: List[List[CGFStatement]] = list()

    for cnf in m.cnf:
        new_cnf = CNFStatement()
        new_cnf.rhs = cnf.rhs
        new_cnf.lhs = cnf.lhs
        new_cnf.is_terminal = cnf.is_terminal
        chain_of_cnf.append(new_cnf)

    for chain in m.cgf:
        new_chain = chain.copy()
        list_of_chain_of_cgf.append(new_chain)

    new_map = CGFToCNFMap()
    new_map.cgf = list_of_chain_of_cgf
    new_map.cnf = chain_of_cnf

    return new_map


class CKYParser:

    def __init__(self, grammar: CFG):
        self.__original_cgf = grammar
        self.__formatted_cgf = self.__format_grammar()
        self.__cnf_container = self.__parse_cnf()

    def __format_grammar(self) -> Dict[Nonterminal, List[CGFStatement]]:
        """
        Convert the raw CGF grammar into a map.
        :return: A map of rules.
        """
        rules: List[Production] = self.__original_cgf.productions()
        result: Dict[Nonterminal, List[CGFStatement]] = dict()

        for rule in rules:
            new_cgf = CGFStatement()
            new_cgf.lhs, new_cgf.rhs = rule.lhs(), rule.rhs()
            new_cgf.is_terminal = True if len(new_cgf.rhs) == 1 and type(new_cgf.rhs[0]) == str else False

            # Add the new CGF to the map.
            if new_cgf.lhs in result.keys():
                result.get(new_cgf.lhs).append(new_cgf)
            else:
                result[new_cgf.lhs] = [new_cgf]

        return result

    def __parse_cnf(self) -> CNFContainer:
        """
        Convert the formatted CGF into
        CNFs.
        :return: A CNF Container which contains all CNF statements.
        """
        result: List[CNFStatement] = list()
        cgf_rules_lookup_table: Dict[Nonterminal, Dict[str, List[CGFStatement]]] = dict()
        cgf_cnf_map_lookup_table: Dict[Nonterminal, Dict[str, List[CGFToCNFMap]]] = dict()

        def extract_rules(rules: List[CGFStatement]) -> Dict[str, List[CGFStatement]]:
            """
            Extract various CGF rules into 4 categories.
            1. Terminal
            2. Unary
            3. Binary: Both of RHS nodes has to be Non-terminal.
            4. Binary+: Everything else.
            :param rules: CGF rules.
            :return: Organized rules.
            """
            temp: Dict[str, List[CGFStatement]] = {
                "terminal": [],
                "unary": [],
                "binary": [],
                "binary+": [],
            }

            for rule in rules:
                if rule.is_terminal:
                    temp["terminal"].append(rule)
                elif len(rule.rhs) == 1 and type(rule.rhs[0]) == Nonterminal:
                    temp["unary"].append(rule)
                elif len(rule.rhs) == 2 and type(rule.rhs[0]) == Nonterminal and type(rule.rhs[0]) == Nonterminal:
                    temp["binary"].append(rule)
                else:
                    temp["binary+"].append(rule)

            return temp

        def build_cgf_cnf_map(starting_point: Nonterminal) -> Dict[str, List[CGFToCNFMap]]:

            if starting_point in cgf_cnf_map_lookup_table.keys():
                return cgf_cnf_map_lookup_table.get(starting_point)

            cgf_rules = cgf_rules_lookup_table.get(starting_point)
            terminal = cgf_rules["terminal"]
            unary = cgf_rules["unary"]
            binary = cgf_rules["binary"]
            binary_plus = cgf_rules["binary_plus"]

            terminal_cnf_maps: List[CGFToCNFMap] = list()
            unary_cnf_maps: List[CGFToCNFMap] = list()
            binary_cnf_maps: List[CGFToCNFMap] = list()
            binary_plus_cnf_maps: List[CGFToCNFMap] = list()

            # Build terminal, binary, and binary+ rules first.
            for terminal_rule in terminal:
                new_map, cnf = CGFToCNFMap(), CNFStatement()
                cnf.lhs = terminal_rule.lhs, cnf.rhs = terminal_rule.rhs
                cnf.map = new_map
                new_map.cgf.append([terminal_rule])
                new_map.cnf.append(cnf)
                terminal_cnf_maps.append(new_map)

            for binary_rule in binary:
                new_map, cnf = CGFToCNFMap(), CNFStatement()
                cnf.lhs = binary_rule.lhs, cnf.rhs = binary_rule.rhs
                cnf.map = new_map
                new_map.cgf.append([binary_rule])
                new_map.cnf.append(cnf)
                binary_cnf_maps.append(new_map)

            for binary_plus_rule in binary_plus:
                new_map = CGFToCNFMap()
                new_map.cgf.append([binary_plus_rule])
                cnf_rules = build_binary_plus_cnf(binary_plus_rule)
                new_map.cnf = cnf_rules
                for cnf in cnf_rules:
                    cnf.map = new_map
                binary_plus_cnf_maps.append(new_map)

            # Handle unary rules.
            for unary_rule in unary:
                rhs = unary_rule.rhs
                assert isinstance(rhs, Nonterminal)
                rhs_cnf = build_cgf_cnf_map(rhs)

                unary_cnf = rhs_cnf.get("unary")
                terminal_cnf = rhs_cnf.get("terminal")
                binary_cnf = rhs_cnf.get("binary")
                binary_plus_cnf = rhs_cnf.get("binary+")

                # Concatenate terminal
                for terminal_map in terminal_cnf:
                    cloned_map = clone_map(terminal_map)
                    cloned_map.cgf[0][0].lhs = starting_point
                    cloned_map.cnf[0].lhs = starting_point
                    unary_cnf_maps.append(cloned_map)

                # Concatenate binary
                for binary_map in binary_cnf:
                    cloned_map = clone_map(binary_map)
                    cloned_map.cgf[0][0].lhs = starting_point
                    cloned_map.cnf[0].lhs = starting_point
                    unary_cnf_maps.append(cloned_map)

                # Concatenate binary+
                for binary_plus_map in binary_plus_cnf:
                    cloned_map = clone_map(binary_plus_map)
                    cloned_map.cgf[0][0].lhs = starting_point
                    cloned_map.cnf[0].lhs = starting_point
                    unary_cnf_maps.append(cloned_map)

                # Concatenate unary
                for unary_map in unary_cnf:
                    cloned_map = clone_map(unary_map)
                    for list_of_chain_cgf in cloned_map.cgf:
                        list_of_chain_cgf.insert(0, unary_rule)
                    cloned_map.cnf[0].lhs = starting_point
                    unary_cnf_maps.append(cloned_map)

            return {
                "terminal": terminal_cnf_maps,
                "unary": unary_cnf_maps,
                "binary": binary_cnf_maps,
                "binary+": binary_plus_cnf_maps
            }

        def build_binary_plus_cnf(statement: Union[CGFStatement, CNFStatement]) -> List[CNFStatement]:
            lhs, rhs = statement.lhs, statement.rhs
            result_cnf = CNFStatement()
            result_cnf.lhs = lhs

            # If the rhs consists of two terminal nodes, simply return.
            if len(rhs) == 2 and type(rhs[0]) == Nonterminal and type(rhs[1]) == Nonterminal:
                result_cnf.rhs = rhs
                return [result_cnf]
            # If the rhs is a terminal node, simply return.
            elif len(rhs) == 1 and type(rhs[0]) == str:
                result_cnf.rhs = rhs
                return [result_cnf]
            # If the rhs is binary_plus, do the following.
            else:
                rhs_length = len(rhs)
                rhs_first_half = rhs[:rhs_length - 1]
                rhs_first_half_str = "-".join(map(lambda s: str(s), rhs_first_half))
                rhs_first_half_symbol = Nonterminal(
                    f'CNF-Symbol-From-{str(lhs)}-To-{rhs_first_half_str}-{random.randint(0, 10000)}')
                first_half_temp_cnf = CNFStatement()
                first_half_temp_cnf.lhs = rhs_first_half_symbol
                first_half_temp_cnf.rhs = rhs_first_half
                first_half_rules = build_binary_plus_cnf(first_half_temp_cnf)
                if type(rhs[-1]) == Nonterminal:
                    result_cnf.rhs = [rhs_first_half_symbol, rhs[-1]]
                    return [result_cnf] + first_half_rules
                else:
                    second_half_rule = CNFStatement()
                    second_half_rule.rhs = rhs[-1]
                    second_half_rule.lhs = Nonterminal(
                        f'CNF-Symbol-From-{str(lhs)}-To-{rhs[-1]}-{random.randint(0, 10000)}')
                    result_cnf.rhs = [rhs_first_half_symbol, second_half_rule.lhs]
                    return [result_cnf] + first_half_rules + [second_half_rule]

        # Sort all rules.
        for cgf_lhs in self.__formatted_cgf.keys():
            cgf_rules_lookup_table[cgf_lhs] = extract_rules(self.__formatted_cgf.get(cgf_lhs))

        for node in self.__formatted_cgf.keys():
            build_cgf_cnf_map(node)

        new_container = CNFContainer(cgf_cnf_map_lookup_table)

        return new_container
