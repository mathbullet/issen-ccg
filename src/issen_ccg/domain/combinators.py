"""English CCG schemas, independently implemented from their category equations.

Clark & Curran (2007), Appendix A: https://aclanthology.org/J07-4004/
Category variables belong to an input category, so a feature bound in an argument
also appears in that category's result. The two operands have distinct variables.
"""

from dataclasses import dataclass, replace

from issen_ccg.domain.category import Category, parse_category


@dataclass(frozen=True)
class Combination:
    category: Category
    name: str
    head_child: int


class Unification:
    def __init__(self):
        self.leaders = [0, 1]
        self.features = ["", ""]

    def root(self, side):
        return self.leaders[side]

    def bind(self, side, feature):
        side = self.root(side)
        if self.features[side] and self.features[side] != feature:
            return False
        self.features[side] = feature
        return True

    def match(self, left, right):
        if left.coordinated != right.coordinated:
            return False
        if left.atom or right.atom:
            if left.atom != right.atom:
                return False
            lf = "" if left.feature in ("X", "nb") else left.feature
            rf = "" if right.feature in ("X", "nb") else right.feature
            if left.atom != "S":
                return not lf or not rf or lf == rf
            if lf and rf:
                return lf == rf
            if lf:
                return self.bind(1, lf)
            if rf:
                return self.bind(0, rf)
            a, b = self.root(0), self.root(1)
            if self.features[a] and self.features[b] and self.features[a] != self.features[b]:
                return False
            self.features[a] = self.features[a] or self.features[b]
            self.leaders[b] = a
            return True
        return (
            left.slash == right.slash
            and self.match(left.result, right.result)
            and self.match(left.argument, right.argument)
        )

    def apply(self, category, side):
        if category.atom:
            feature = category.feature
            if category.atom == "S" and feature in ("", "X"):
                feature = self.features[self.root(side)]
            return replace(category, feature=feature)
        return replace(
            category,
            result=self.apply(category.result, side),
            argument=self.apply(category.argument, side),
        )


def combine(left: str, right: str):
    try:
        lhs, rhs = parse_category(left), parse_category(right)
    except ValueError:
        return ()
    output = []

    def add(category, name, head):
        if category.atom == "NP" and category.feature == "nb":
            category = replace(category, feature="")
        value = Combination(category, name, head)
        if value not in output:
            output.append(value)

    punctuation = {",", ".", ":", ";", "LRB", "RRB", "LQU", "RQU"}
    if rhs.coordinated and lhs.atom in punctuation | {"conj"}:
        add(rhs, "lp" if lhs.atom in punctuation else "conj", 1)
        return tuple(output)
    if rhs.coordinated and not lhs.coordinated:
        unifier = Unification()
        if unifier.match(lhs, replace(rhs, coordinated=False)):
            add(unifier.apply(lhs, 0), "conj", 0)
        return tuple(output)
    if lhs.coordinated or rhs.coordinated:
        return ()
    if lhs.atom in punctuation:
        add(rhs, "lp", 1)
        if lhs.atom in (",", ";") and rhs.atom not in punctuation | {"conj"}:
            add(replace(rhs, coordinated=True), "conj", 1)
    if rhs.atom in punctuation:
        add(lhs, "rp", 0)
    if lhs.atom == "conj" and rhs.atom not in punctuation | {"conj"}:
        add(replace(rhs, coordinated=True), "conj", 1)
        if rhs.atom == "N":
            add(rhs, "conj", 1)
    special = {
        ("NP", ","): ("S/S",),
        (",", "NP"): (r"(S\NP)\(S\NP)",),
        ("S[dcl]/S[dcl]", ","): ("S/S", r"(S\NP)\(S\NP)", r"(S\NP)/(S\NP)", r"S\S"),
        (r"S[dcl]\S[dcl]", ","): ("S/S",),
    }
    for category in special.get((left, right), ()):
        add(parse_category(category), "rp" if right == "," else "lp", 0 if right == "," else 1)
    if lhs.atom == rhs.atom == "NP":
        add(parse_category("NP"), "appo", 0)
    if left == right == "S[dcl]":
        add(lhs, "appo", 0)

    if lhs.slash == "/":
        unifier = Unification()
        if unifier.match(lhs.argument, rhs):
            result = rhs if lhs.argument == lhs.result else unifier.apply(lhs.result, 0)
            add(result, "fa", 0)
    if rhs.slash == "\\":
        unifier = Unification()
        if unifier.match(lhs, rhs.argument):
            result = lhs if rhs.argument == rhs.result else unifier.apply(rhs.result, 1)
            add(result, "ba", 1)

    def compose(primary, spine, direction, crossed=False):
        """Bind shared Y and rebuild X/Y + Y|Z or Y|Z + X\\Y up to degree three."""
        arguments = []
        cursor = spine
        for degree in range(1, 4):
            if cursor.slash != direction:
                break
            arguments.append((cursor.slash, cursor.argument))
            cursor = cursor.result
            if degree > 1:
                if (
                    direction != "/"
                    or cursor.slash != "\\"
                    or cursor.result.atom != "S"
                    or cursor.argument.atom != "NP"
                    or cursor.result.feature in ("", "X")
                ):
                    continue
            if crossed and cursor.atom in ("N", "NP"):
                continue
            unifier = Unification()
            if primary is lhs:
                matched = unifier.match(primary.argument, cursor)
                primary_side, spine_side = 0, 1
                name, head = "fc", 0
            else:
                matched = unifier.match(cursor, primary.argument)
                primary_side, spine_side = 1, 0
                name, head = ("bx" if crossed else "bc"), 1
            if matched:
                if primary.result == primary.argument:
                    add(spine, name if degree == 1 else f"g{name}", head)
                    continue
                result = unifier.apply(primary.result, primary_side)
                for slash, argument in reversed(arguments):
                    result = Category(
                        result=result, slash=slash, argument=unifier.apply(argument, spine_side)
                    )
                add(result, name if degree == 1 else f"g{name}", head)

    if lhs.slash == "/" and rhs.slash == "/":
        compose(lhs, rhs, "/")
    if rhs.slash == "\\" and lhs.slash == "\\" and rhs.argument.atom not in ("N", "NP"):
        compose(rhs, lhs, "\\")
    if rhs.slash == "\\" and lhs.slash == "/":
        compose(rhs, lhs, "/", crossed=True)
    if left == r"(S[dcl]\S[dcl])\NP" and right == r"S\S":
        add(lhs, "gbc", 0)
    return tuple(output)


def unary_schemas():
    rules = [("N", "NP", "lex")]
    vp_modifier = r"(S\NP)\(S\NP)"
    for feature in ("pss", "ng", "adj", "to", "dcl"):
        source = rf"S[{feature}]\NP"
        rules.append((source, r"NP\NP", "lex"))
        if feature != "dcl":
            rules.extend((source, target, "lex") for target in (vp_modifier, "S/S"))
    rules.extend(
        [
            (r"S[ng]\NP", "NP", "lex"),
            (r"S[ng]\NP", r"S\S", "lex"),
            (r"S[ng]\NP", r"(S\NP)/(S\NP)", "lex"),
            (r"S[to]\NP", r"N\N", "lex"),
            ("S[dcl]/NP", r"NP\NP", "lex"),
            (r"(S[to]\NP)/NP", r"NP\NP", "lex"),
            ("S[dcl]", r"NP\NP", "lex"),
            ("S[dcl]", r"S\S", "lex"),
            ("NP", r"NP/(NP\NP)", "lex"),
            ("NP", r"S/(S\NP)", "tr"),
            ("NP", "S/(S/NP)", "tr"),
            ("NP", r"(S\NP)\((S\NP)/NP)", "tr"),
            ("NP", r"((S\NP)/NP)\(((S\NP)/NP)/NP)", "tr"),
        ]
    )
    for argument in ("PP", r"S[to]\NP", r"S[adj]\NP"):
        target = rf"((S\NP)/({argument}))\(((S\NP)/({argument}))/NP)"
        rules.append(("NP", target, "tr"))
    for source in ("PP", r"S[adj]\NP", r"S[to]\NP"):
        rules.append((source, rf"(S\NP)\((S\NP)/({source}))", "tr"))
    return tuple(
        (parse_category(source), parse_category(target), name) for source, target, name in rules
    )
