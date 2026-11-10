"""Formulas, parsing, and the three ways of evaluating a formula on traces."""
import random
import unittest
from itertools import product

from ltlf_learner import Formula, holds, parse
from ltlf_learner.enumerate import Enumerator
from ltlf_learner.formula import canonical
from ltlf_learner.progression import Progressor, distinguish_by_progression
from ltlf_learner.traces import PackedSample, filter_consistent, normalize_trace, valuation_alphabet


class FormulaTests(unittest.TestCase):
    def test_parse_and_print(self):
        for text in ("G(->(a,F(b)))", "U(a,U(b,c))", "&(F(a),G(!b))", "X(true)", "!(F(a))"):
            self.assertEqual(str(parse(str(parse(text)))), str(parse(text)))

    def test_negated_atom_is_one_leaf(self):
        f = parse("G(!(a))")
        self.assertEqual(str(f), "G(!a)")
        self.assertEqual(f.size, 2)
        self.assertEqual(parse("!(!(a))"), parse("a"))

    def test_size_and_depth(self):
        f = parse("G(->(a,X(U(a,b))))")
        self.assertEqual((f.size, f.temporal_depth()), (7, 3))
        self.assertEqual(parse("a").temporal_depth(), 0)

    def test_bad_input(self):
        for text in ("G(a", "&(a)", "a b", "Q(a)", ""):
            with self.assertRaises(ValueError):
                parse(text)

    def test_canonical_ignores_operand_order(self):
        self.assertEqual(canonical(parse("&(a,b)")), canonical(parse("&(b,a)")))
        self.assertEqual(canonical(parse("->(a,b)")), canonical(parse("|(!a,b)")))
        self.assertNotEqual(canonical(parse("U(a,b)")), canonical(parse("U(b,a)")))


class SemanticsTests(unittest.TestCase):
    def test_empty_trace(self):
        self.assertTrue(holds(parse("G(a)"), ()))
        self.assertTrue(holds(parse("!(a)"), ()))
        self.assertFalse(holds(parse("F(a)"), ()))
        self.assertFalse(holds(parse("X(true)"), ()))

    def test_strong_next(self):
        self.assertFalse(holds(parse("X(a)"), (("a",),)))
        self.assertTrue(holds(parse("X(a)"), ((), ("a",))))

    def test_three_evaluators_agree(self):
        """holds, PackedSample and progression give the same truth value."""
        atoms = ("a", "b")
        alphabet = valuation_alphabet(atoms)
        traces = [t for n in range(4) for t in product(alphabet, repeat=n)]
        enum = Enumerator(atoms, 5, 2)
        for _ in range(4):
            enum.next_size()
        formulas = [f for s in range(1, 6) for f in enum.formulas_of_size(s)]
        rng = random.Random(1)
        sample = rng.sample(formulas, 300)
        progressor, memo = Progressor(), {}
        packed = PackedSample([(t, True) for t in traces])
        for f in sample:
            values = packed.value(f, memo)
            for t in traces[:40]:
                expected = holds(f, t)
                self.assertEqual(progressor.holds(f, t), expected, (f, t))
            truth = [holds(f, t) for t in traces]
            self.assertEqual(filter_consistent([f], list(zip(traces, truth))), [f])
            self.assertTrue(values & packed.starts or not any(truth))

    def test_distinguish_finds_shortest_witness(self):
        alphabet = valuation_alphabet(("a", "b"))
        witness = distinguish_by_progression(parse("U(a,b)"), parse("F(b)"), alphabet)
        self.assertEqual(witness, ((), ("b",)))
        self.assertIsNone(distinguish_by_progression(parse("F(F(a))"), parse("F(a)"), alphabet))
        self.assertIsNone(distinguish_by_progression(parse("->(a,b)"), parse("|(!a,b)"), alphabet))

    def test_normalize_trace(self):
        self.assertEqual(normalize_trace([{"b", "a"}, []], ("a", "b")), (("a", "b"), ()))
        with self.assertRaises(ValueError):
            normalize_trace([{"z"}], ("a",))


class EnumeratorTests(unittest.TestCase):
    def test_bounds_and_no_duplicates(self):
        enum = Enumerator(("a", "b"), 5, 2)
        for _ in range(4):
            enum.next_size()
        for size in range(1, 6):
            formulas = enum.formulas_of_size(size)
            self.assertEqual(len(formulas), len(set(formulas)))
            for f in formulas:
                self.assertEqual(f.size, size)
                self.assertLessEqual(f.temporal_depth(), 2)

    def test_redundant_forms_are_skipped(self):
        enum = Enumerator(("a",), 4, 2)
        for _ in range(3):
            enum.next_size()
        texts = {str(f) for s in range(1, 5) for f in enum.formulas_of_size(s)}
        self.assertIn("F(a)", texts)
        self.assertNotIn("F(F(a))", texts)
        self.assertNotIn("&(a,a)", texts)
        self.assertNotIn("|(a,!a)", texts)

    def test_single_atom_size_one(self):
        enum = Enumerator(("a",), 1, 0)
        self.assertEqual({str(f) for f in enum.formulas_of_size(1)}, {"a", "!a"})


if __name__ == "__main__":
    unittest.main()
