"""End-to-end learning runs on small targets."""
import importlib.util
import shutil
import unittest

from ltlf_learner import learn, parse
from ltlf_learner.environments import office_world
from ltlf_learner.hints import teacher_level1, wrong_level1

TARGETS = ("F(a)", "G(a)", "U(a,b)", "G(->(a,F(b)))", "F(&(a,X(b)))")
HAVE_Z3 = importlib.util.find_spec("z3") is not None
HAVE_MONA = importlib.util.find_spec("ltlf2dfa") is not None and shutil.which("mona") is not None


def run(target, **options):
    return learn(target, ("a", "b"), parse(target).size + 1, seed=41, **options)


class LearnerTests(unittest.TestCase):
    def test_exhaustive_rule_identifies_every_target(self):
        for target in TARGETS:
            for order in ("enumeration", "semantic", "semantic_far", "split"):
                r = run(target, order=order)
                self.assertTrue(r.equivalent, (target, order, r.hypothesis))
                self.assertEqual(r.status, "completed")
                self.assertGreaterEqual(r.queries, r.initial_queries)

    def test_top_k_rule_runs(self):
        r = run("G(->(a,F(b)))", rule="top_k", k=3)
        self.assertIsNotNone(r.hypothesis)
        self.assertLessEqual(r.comparisons, 100)

    def test_seeded_runs_are_reproducible(self):
        a, b = run("G(->(a,F(b)))", order="semantic"), run("G(->(a,F(b)))", order="semantic")
        self.assertEqual((str(a.hypothesis), a.queries), (str(b.hypothesis), b.queries))

    def test_single_atom_and_constant_target(self):
        r = learn("a", ("a",), 2, seed=0)
        self.assertTrue(r.equivalent)
        r = learn("G(a)", ("a",), 3, seed=0, sample_size=0)   # no initial sample at all
        self.assertTrue(r.equivalent)

    def test_hints(self):
        target = "G(->(a,F(b)))"
        self.assertIn("missing_behavior", teacher_level1(parse(target)))
        for options in ({"level1": "teacher"}, {"level1": "teacher", "level1_mode": "prune"}):
            self.assertTrue(run(target, **options).equivalent, options)
        pruned = run(target, level1="teacher", level1_mode="prune")
        self.assertEqual(pruned.levels, ["template"])
        self.assertLess(pruned.candidates, run(target).candidates)
        # A Level-2 requirement implied by the target.
        r = run("G(->(a,X(b)))", level2=["G(->(a,F(b)))"])
        self.assertTrue(r.equivalent)
        self.assertEqual(r.levels, ["level2"])

    def test_wrong_ranking_hint_is_harmless(self):
        target = parse("G(->(a,F(b)))")
        wrong = wrong_level1(target)
        self.assertIsNotNone(wrong)
        self.assertNotIn("missing_behavior", wrong)
        self.assertTrue(run(str(target), level1=wrong).equivalent)

    def test_environment(self):
        r = learn("F(&(f,F(g)))", ("f", "g", "n"), 6, seed=41, environment=office_world())
        self.assertTrue(r.equivalent)
        env = office_world().project(("f", "g", "n"))
        self.assertTrue(env.is_feasible(r.witness or (env.label(env.start),)))

    def test_certify(self):
        r = run("F(&(a,X(b)))", certify_tests=30)
        self.assertTrue(r.certification["passed"])
        self.assertEqual(r.certification["tests_used"], 30)

    @unittest.skipUnless(HAVE_Z3, "needs z3-solver")
    def test_sat_version_space(self):
        for target in ("F(a)", "U(a,b)", "G(->(a,F(b)))"):
            self.assertTrue(run(target, version_space="sat").equivalent, target)
        with self.assertRaises(ValueError):
            run("F(a)", version_space="sat", rule="top_k")

    @unittest.skipUnless(HAVE_MONA, "needs ltlf2dfa and mona")
    def test_automata_give_the_same_run(self):
        for target in ("U(a,b)", "G(->(a,F(b)))"):
            a, b = run(target), run(target, distinguish="automata")
            self.assertEqual((str(a.hypothesis), a.queries), (str(b.hypothesis), b.queries))


if __name__ == "__main__":
    unittest.main()
