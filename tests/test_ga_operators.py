"""Tests for the NSGA-II building blocks in configure11_functions.py."""
import numpy as np
import pytest

import configure11_functions as cf

# Six solutions with two objectives (both minimised).
#   0:(1,5)  1:(2,3)  2:(4,1)  -> non-dominated (front 0)
#   3:(3,4) is dominated by 1; 5:(2,6) by 0 and 1 -> front 1
#   4:(5,5) is also dominated by 3 -> front 2
OBJS = np.array([[1, 5], [2, 3], [4, 1], [3, 4], [5, 5], [2, 6]], dtype=float)


def test_initial_pop_is_binary_with_baseline_and_full_solutions():
    np.random.seed(0)
    pop = cf.initial_pop(20, 7)
    assert pop.shape == (20, 7)
    assert set(np.unique(pop)) <= {0, 1}
    assert not pop[-2].any()       # no intervention (baseline)
    assert pop[-1].all()           # every intervention


def test_remove_duplicate_same_population():
    pop = np.array([[0, 1], [1, 1], [0, 1], [1, 0], [1, 1]])
    unique, removed = cf.remove_duplicate_same_population(pop)
    np.testing.assert_array_equal(unique, [[0, 1], [1, 1], [1, 0]])
    assert sorted(removed) == [2, 4]
    np.testing.assert_array_equal(pop[2], [0, 1])  # input left unchanged


def test_remove_duplicate_list_returns_removed_rows():
    records = np.array([[1.0, 2.0], [3.0, 4.0], [1.0, 2.0]])
    unique, removed = cf.remove_duplicate_list(records)
    np.testing.assert_array_equal(unique, [[1.0, 2.0], [3.0, 4.0]])
    assert removed == [2]


def test_remove_duplicate_different_population():
    offspring = np.array([[0, 0, 1], [1, 1, 1], [0, 1, 0]])
    simulated = np.array([[1, 1, 1], [1, 0, 0]])
    np.testing.assert_array_equal(
        cf.remove_duplicate_different_population(offspring, simulated),
        [[0, 0, 1], [0, 1, 0]])


def test_remove_same_objectives_population():
    pop = np.arange(12).reshape(4, 3)
    np.testing.assert_array_equal(
        cf.remove_same_objectives_population(pop, [1, 3]), [[0, 1, 2], [6, 7, 8]])
    np.testing.assert_array_equal(cf.remove_same_objectives_population(pop, []), pop)


def test_non_dominated_sorting_known_fronts():
    fronts = cf.non_dominated_sorting(len(OBJS), OBJS)
    assert fronts == {0: [0, 1, 2], 1: [3, 5], 2: [4]}


def test_crowding_distance_boundaries_and_interior():
    fronts = cf.non_dominated_sorting(len(OBJS), OBJS)
    distance = cf.calculate_crowding_distance(fronts, OBJS)
    # Front 0: solutions 0 and 2 are the extremes of both objectives.
    assert distance[0] == {0: 666666666 + 333333333, 1: 2.0, 2: 666666666 + 333333333}
    # A single-member front is a boundary for both objectives.
    assert distance[2] == {4: 666666666 + 333333333}


def test_fitness_sort_orders_by_front_then_distance():
    fronts = cf.non_dominated_sorting(len(OBJS), OBJS)
    order = cf.fitness_sort(cf.calculate_crowding_distance(fronts, OBJS), len(OBJS))
    assert set(order[:2]) == {0, 2}    # boundary solutions of front 0 first
    assert order[2] == 1
    assert set(order[3:5]) == {3, 5}
    assert order[5] == 4


@pytest.mark.parametrize("seed", range(20))
def test_fitter_parent_always_returns_the_fitter_of_two(seed):
    # With two solutions, the tournament always compares 0 and 1;
    # solution 1 is listed first, so it is the fitter one.
    np.random.seed(seed)
    assert cf.fitter_parent(np.array([1, 0]), 2) == 1


def test_crossover_swaps_tails_of_parents():
    np.random.seed(1)
    pop = np.array([[0] * 10, [1] * 10])
    child_1, child_2 = cf.crossover_random_single_point_swap(pop, 0, 1, 1, 9)
    for child, head, tail in ((child_1, 0, 1), (child_2, 1, 0)):
        assert child.shape == (10,)
        cut = int(np.argmax(child != head))
        assert 1 <= cut < 9
        assert (child[:cut] == head).all() and (child[cut:] == tail).all()
    np.testing.assert_array_equal(pop, [[0] * 10, [1] * 10])  # parents unchanged


def test_crossover_is_reproducible_with_a_seed():
    pop = np.random.RandomState(0).randint(2, size=(6, 12))
    np.random.seed(42)
    first = cf.crossover_random_single_point_swap(pop, 1, 4, 1, 11)
    np.random.seed(42)
    second = cf.crossover_random_single_point_swap(pop, 1, 4, 1, 11)
    for a, b in zip(first, second):
        np.testing.assert_array_equal(a, b)


def test_mutation_with_zero_probability_keeps_different_children():
    c1, c2 = np.array([0, 0, 0, 0]), np.array([1, 1, 1, 1])
    out_1, out_2 = cf.mutation_random_bitflip(c1.copy(), c2.copy(), 4, 0.0, 4)
    np.testing.assert_array_equal(out_1, c1)
    np.testing.assert_array_equal(out_2, c2)


def test_mutation_with_certain_probability_flips_one_gene_each():
    np.random.seed(3)
    c1, c2 = np.zeros(8, dtype=int), np.ones(8, dtype=int)
    out_1, out_2 = cf.mutation_random_bitflip(c1.copy(), c2.copy(), 8, 1.0, 8)
    assert out_1.sum() == 1          # one 0 became 1
    assert out_2.sum() == 7          # one 1 became 0
    assert set(np.unique(np.concatenate((out_1, out_2)))) <= {0, 1}


def test_mutation_makes_identical_children_different():
    np.random.seed(5)
    child = np.array([1, 0, 1, 0, 1])
    out_1, out_2 = cf.mutation_random_bitflip(child.copy(), child.copy(), 5, 0.0, 5)
    assert not np.array_equal(out_1, out_2)


def test_separate_new_old():
    old = np.array([[0, 0, 1], [1, 1, 0], [1, 0, 1]])
    new = np.array([[1, 1, 0], [0, 1, 1], [0, 0, 1]])
    new_chroms, old_chroms, old_index = cf.separate_new_old(new, old)
    np.testing.assert_array_equal(new_chroms, [[0, 1, 1]])
    np.testing.assert_array_equal(old_chroms, [[0, 0, 1], [1, 1, 0]])
    assert old_index == [2, 0]
