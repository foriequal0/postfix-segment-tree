//! # Skip Indexing
//!
//! Postfix Segment Tree stores sums of 2^*n* elements in its nodes.
//! We can find an integer sequence `w` that `x = sum(2 ^ w[n])` for every integer `x`.
//! It means that we can calculate the sum of `c` elements using a

use crate::internal::node_id::{LeafNodeId, NodeId};

pub(crate) struct SkippingIterator {
    index: usize,
    end: usize,
}

impl SkippingIterator {
    pub(crate) fn new(len: usize) -> Self {
        Self { index: 0, end: len }
    }

    pub(crate) fn skip_to_pivot(&mut self, index: usize) -> usize {
        debug_assert!(index >= self.index);
        debug_assert!(index <= self.end);

        let index = get_pivot(index, self.end);
        self.index = index;
        index
    }
}

// TODO: PROOF let pivot = get_pivot(index, end), index >= min_reachable_index_for_elements(pivot), pivot >= index
fn get_pivot(index: usize, end: usize) -> usize {
    debug_assert!(index <= end);

    let mut i = 0;
    while i < index {
        let leaf_node_id = LeafNodeId::new(i);
        let node_id = step_skipping_iterator(end, leaf_node_id).unwrap();
        i = node_id.index() + 1;
    }

    i
}

impl Iterator for SkippingIterator {
    type Item = NodeId;

    fn next(&mut self) -> Option<Self::Item> {
        let leaf_node_id = LeafNodeId::new(self.index);
        if let Some(node_id) = step_skipping_iterator(self.end, leaf_node_id) {
            self.index = node_id.index() + 1;
            Some(node_id)
        } else {
            None
        }
    }
}

fn step_skipping_iterator(elements: usize, id: LeafNodeId) -> Option<NodeId> {
    debug_assert!(elements <= max_reachable_elements_for_current_index(id.index()));
    if id.index() >= elements {
        return None;
    }

    let offset = elements - id.index();
    let level = offset.ilog2();
    let width = 1 << level;
    let index = id.index() + width - 1;

    Some(NodeId::new(index, level))
}

fn max_reachable_elements_for_current_index(index: usize) -> usize {
    if index == 0 {
        return usize::MAX;
    }

    // TODO: proof
    index + ((1 << index.trailing_zeros() as usize) - 1)
}

pub(crate) struct IncreasingSkippingIterator {
    index: usize,
    end: usize,
}

impl IncreasingSkippingIterator {
    pub(crate) fn new(index: usize, end: usize) -> Self {
        debug_assert!(index >= min_reachable_index_for_elements(end));

        Self { index, end }
    }
}

impl Iterator for IncreasingSkippingIterator {
    type Item = NodeId;
    fn next(&mut self) -> Option<Self::Item> {
        let leaf_node_id = LeafNodeId::new(self.index);
        if let Some(node_id) = step_increasing_skipping_iterator(self.end, leaf_node_id) {
            self.index = node_id.index() + 1;
            Some(node_id)
        } else {
            None
        }
    }
}

fn step_increasing_skipping_iterator(elements: usize, id: LeafNodeId) -> Option<NodeId> {
    debug_assert!(id.index() >= min_reachable_index_for_elements(id.index()));
    if id.index() >= elements {
        return None;
    }

    let offset = elements - id.index();
    let level = offset.trailing_zeros();
    let width = 1 << level;
    let index = id.index() + width - 1;
    Some(NodeId::new(index, level))
}

fn min_reachable_index_for_elements(elements: usize) -> usize {
    if elements == 0 {
        return 0;
    }

    // TODO: proof
    elements - (1 << elements.trailing_zeros() as usize)
}

#[cfg(test)]
mod tests {
    //! These tests are Deliberately exhaustive to visualize the pattern
    use super::*;

    fn id(index: usize, level: u32) -> NodeId {
        NodeId::new(index, level)
    }

    #[test]
    fn test_skipping_iterator() {
        fn iter(len: usize) -> Vec<NodeId> {
            SkippingIterator::new(len).collect()
        }

        assert_eq!(iter(0), vec![]);
        assert_eq!(
            iter(1),
            vec![
                id(0, 0), // 1st
            ]
        );
        assert_eq!(
            iter(2),
            vec![
                id(1, 1), // 2nd
            ]
        );
        assert_eq!(
            iter(3),
            vec![
                id(1, 1), // 2nd
                id(2, 0), // +1
            ]
        );
        assert_eq!(
            iter(4),
            vec![
                id(3, 2), // 4th
            ]
        );
        assert_eq!(
            iter(5),
            vec![
                id(3, 2), // 4th
                id(4, 0), // +1
            ]
        );
        assert_eq!(
            iter(6),
            vec![
                id(3, 2), // 4th
                id(5, 1), // +2
            ]
        );
        assert_eq!(
            iter(7),
            vec![
                id(3, 2), // 4th,
                id(5, 1), // + 2
                id(6, 0), // + 1
            ]
        );
        assert_eq!(
            iter(8),
            vec![
                id(7, 3), // 8th
            ]
        );
    }

    #[test]
    fn test_skipping_iterator_levels_monotonically_decreasing() {
        fn get_first_non_monotonically_decreasing(iter: &mut SkippingIterator) -> Option<NodeId> {
            let mut prev_level = None;
            while let Some(node_id) = iter.next() {
                if let Some(prev_level) = prev_level {
                    if node_id.level() >= prev_level {
                        return Some(node_id);
                    }
                }

                prev_level = Some(node_id.level());
            }

            None
        }

        const N: usize = 100;
        for i in 0..N {
            let mut iter = SkippingIterator::new(i);
            let first = get_first_non_monotonically_decreasing(&mut iter);
            assert_eq!(first, None);
        }
    }

    #[test]
    fn test_skip_to_pivot_index() {
        fn get(index: usize, end: usize) -> usize {
            get_pivot(index, end)
        }

        assert_eq!(get(0, 0), 0);

        assert_eq!(get(0, 1), 0);
        assert_eq!(get(1, 1), 1); // +1

        assert_eq!(get(0, 2), 0);
        assert_eq!(get(1, 2), 2); // +2
        assert_eq!(get(2, 2), 2);

        assert_eq!(get(0, 3), 0);
        assert_eq!(get(1, 3), 2); // +2
        assert_eq!(get(2, 3), 2);
        assert_eq!(get(3, 3), 3); // +1

        assert_eq!(get(0, 4), 0);
        assert_eq!(get(1, 4), 4); // +4
        assert_eq!(get(2, 4), 4);
        assert_eq!(get(3, 4), 4);
        assert_eq!(get(4, 4), 4);

        assert_eq!(get(0, 5), 0);
        assert_eq!(get(1, 5), 4); // +4
        assert_eq!(get(2, 5), 4);
        assert_eq!(get(3, 5), 4);
        assert_eq!(get(4, 5), 4);
        assert_eq!(get(5, 5), 5); // +1

        assert_eq!(get(0, 6), 0);
        assert_eq!(get(1, 6), 4); // +4
        assert_eq!(get(2, 6), 4);
        assert_eq!(get(3, 6), 4);
        assert_eq!(get(4, 6), 4);
        assert_eq!(get(5, 6), 6); // +2
        assert_eq!(get(6, 6), 6);

        assert_eq!(get(0, 7), 0);
        assert_eq!(get(1, 7), 4); // +4
        assert_eq!(get(2, 7), 4);
        assert_eq!(get(3, 7), 4);
        assert_eq!(get(4, 7), 4);
        assert_eq!(get(5, 7), 6); // +2
        assert_eq!(get(6, 7), 6);
        assert_eq!(get(7, 7), 7); // +1

        assert_eq!(get(0, 8), 0);
        assert_eq!(get(1, 8), 8); // +8
        assert_eq!(get(2, 8), 8);
        assert_eq!(get(3, 8), 8);
        assert_eq!(get(4, 8), 8);
        assert_eq!(get(5, 8), 8);
        assert_eq!(get(6, 8), 8);
        assert_eq!(get(7, 8), 8);
        assert_eq!(get(8, 8), 8);
    }

    #[test]
    fn test_min_index_of_pivot_should_be_greater_than_or_equal_to_index() {
        const N: usize = 100;
        for index in 0..N {
            for end in index..N {
                let pivot = get_pivot(index, end);
                let min_reachable_index = min_reachable_index_for_elements(pivot);

                assert!(index <= pivot);
                assert!(min_reachable_index <= index);
            }
        }
    }

    #[test]
    fn test_increasing_skip_indexing_iterator() {
        fn iter(index: usize, end: usize) -> Vec<NodeId> {
            IncreasingSkippingIterator::new(index, end).collect()
        }

        assert_eq!(iter(0, 0), vec![]);

        assert_eq!(iter(0, 1), vec![id(0, 0)]);
        assert_eq!(iter(1, 1), vec![]);

        assert_eq!(iter(0, 2), vec![id(1, 1)]);
        assert_eq!(iter(1, 2), vec![id(1, 0)]);
        assert_eq!(iter(2, 2), vec![]);

        assert_eq!(iter(2, 3), vec![id(2, 0)]);
        assert_eq!(iter(3, 3), vec![]);

        assert_eq!(iter(0, 4), vec![id(3, 2)]);
        assert_eq!(iter(1, 4), vec![id(1, 0), id(3, 1)]);
        assert_eq!(iter(2, 4), vec![id(3, 1)]);
        assert_eq!(iter(3, 4), vec![id(3, 0)]);
        assert_eq!(iter(4, 4), vec![]);

        assert_eq!(iter(4, 5), vec![id(4, 0)]);
        assert_eq!(iter(5, 5), vec![]);

        assert_eq!(iter(4, 6), vec![id(5, 1)]);
        assert_eq!(iter(5, 6), vec![id(5, 0)]);
        assert_eq!(iter(6, 6), vec![]);

        assert_eq!(iter(6, 7), vec![id(6, 0)]);
        assert_eq!(iter(7, 7), vec![]);

        assert_eq!(
            iter(0, 8),
            vec![
                id(7, 3), // 8th
            ]
        );
        assert_eq!(
            iter(1, 8),
            vec![
                id(1, 0), // 1st
                id(3, 1), // +2
                id(7, 2), // +4
            ]
        );
        assert_eq!(
            iter(2, 8),
            vec![
                id(3, 1), // 2nd
                id(7, 2), // +4
            ]
        );
        assert_eq!(
            iter(3, 8),
            vec![
                id(3, 0), // 1st
                id(7, 2), // +4
            ]
        );
        assert_eq!(
            iter(4, 8),
            vec![
                id(7, 2), // 4th
            ]
        );
        assert_eq!(
            iter(5, 8),
            vec![
                id(5, 0), // 1st
                id(7, 1), // +2
            ]
        );
        assert_eq!(
            iter(6, 8),
            vec![
                id(7, 1), // 2nd
            ]
        );
        assert_eq!(
            iter(7, 8),
            vec![
                id(7, 0), // 1st
            ]
        );
        assert_eq!(iter(8, 8), vec![]);
    }

    #[test]
    fn test_increasing_skipping_iterator_levels_monotonically_increasing() {
        fn get_first_non_monotonically_increasing(
            iter: &mut IncreasingSkippingIterator,
        ) -> Option<NodeId> {
            let mut prev_level = None;
            while let Some(node_id) = iter.next() {
                if let Some(prev_level) = prev_level {
                    if node_id.level() <= prev_level {
                        return Some(node_id);
                    }
                }

                prev_level = Some(node_id.level());
            }

            None
        }

        const N: usize = 100;
        for elements in 0..N {
            let min_index = min_reachable_index_for_elements(elements);
            for i in min_index..N {
                let mut iter = IncreasingSkippingIterator::new(i, elements);
                let first = get_first_non_monotonically_increasing(&mut iter);
                assert_eq!(first, None);
            }
        }
    }

    #[test]
    fn test_combined_iterator() {
        fn iter(index: usize, end: usize) -> (Vec<NodeId>, Vec<NodeId>) {
            let mut iter = SkippingIterator::new(end);
            let pivot = iter.skip_to_pivot(index);
            let first = IncreasingSkippingIterator::new(index, pivot).collect();
            let second = iter.collect();
            (first, second)
        }

        assert_eq!(iter(0, 0), (vec![], vec![]));

        assert_eq!(iter(0, 1), (vec![], vec![id(0, 0)]));
        assert_eq!(iter(1, 1), (vec![], vec![]));

        assert_eq!(iter(0, 2), (vec![], vec![id(1, 1)]));
        assert_eq!(iter(1, 2), (vec![id(1, 0)], vec![]));
        assert_eq!(iter(2, 2), (vec![], vec![]));

        assert_eq!(iter(0, 3), (vec![], vec![id(1, 1), id(2, 0)]));
        assert_eq!(iter(1, 3), (vec![id(1, 0)], vec![id(2, 0)]));
        assert_eq!(iter(2, 3), (vec![], vec![id(2, 0)]));
        assert_eq!(iter(3, 3), (vec![], vec![]));

        assert_eq!(iter(0, 4), (vec![], vec![id(3, 2)]));
        assert_eq!(iter(1, 4), (vec![id(1, 0), id(3, 1)], vec![]));
        assert_eq!(iter(2, 4), (vec![id(3, 1)], vec![]));
        assert_eq!(iter(3, 4), (vec![id(3, 0)], vec![]));
        assert_eq!(iter(4, 4), (vec![], vec![]));

        assert_eq!(iter(0, 5), (vec![], vec![id(3, 2), id(4, 0)]));
        assert_eq!(iter(1, 5), (vec![id(1, 0), id(3, 1)], vec![id(4, 0)]));
        assert_eq!(iter(2, 5), (vec![id(3, 1)], vec![id(4, 0)]));
        assert_eq!(iter(3, 5), (vec![id(3, 0)], vec![id(4, 0)]));
        assert_eq!(iter(4, 5), (vec![], vec![id(4, 0)]));
        assert_eq!(iter(5, 5), (vec![], vec![]));

        assert_eq!(iter(0, 6), (vec![], vec![id(3, 2), id(5, 1)]));
        assert_eq!(iter(1, 6), (vec![id(1, 0), id(3, 1)], vec![id(5, 1)]));
        assert_eq!(iter(2, 6), (vec![id(3, 1)], vec![id(5, 1)]));
        assert_eq!(iter(3, 6), (vec![id(3, 0)], vec![id(5, 1)]));
        assert_eq!(iter(4, 6), (vec![], vec![id(5, 1)]));
        assert_eq!(iter(5, 6), (vec![id(5, 0)], vec![]));
        assert_eq!(iter(6, 6), (vec![], vec![]));

        assert_eq!(iter(0, 7), (vec![], vec![id(3, 2), id(5, 1), id(6, 0)]));
        assert_eq!(
            iter(1, 7),
            (vec![id(1, 0), id(3, 1)], vec![id(5, 1), id(6, 0)])
        );
        assert_eq!(iter(2, 7), (vec![id(3, 1)], vec![id(5, 1), id(6, 0)]));
        assert_eq!(iter(3, 7), (vec![id(3, 0)], vec![id(5, 1), id(6, 0)]));
        assert_eq!(iter(4, 7), (vec![], vec![id(5, 1), id(6, 0)]));
        assert_eq!(iter(5, 7), (vec![id(5, 0)], vec![id(6, 0)]));
        assert_eq!(iter(6, 7), (vec![], vec![id(6, 0)]));
        assert_eq!(iter(7, 7), (vec![], vec![]));

        assert_eq!(iter(0, 8), (vec![], vec![id(7, 3)]));
        assert_eq!(iter(1, 8), (vec![id(1, 0), id(3, 1), id(7, 2)], vec![]));
        assert_eq!(iter(2, 8), (vec![id(3, 1), id(7, 2)], vec![]));
        assert_eq!(iter(3, 8), (vec![id(3, 0), id(7, 2)], vec![]));
        assert_eq!(iter(4, 8), (vec![id(7, 2)], vec![]));
        assert_eq!(iter(5, 8), (vec![id(5, 0), id(7, 1)], vec![]));
        assert_eq!(iter(6, 8), (vec![id(7, 1)], vec![]));
        assert_eq!(iter(7, 8), (vec![id(7, 0)], vec![]));
        assert_eq!(iter(8, 8), (vec![], vec![]));
    }
}
