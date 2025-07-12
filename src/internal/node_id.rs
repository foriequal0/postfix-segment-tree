use crate::internal::consts;

#[cfg_attr(test, derive(Eq, PartialEq, Debug))]
pub(crate) struct NodeId {
    index: usize,
    level: u32,
}

impl NodeId {
    pub(crate) fn new(index: usize, level: u32) -> Self {
        debug_assert!(index <= consts::MAX_LEN);
        debug_assert!(level <= get_max_level_from_index(index));

        NodeId { index, level }
    }

    pub(crate) fn index(&self) -> usize {
        self.index
    }

    pub(crate) fn level(&self) -> u32 {
        self.level
    }

    pub(crate) fn left_child(&self) -> NodeId {
        debug_assert!(self.level > 0);

        let child_level = self.level - 1;
        let width = 1 << child_level;
        NodeId {
            index: self.index - width,
            level: child_level,
        }
    }

    pub(crate) fn right_child(&self) -> NodeId {
        debug_assert!(self.level > 0);

        let child_level = self.level - 1;
        NodeId {
            index: self.index,
            level: child_level,
        }
    }

    pub(crate) fn node_index(&self) -> usize {
        let nodes_len = get_nodes_len_for(self.index);
        nodes_len + self.level as usize
    }
}

#[derive(Copy, Clone)]
#[cfg_attr(test, derive(Eq, PartialEq, Debug))]
pub(crate) struct LeafNodeId {
    index: usize,
}

impl LeafNodeId {
    pub(crate) fn new(index: usize) -> Self {
        debug_assert!(index <= consts::MAX_LEN);

        LeafNodeId { index }
    }

    pub(crate) fn index(&self) -> usize {
        self.index
    }

    pub(crate) fn node_index(&self) -> usize {
        get_nodes_len_for(self.index)
    }

    pub(crate) fn max_level(&self) -> u32 {
        get_max_level_from_index(self.index)
    }

    pub(crate) fn with_level(&self, level: u32) -> NodeId {
        debug_assert!(level <= self.max_level());

        NodeId::new(self.index, level)
    }
}

/// Gets the total number of nodes required to store elements of count `len`.
/// See also [`crate#encoding-layout`]
pub(crate) fn get_nodes_len_for(len: usize) -> usize {
    debug_assert!(len <= consts::MAX_LEN);

    // Let's shorten `get_nodes_len_for` as `f`
    //
    // We'll keep abuse the property that when there are `2^n` leaf nodes in a full binary tree,
    // there are `2 * 2^n - 1` nodes in the tree in total.
    // so `f(2^n) = 2 * 2^n - 1`
    //
    // We can define `f` as a recursive function
    // ```
    // f(0) = 0
    // f(n) = f(largest_power_of_two(n))       ; count the largest full binary tree portion
    //        + f(n - largest_power_of_two(n)) ; the rest
    // (largest_power_of_two(i) = 2^ilog2(i))
    // ```
    //
    // This is a *O*(log *n*) operation. And I initially tried to convert the recursive form
    // into an iterative form, but it became just bit operations.
    //
    // Let me summarize some operations in plain English.
    // * `ilog2(x)` => largest index of the bit set in `x`
    // * `largest_power_of_two(x)` = `2 ^ ilog(x)`
    //   => a value with only the largest bit set in `x`, the rest is cleared.
    // * `1 - largest_power_of_two(x)`
    //   => a value without the largest bit set in `x`.
    //
    // Then, we can see the pattern as we expand the recursive form.
    // ```
    // f(i) = f(largest_power_of_two(i)) + f(i - largest_power_of_two(i))
    //      = f(2 ^ ilog2(i))            + f(i - largest_power_of_two(i))
    //      = 2 * 2 ^ ilog(i) - 1        + f(i - largest_power_of_two(i))
    //      = 2 * (a value with only the largest bit set in `i`) - 1
    //        + f(a value without the largest bit set in `i`)
    //      = 2 * (a value with only the largest bit set in `i`) - 1
    //        + 2 * (a value with only the 2nd largest bit set in `i`) - 1
    //        + f(a value without the 2 largest bit set in `i`)
    //      = ...
    //      = 2 * i - (count of set bits)
    // ```
    len * 2 - len.count_ones() as usize
}

/// How many adjacent parent nodes are following after the leaf node for the `index`.
///
/// `get_max_level_from_index(2^n - 1) == n` will hold.
fn get_max_level_from_index(index: usize) -> u32 {
    // It would be enough to use `u8` for the return type for our use-case.
    // But `usize::ilog2()`, `usize::{trailing,leading}_{zeros,ones}()` returns `u32` for unknown reasons.
    // Let's just use `u32` to reduce conversions.

    // Let `g` = `get_max_level_from_index`
    // `f(i+1) = f(i) + g(i) + 1`
    // then,
    // `g(i) = f(i+1) - f(i) - 1
    //       = (2 * (i+1) - (i+1).count_ones()) - (2 * i - i.count_ones()) - 1
    //       = i.count_ones() - (i+1).count_ones() + 1
    // `i+1` can be seen as an operation that clears all trailing ones of `i`, then set the next bit, so:
    //       = (i.trailing_ones() - 1) + 1
    index.trailing_ones()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_childs() {
        let id = NodeId::new(1, 1);
        assert_eq!(id.left_child(), NodeId::new(0, 0));
        assert_eq!(id.right_child(), NodeId::new(1, 0));

        let id = NodeId::new(3, 1);
        assert_eq!(id.left_child(), NodeId::new(2, 0));
        assert_eq!(id.right_child(), NodeId::new(3, 0));

        let id = NodeId::new(3, 2);
        assert_eq!(id.left_child(), NodeId::new(1, 1));
        assert_eq!(id.right_child(), NodeId::new(3, 1));

        let id = NodeId::new(5, 1);
        assert_eq!(id.left_child(), NodeId::new(4, 0));
        assert_eq!(id.right_child(), NodeId::new(5, 0));

        let id = NodeId::new(7, 1);
        assert_eq!(id.left_child(), NodeId::new(6, 0));
        assert_eq!(id.right_child(), NodeId::new(7, 0));

        let id = NodeId::new(7, 2);
        assert_eq!(id.left_child(), NodeId::new(5, 1));
        assert_eq!(id.right_child(), NodeId::new(7, 1));

        let id = NodeId::new(7, 3);
        assert_eq!(id.left_child(), NodeId::new(3, 2));
        assert_eq!(id.right_child(), NodeId::new(7, 2));
    }

    #[test]
    fn test_node_index() {
        fn get(index: usize, level: u32) -> usize {
            NodeId::new(index, level).node_index()
        }

        assert_eq!(get(0, 0), 0);
        assert_eq!(get(1, 0), 1);
        assert_eq!(get(1, 1), 2);
        assert_eq!(get(2, 0), 3);
        assert_eq!(get(3, 0), 4);
        assert_eq!(get(3, 1), 5);
        assert_eq!(get(3, 2), 6);
        assert_eq!(get(4, 0), 7);
        assert_eq!(get(5, 0), 8);
        assert_eq!(get(5, 1), 9);
        assert_eq!(get(6, 0), 10);
        assert_eq!(get(7, 0), 11);
        assert_eq!(get(7, 1), 12);
        assert_eq!(get(7, 2), 13);
        assert_eq!(get(7, 3), 14);
    }

    #[test]
    fn test_leaf_node_index() {
        fn get(index: usize) -> usize {
            LeafNodeId::new(index).node_index()
        }

        assert_eq!(get(0), 0);
        assert_eq!(get(1), 1);
        assert_eq!(get(2), 3);
        assert_eq!(get(3), 4);
        assert_eq!(get(4), 7);
        assert_eq!(get(5), 8);
        assert_eq!(get(6), 10);
        assert_eq!(get(7), 11);
    }

    #[test]
    fn test_leaf_node_max_level() {
        fn get(index: usize) -> u32 {
            LeafNodeId::new(index).max_level()
        }

        assert_eq!(get(0), 0);
        assert_eq!(get(1), 1);
        assert_eq!(get(2), 0);
        assert_eq!(get(3), 2);
        assert_eq!(get(4), 0);
        assert_eq!(get(5), 1);
        assert_eq!(get(6), 0);
        assert_eq!(get(7), 3);
        assert_eq!(get(8), 0);
    }
}
