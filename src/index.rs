use std::ops::Index;

use crate::PostfixSegmentTree;
use crate::internal::node_id::LeafNodeId;

impl<T> PostfixSegmentTree<T> {
    /// Returns an element at `index`.
    ///
    /// # Examples
    ///
    /// ```
    /// use postfix_segment_tree::PostfixSegmentTree;
    ///
    /// let mut tree = PostfixSegmentTree::from_iter([1, 2, 3]);
    /// assert_eq!(tree.get(1), Some(&2));
    /// ```
    ///
    /// # Time Complexity
    ///
    /// *O*(1)
    pub fn get(&self, index: usize) -> Option<&T> {
        if index >= self.len() {
            return None;
        }

        let id = LeafNodeId::new(index);
        Some(self.get_leaf_node(id))
    }
}

impl<T> Index<usize> for PostfixSegmentTree<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        let id = LeafNodeId::new(index);
        self.get_leaf_node(id)
    }
}
