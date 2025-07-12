//! # Internal operations
//!
//! Internal nodes become dirty when you modify the value. ("DIRTY:" tag)
//! They need to be cleaned by recalculation or truncate. ("CLEAN:" tag)

use crate::PostfixSegmentTree;
use crate::internal::consts;
use crate::internal::node_id::{LeafNodeId, NodeId, get_nodes_len_for};
use std::ops::AddAssign;

// internal operations: node access
impl<T> PostfixSegmentTree<T> {
    pub(crate) fn get_node(&self, id: NodeId) -> &T {
        let node_index = id.node_index();
        &self.nodes[node_index]
    }

    /// DIRTY: parents of `id`, when you arbitrarily modify the value of the returned reference
    pub(crate) fn get_node_mut(&mut self, id: NodeId) -> &mut T {
        debug_assert!(id.index() < self.len());

        let node_index = id.node_index();
        &mut self.nodes[node_index]
    }

    pub(crate) fn get_leaf_node(&self, id: LeafNodeId) -> &T {
        debug_assert!(id.index() < self.len());

        let node_index = id.node_index();
        &self.nodes[node_index]
    }

    /// DIRTY: parents of `id`, when you arbitrarily modify the value of the returned reference
    pub(crate) fn get_leaf_node_mut(&mut self, id: LeafNodeId) -> &mut T {
        debug_assert!(id.index() < self.len());

        let node_index = id.node_index();
        &mut self.nodes[node_index]
    }

    /// DIRTY: parents of `left` and `right`, when `get_leaf_node(left) != get_leaf_node(right)`
    pub(crate) fn swap_leaf_nodes(&mut self, left: LeafNodeId, right: LeafNodeId) {
        debug_assert!(left.index() < self.len());
        debug_assert!(right.index() < self.len());

        let left_node_index = left.node_index();
        let right_node_index = right.node_index();
        self.nodes.swap(left_node_index, right_node_index)
    }
}

// internal operations: push and pop
impl<T> PostfixSegmentTree<T>
where
    T: Default,
{
    /// Push a new default element and its empty parent nodes.
    ///
    /// `nodes` are need to be recalculated appropriately after inserting a new element.
    ///
    /// # Time complexity
    ///
    /// *O*([`len`])
    ///
    /// [`len`]: PostfixSegmentTree::len
    ///
    /// DIRTY: parents of `len() - 1`
    pub(crate) fn push_default_dirty(&mut self) -> LeafNodeId {
        debug_assert!(self.len() <= consts::MAX_LEN - 1);

        let len = self.len();
        let nodes_len = get_nodes_len_for(len + 1);
        debug_assert!(nodes_len > self.nodes_len());

        self.nodes.resize_with(nodes_len, T::default);
        self.len += 1;

        LeafNodeId::new(self.len - 1)
    }

    /// Pop the last leaf node, and truncate nodes
    ///
    /// # Time complexity
    ///
    /// *O*(1)
    pub(crate) fn pop(&mut self) -> T {
        debug_assert!(self.len() > 0);

        let len = self.len();

        let mut popped = T::default();
        let last_node = self.get_leaf_node_mut(LeafNodeId::new(len - 1));
        std::mem::swap(&mut popped, last_node);

        self.truncate(len - 1);

        popped
    }
}

// internal operations: rotate
impl<T> PostfixSegmentTree<T> {
    /// Rotates all elements from `id` to the right by 1 to insert a new element.
    ///
    /// `elements[id]` at the end of this operation will be the last element before this operation as a result.
    ///
    /// # Time complexity
    ///
    /// *O*([`len`])
    ///
    /// [`len`]: PostfixSegmentTree::len
    ///
    /// DIRTY: all parents of `node_id.index() >= id`
    pub(crate) fn rotate_leaf_nodes_right_by_one_dirty(&mut self, id: LeafNodeId) {
        debug_assert!(id.index() < self.len());

        // right rotation. [1, 2, 3, 4] will be [4, 1, 2, 3].
        // notice the wraparound.
        let index = id.index();
        let shift_len = self.len() - index - 1;
        for i in (index..(index + shift_len)).rev() {
            let left = LeafNodeId::new(i);
            let right = LeafNodeId::new(i + 1);
            self.swap_leaf_nodes(left, right); // use swap to not require Copy/Clone
        }
    }

    /// Rotates all elements after `id` to the left by 1 to remove an element.
    ///
    /// The last element at the end of this operation will be `elements[id]` before this operation as a result.
    ///
    /// # Time complexity
    ///
    /// *O*([`len`])
    ///
    /// [`len`]: PostfixSegmentTree::len
    ///
    /// DIRTY: all parents of `node_id.index() >= id`
    pub(crate) fn rotate_leaf_nodes_left_by_one_dirty(&mut self, id: LeafNodeId) {
        debug_assert!(id.index() < self.len());

        // left rotation. [1, 2, 3, 4] will be [2, 3, 4, 1].
        // notice the wraparound.
        let index = id.index();
        let shift_len = self.len() - 1 - index;
        for i in index..(index + shift_len) {
            let left = LeafNodeId::new(i);
            let right = LeafNodeId::new(i + 1);
            self.swap_leaf_nodes(left, right); // use swap to not require Copy/Clone
        }
    }
}

// internal operations: recalculate
impl<T> PostfixSegmentTree<T>
where
    for<'a> T: AddAssign<&'a T> + Default,
{
    /// Recalculate internal nodes after updating an element at `id.index()`
    ///
    /// # Time complexity
    ///
    /// *O*(log [`nodes_len`])
    ///
    /// [`nodes_len`]: PostfixSegmentTree::nodes_len
    ///
    /// CLEAN: parents of `id`
    pub(crate) fn recalculate_nodes_after_update(&mut self, id: LeafNodeId) {
        debug_assert!(id.index() < self.len());

        let mut current_index = id.index();
        let mut current_level = 1; // starts from 1 since leaf nodes are always CLEAN
        let len = self.len();
        while current_index < len {
            let leaf_node_id = LeafNodeId::new(current_index);
            let max_level = leaf_node_id.max_level();
            while current_level <= max_level {
                let node_id = leaf_node_id.with_level(current_level);
                self.recalculate_node(node_id);

                current_level += 1;
            }

            current_index += 1 << (current_level - 1);
        }
    }

    /// Recalculate internal nodes after updating `elements[id.index()..]`
    ///
    /// It updates all nodes after `id`
    ///
    /// # Time complexity
    ///
    /// *O*([`nodes_len`])
    ///
    /// [`nodes_len`]: PostfixSegmentTree::nodes_len
    ///
    /// CLEAN: all parents of `node_id.index() >= id.index()`
    pub(crate) fn recalculate_nodes_after_bulk_update(&mut self, id: LeafNodeId) {
        debug_assert!(id.index() < self.len());

        let len = self.len();
        for i in id.index()..len {
            let leaf_node_id = LeafNodeId::new(i);
            let max_level = leaf_node_id.max_level();

            // starts from 1 since leaf nodes are always CLEAN
            for level in 1..=max_level {
                let node_id = leaf_node_id.with_level(level);
                self.recalculate_node(node_id);
            }
        }
    }

    /// Recalculate a node at `id` using their children.
    ///
    /// CLEAN: `id`
    fn recalculate_node(&mut self, id: NodeId) {
        debug_assert!(id.index() < self.len());
        debug_assert!(id.level() >= 1);

        let mut sum = T::default();

        // child.index() <= id.index()
        // child.level() == id.level() - 1
        sum += self.get_node(id.left_child());
        sum += self.get_node(id.right_child());

        *self.get_node_mut(id) = sum;
    }
}
