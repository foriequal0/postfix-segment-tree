use std::ops::AddAssign;

use crate::PostfixSegmentTree;
use crate::internal::node_id::{LeafNodeId, NodeId, get_nodes_len_for};

// Internal node access operations.
//
// Internal nodes become dirty when you modify the value. ("DIRTY:" tag)
// They need to be cleaned by recalculation or truncate. ("CLEAN:" tag)
impl<T> PostfixSegmentTree<T> {
    pub(crate) fn get_node(&self, id: NodeId) -> &T {
        let node_index = id.node_index();
        &self.nodes[node_index]
    }

    /// DIRTY: parents of `id`, when you arbitrarily modify the value of the returned reference
    pub(crate) fn get_node_mut(&mut self, id: NodeId) -> &mut T {
        let node_index = id.node_index();
        &mut self.nodes[node_index]
    }

    pub(crate) fn get_leaf_node(&self, id: LeafNodeId) -> &T {
        let node_index = id.node_index();
        &self.nodes[node_index]
    }

    /// DIRTY: parents of `id`, when you arbitrarily modify the value of the returned reference
    pub(crate) fn get_leaf_node_mut(&mut self, id: LeafNodeId) -> &mut T {
        let node_index = id.node_index();
        &mut self.nodes[node_index]
    }

    /// DIRTY: parents of `left` and `right`, when `left` != `right`
    pub(crate) fn swap_leaf_nodes(&mut self, left: LeafNodeId, right: LeafNodeId) {
        let left_node_index = left.node_index();
        let right_node_index = right.node_index();
        self.nodes.swap(left_node_index, right_node_index)
    }
}

// internal operations.
impl<T> PostfixSegmentTree<T> {
    /// Shifts all elements from `index` to the right by 1 to insert a new element.
    ///
    /// `elements[index]` at the end of this operation will be `elements[len() - 1]` before this operation as a result.
    ///
    /// # Time complexity
    ///
    /// *O*([`len`])
    ///
    /// [`len`]: PostfixSegmentTree::len
    ///
    /// DIRTY: all parents of `node_id.index() >= index`
    pub(crate) fn shift_nodes_right_by_one(&mut self, index: usize) {
        let len = self.len();
        if len == 0 {
            return;
        }

        debug_assert!(index < len);

        let shift_len = self.len() - index - 1;
        for i in (index..(index + shift_len)).rev() {
            let left = LeafNodeId::new(i);
            let right = LeafNodeId::new(i + 1);
            self.swap_leaf_nodes(left, right); // use swap to not require Copy/Clone
        }
    }

    /// Shift all elements after `index` to the left by 1 to remove an element.
    ///
    /// `nodes` are need to be recalculated appropriately after removing an element.
    ///
    /// `elements[len() - 1]` at the end of this operation will be `elements[index]` before this operation as a result.
    ///
    /// # Time complexity
    ///
    /// *O*([`len`])
    ///
    /// [`len`]: PostfixSegmentTree::len
    ///
    /// DIRTY: all parents of `node_id.index() >= index`
    pub(crate) fn shift_nodes_left_by_one(&mut self, index: usize) {
        let len = self.len();
        if len == 0 {
            return;
        }

        debug_assert!(index < len);

        let shift_len = len - 1 - index;
        for i in index..(index + shift_len) {
            let left = LeafNodeId::new(i);
            let right = LeafNodeId::new(i + 1);
            self.swap_leaf_nodes(left, right); // use swap to not require Copy/Clone
        }
    }

    /// Push empty nodes for a new element to be inserted.
    ///
    /// `nodes` are need to be recalculated appropriately after inserting a new element.
    ///
    /// # Time complexity
    ///
    /// *O*([`len`])
    ///
    /// [`len`]: PostfixSegmentTree::len
    ///
    /// DIRTY: parents of `len() -1`
    pub(crate) fn resize_by_one(&mut self)
    where
        T: Default,
    {
        let len = self.len();
        let nodes_len = get_nodes_len_for(len + 1);
        debug_assert!(nodes_len > self.nodes_len());

        self.nodes.resize_with(nodes_len, T::default);
        self.len += 1;
    }

    /// Truncate trailing nodes to remove an element.
    ///
    /// # Time complexity
    ///
    /// *O*(1)
    ///
    /// CLEAN: parents of `len() - 1`
    pub(crate) fn truncate_by_one(&mut self) {
        let len = self.len();
        debug_assert!(len > 0);

        self.truncate(len - 1);
    }

    /// Recalculate internal nodes after updating an element at `index`
    ///
    /// # Time complexity
    ///
    /// *O*(log [`nodes_len`])
    ///
    /// [`nodes_len`]: PostfixSegmentTree::nodes_len
    ///
    /// CLEAN: parents of `id`
    pub(crate) fn recalculate_nodes_after_update(&mut self, id: LeafNodeId)
    where
        for<'a> T: AddAssign<&'a T> + Default,
    {
        let mut current_index = id.index();
        let mut current_level = 1;
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

    /// Recalculate internal nodes after updating `elements[index..]`
    ///
    /// It updates all nodes after `NodeId::new(index, 0)`
    ///
    /// # Time complexity
    ///
    /// *O*([`nodes_len`])
    ///
    /// [`nodes_len`]: PostfixSegmentTree::nodes_len
    ///
    /// CLEAN: all parents of `node_id.index() >= id.index()`
    pub(crate) fn recalculate_nodes_after_bulk_update(&mut self, id: LeafNodeId)
    where
        for<'a> T: AddAssign<&'a T> + Default,
    {
        let len = self.len();
        for i in id.index()..len {
            let leaf_node_id = LeafNodeId::new(i);
            let max_level = leaf_node_id.max_level();
            for level in 1..=max_level {
                let node_id = leaf_node_id.with_level(level);
                self.recalculate_node(node_id);
            }
        }
    }

    /// Recalculate a node at `NodeId::new(index, level)` using their children.
    ///
    /// CLEAN: `id`
    fn recalculate_node(&mut self, id: NodeId)
    where
        for<'a> T: AddAssign<&'a T> + Default,
    {
        debug_assert!(id.index() < self.len());
        debug_assert!(id.level() >= 1);

        let mut sum = T::default();

        sum += self.get_node(id.left_child());
        sum += self.get_node(id.right_child());

        *self.get_node_mut(id) = sum;
    }
}
