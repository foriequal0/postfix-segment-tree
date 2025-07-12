//! [`PostfixSegmentTree`] is a variant of Segment Tree that can calculate `push` in amortized *O*(1) time.
//!
//! # Overview and Comparision
//!
//! It is similar to Segment Tree and Fenwick Tree:
//! * *O*(log *n*) to query [`prefix_sum`].
//! * *O*(log *n*) to [`update`] an element in the tree.
//! * Also, elements are stored in a [`Vec`], forms an implicit tree for compact size.
//!
//! It encodes nodes in postfix order like Fenwick Tree,
//! while typical Segment Tree encodes them in prefix order.
//! The postfix order gives you amortized *O*(1) for [`push`] thanks to the index stability.
//! With the prefix order, you have to move all nodes to the new index when the new root node is created.
//! (O(*n*) moves for every *n = 2^k* insertion, so amortized *O*(log *n*))
//!
//! Also, it is not succinct like Segment Tree.
//! In other words, it has information redundancies.
//! It requires up to *2 \* n - 1* nodes for *n* elements like Segment Tree,
//! while the Fenwick Tree requires exact *n* nodes for *n* elements.
//! However, it allows you *O*(1) for [`get`], rather than Fenwick Tree's *O*(log *n*).
//!
//! # Time complexities
//!
//! It's a prefix sum tree, so it can:
//! * [`prefix_sum`], [`sum`]: *O*(log *index*) *O*(log *index*)
//! * [`update`]: *O*(log [`len`])
//!
//! Also, it has some nice properties due to its encoding layout:
//! * `push`: amortized *O*(1) (unlike amortized O(log *n*) Segment Tree)
//! * `index`: *O*(1) (unlike O(log *n*) Fenwick Tree)
//!
//! `push` is amortized *O*(1), so it is naturally *O*(n) even if you construct it naively.
//! The typical Segment Tree that naive construction gives you *O*(n log *n*),
//! while *O*(n) is possible with specialized bulk implementation.
//!
//! And since it's based on a `Vec`:
//! * `insert`: *O*(log [`len`])
//! * `remove`: *O*(log [`len`])
//!
//! But unlike Segment Tree and Fenwick Tree, the implementation is relatively straightforward,
//! since access is *O*(1) and doesn't need scary tree operations.
//!
//! [`prefix_sum`]: PostfixSegmentTree::prefix_sum
//! [`sum`]: PostfixSegmentTree::sum
//! [`update`]: PostfixSegmentTree::update
//! [`push`]: PostfixSegmentTree::push
//! [`get`]: PostfixSegmentTree::get
//! [`len`]: PostfixSegmentTree::len
//!
//! # Encoding Layout
//!
//! Nodes of the tree in the Postfix Segment Tree are encoded in postfix order,
//! rather than conventional Segment Tree's prefix order.
//!
//! Assume we have an array of `elements`. We construct `nodes` as the following pseudocode:
//!
//! ```pseudocode
//! let node_id = NodeId::new(index, level);
//! let width = 1 << level;
//! nodes[node_id] = elements[index - width + 1..=index].iter().sum()
//! ```
//!
//! Then we can visualize `nodes` as following.
//!
//! ```text
//! level: 3 [                             14]
//!        2 [            6] [             13]
//!        1 [    2] [    5] [    9] [     12] [     17]
//!  leaf: 0 [0] [1] [3] [4] [7] [8] [10] [11] [15] [16] [18] ...
//! index:    0   1   2   3   4   5    6    7    8    9   10  ...
//! ```
//!
//! A number in `[]` is the `node_id`, and it is also the index of node value in `nodes`
//! The width of `[  node_id  ]` indicates how `nodes[node_id]` covers the sum of `element[index]`.
//!
//! Leaf nodes (`node_id = NodeId::new(index, 0)`) will have the same value with `elements[index]`
//!
//! This postfix order encoding layout has a nice little property:
//! the index and the range of nodes are stable over the push operation.
//! The new leaf node for the new element is pushed at the end when the element is pushed at the end,
//! and new parent nodes that cover the range of previous nodes follow after.
//!
//! So, the existing nodes aren't moved or updated by the push operation.
//! In general, when an element is inserted in the middle of existing elements,
//! nodes that correspond to the elements before the inserted index keep their index not changed.
//!
//! As a result, the index of any element is independent of the total number of elements.
//! It makes insertion and remove operations in the middle a little bit much easier.
//!
//! # Trivia
//!
//! It actually forms a minimal set of full binary trees,
//! but it's a hybrid of Segment Tree and Fenwick Tree, so let's call it a tree.
mod index;
mod internal;
mod iterator;

pub use crate::iterator::ElementIterator;

use crate::internal::consts;
use crate::internal::node_id::{LeafNodeId, get_nodes_len_for};
use crate::internal::skipping_iterator::{IncreasingSkippingIterator, SkippingIterator};
use std::ops::AddAssign;

/// A variant of Segment Tree that can calculate `push` in amortized *O*(1) time.
pub struct PostfixSegmentTree<T> {
    pub(crate) nodes: Vec<T>,
    pub(crate) len: usize,
}

// memory managements operations
impl<T> PostfixSegmentTree<T> {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            len: 0,
        }
    }

    /// Returns the total number of nodes
    ///
    /// `nodes_len` == [`crate::internal::node_id::get_nodes_len_for`]\(`len`) == `len` \* 2 - `len.count_ones()` will hold
    ///
    /// # Examples
    ///
    /// ```
    /// use postfix_segment_tree::PostfixSegmentTree;
    ///
    /// let mut tree = PostfixSegmentTree::new();
    /// tree.push(1); // create 1 node
    /// tree.push(2); // create 2 nodes
    /// tree.push(3); // create 1 node
    /// tree.push(4); // create 3 nodes
    ///
    /// assert_eq!(tree.nodes_len(), 7);
    /// ```
    ///
    /// [`len`]: PostfixSegmentTree::len
    pub fn nodes_len(&self) -> usize {
        self.nodes.len()
    }

    /// Returns the total number of elements, which is the total number of leaf nodes.
    ///
    /// # Examples
    ///
    /// ```
    /// use postfix_segment_tree::PostfixSegmentTree;
    ///
    /// let mut tree = PostfixSegmentTree::from_iter([1, 2, 3]);
    /// assert_eq!(tree.len(), 3);
    /// ```
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn nodes_capacity(&self) -> usize {
        self.nodes.capacity()
    }

    /// Reserves capacity for at least `additional` more nodes to be inserted.
    pub fn reserve_nodes(&mut self, additional: usize) {
        self.nodes.reserve(additional);
    }

    /// Reserves capacity for at least `additional` more elements to be inserted.
    pub fn reserve(&mut self, additional: usize) {
        let new_capacity = self.len() + additional;
        assert!(new_capacity <= consts::MAX_LEN);

        let new_nodes_capacity = get_nodes_len_for(new_capacity);
        let nodes_len = self.nodes_len();
        if new_nodes_capacity > nodes_len {
            let additional_nodes = new_nodes_capacity - nodes_len;
            self.reserve_nodes(additional_nodes)
        }
    }

    pub fn reserve_nodes_exact(&mut self, additional: usize) {
        self.nodes.reserve_exact(additional);
    }

    pub fn reserve_exact(&mut self, additional: usize) {
        let new_capacity = self.len() + additional;
        assert!(new_capacity <= consts::MAX_LEN);

        let new_nodes_capacity = get_nodes_len_for(new_capacity);
        let nodes_len = self.nodes_len();
        if new_nodes_capacity > nodes_len {
            let additional_nodes = new_nodes_capacity - nodes_len;
            self.reserve_nodes_exact(additional_nodes)
        }
    }

    pub fn shrink_to_fit(&mut self) {
        self.nodes.shrink_to_fit()
    }

    pub fn shrink_nodes_to(&mut self, min_nodes_capacity: usize) {
        self.nodes.shrink_to(min_nodes_capacity)
    }

    pub fn shrink_to(&mut self, min_capacity: usize) {
        assert!(min_capacity <= consts::MAX_LEN);

        let min_nodes_capacity = get_nodes_len_for(min_capacity);
        self.shrink_nodes_to(min_nodes_capacity)
    }

    /// Resizes the tree to hold `len` elements. It does nothing if `len` >= [`len()`].
    ///
    /// # Examples
    ///
    /// ```
    /// use postfix_segment_tree::PostfixSegmentTree;
    ///
    /// let mut tree = PostfixSegmentTree::from_iter([1, 2, 3]);
    /// tree.truncate(2);
    /// assert_eq!(tree.len(), 2);
    /// ```
    ///
    /// [`len()`]: PostfixSegmentTree::len
    pub fn truncate(&mut self, len: usize) {
        if len >= self.len() {
            return;
        }

        assert!(len <= consts::MAX_LEN);
        let nodes_len = get_nodes_len_for(len);
        self.nodes.truncate(nodes_len);
        self.len = len;
    }
}

impl<T> FromIterator<T> for PostfixSegmentTree<T>
where
    for<'a> T: AddAssign<&'a T> + Default,
{
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let mut tree = Self::new();
        for element in iter {
            tree.push(element);
        }

        tree
    }
}

// sum query
impl<T> PostfixSegmentTree<T>
where
    for<'a> T: AddAssign<&'a T> + Default,
{
    /// Returns the equivalent of `self.iter().take(index).sum()`
    ///
    /// # Examples
    ///
    /// ```
    /// use postfix_segment_tree::PostfixSegmentTree;
    ///
    /// let mut tree = PostfixSegmentTree::from_iter([1, 2, 3]);
    /// assert_eq!(tree.prefix_sum(0), 0);
    /// assert_eq!(tree.prefix_sum(1), 1);
    /// assert_eq!(tree.prefix_sum(2), 3);
    /// assert_eq!(tree.prefix_sum(3), 6);
    /// ```
    ///
    /// # Time complexity
    ///
    /// *O*(log `index`)
    ///
    /// [`len`]: PostfixSegmentTree::len
    pub fn prefix_sum(&self, index: usize) -> T {
        assert!(index <= self.len());

        let mut sum = T::default();
        for id in SkippingIterator::new(index) {
            sum += self.get_node(id);
        }

        sum
    }

    /// Returns the equivalent of `self.iter().skip(index).sum()`
    ///
    /// # Examples
    ///
    /// ```
    /// use postfix_segment_tree::PostfixSegmentTree;
    ///
    /// let mut tree = PostfixSegmentTree::from_iter([1, 2, 3]);
    /// assert_eq!(tree.postfix_sum(0), 6);
    /// assert_eq!(tree.postfix_sum(1), 5);
    /// assert_eq!(tree.postfix_sum(2), 3);
    /// assert_eq!(tree.postfix_sum(3), 0);
    /// ```
    ///
    /// # Time complexity
    ///
    /// *O*(log `index`)
    ///
    /// [`len`]: PostfixSegmentTree::len
    pub fn postfix_sum(&self, index: usize) -> T {
        assert!(index <= self.len());

        self.sum(index, self.len() - index)
    }

    /// Returns the equivalent of `self.iter().skip(index).take(len).sum()`
    ///
    /// # Examples
    ///
    /// ```
    /// use postfix_segment_tree::PostfixSegmentTree;
    ///
    /// let mut tree = PostfixSegmentTree::from_iter([1, 2, 3, 4]);
    /// assert_eq!(tree.sum(0, 0), 0);
    /// assert_eq!(tree.sum(0, 3), 6);
    /// assert_eq!(tree.sum(1, 2), 5);
    /// assert_eq!(tree.sum(2, 2), 7);
    /// ```
    ///
    /// # Time complexity
    ///
    /// *O*(log `index`)
    ///
    /// [`len`]: PostfixSegmentTree::len
    pub fn sum(&self, index: usize, len: usize) -> T {
        assert!(index <= self.len());
        assert!(len <= self.len() - index);

        let mut sum = T::default();
        let mut iter = SkippingIterator::new(index + len);
        let pivot = iter.skip_to_pivot(index);

        // sum index..pivot
        for id in IncreasingSkippingIterator::new(index, pivot) {
            sum += self.get_node(id);
        }

        // sum pivot..index+count
        for id in iter {
            sum += self.get_node(id);
        }

        sum
    }
}

// update operations
impl<T> PostfixSegmentTree<T>
where
    for<'a> T: AddAssign<&'a T> + Default,
{
    /// Analogous to `elements[index] = element`
    ///
    /// ```
    /// use postfix_segment_tree::PostfixSegmentTree;
    ///
    /// let mut tree = PostfixSegmentTree::from_iter([1, 2, 3]);
    /// tree.update(1, 4);
    ///
    /// // elements are updated
    /// assert_eq!(tree[0], 1);
    /// assert_eq!(tree[1], 4);
    /// assert_eq!(tree[2], 3);
    ///
    /// // partial sums are also updated
    /// assert_eq!(tree.prefix_sum(1), 1);
    /// assert_eq!(tree.prefix_sum(2), 5);
    /// assert_eq!(tree.prefix_sum(3), 8);
    /// ```
    ///
    /// # time complexity
    ///
    /// *O*(log [`len`])
    ///
    /// [`len`]: PostfixSegmentTree::len
    pub fn update(&mut self, index: usize, element: T) {
        assert!(index < self.len());

        let id = LeafNodeId::new(index);
        *self.get_leaf_node_mut(id) = element; // DIRTY: parents of `id`

        self.recalculate_nodes_after_update(id); // CLEAN: parents of `id`
    }

    /// Appends an element to the back of the collection.
    ///
    /// # time complexity
    ///
    /// Amortized *O*(1).
    ///
    /// ## Simple proof
    ///
    /// When you push *n* elements, there are *O*(*n*) nodes pushed in total.
    /// (see [`crate::internal::node_id::get_nodes_len_for`])
    /// Also, nodes are updated exactly once since the node's index and range are stable.
    /// So *O*(*n*) pushes/updates for *n* pushed elements => amortized *O*(1) push per push.
    ///
    /// *O* (log *n*) in the worst case when capacity is fixed, but the frequency decreases exponentially.
    /// *O* ([`nodes_capacity`]) in the worst case when the underlying [`nodes_capacity`] is increased
    /// due to underlying usage of [`Vec`].
    ///
    /// [`nodes_capacity`]: PostfixSegmentTree::nodes_capacity
    pub fn push(&mut self, element: T) {
        assert!(self.len() <= consts::MAX_LEN - 1);

        let new_leaf = self.push_default_dirty(); // DIRTY: parents of `self.len() - 1` after the operation, which is `inserted_at`
        *self.get_leaf_node_mut(new_leaf) = element; // DIRTY: parents of `inserted_at`

        self.recalculate_nodes_after_update(new_leaf); // CLEAN: parents of `inserted_at
    }

    /// Shifts all elements from `index` to the right, then inserts an `element` at `index`.
    ///
    /// # time complexity
    ///
    /// *O*([`len`])
    ///
    /// [`len`]: PostfixSegmentTree::len
    pub fn insert(&mut self, index: usize, element: T) {
        assert!(self.len() <= consts::MAX_LEN - 1);
        assert!(index <= self.len());

        let new_leaf = self.push_default_dirty(); // DIRTY: parents of `self.len() - 1` after the operation, which is `inserted_at`
        *self.get_leaf_node_mut(new_leaf) = element; // DIRTY: parents of `inserted_at`

        let id = LeafNodeId::new(index);
        self.rotate_leaf_nodes_right_by_one_dirty(id); // DIRTY: all parents of `>= id`, which includes `new_leaf`

        self.recalculate_nodes_after_bulk_update(id); // CLEAN: all parents of `>= id`
    }

    /// Remove an element at the `index` of this tree and shift all elements after `index` to the left.
    ///
    /// # Time complexity
    ///
    /// *O*([`len`])
    ///
    /// [`len`]: PostfixSegmentTree::len
    pub fn remove(&mut self, index: usize) -> T {
        assert!(index < self.len());

        let id = LeafNodeId::new(index);

        self.rotate_leaf_nodes_left_by_one_dirty(id); // DIRTY: all parents of `>= id`
        let popped = self.pop();

        self.recalculate_nodes_after_bulk_update(id); // CLEAN: all parents of `>= id`
        popped
    }
}
