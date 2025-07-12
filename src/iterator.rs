use std::iter::FusedIterator;

use crate::PostfixSegmentTree;

impl<T> PostfixSegmentTree<T> {
    /// Returns an [`ElementIterator`], which is an iterator for elements on this tree.
    pub fn iter(&self) -> ElementIterator<'_, T> {
        ElementIterator::new(self, 0, self.len())
    }
}

/// Iterator for elements on [`PostfixSegmentTree`].
pub struct ElementIterator<'a, T> {
    tree: &'a PostfixSegmentTree<T>,
    index: usize,
    end: usize,
}

impl<'a, T> ElementIterator<'a, T> {
    pub(crate) fn new(tree: &'a PostfixSegmentTree<T>, index: usize, end: usize) -> Self {
        ElementIterator { tree, index, end }
    }
}

impl<'a, T> Iterator for ElementIterator<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index + 1 >= self.end {
            return None;
        }

        let value = self.tree.get(self.index);
        if self.index < self.end {
            self.index += 1;
        }

        value
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.end - self.index;
        (len, Some(len))
    }

    fn last(self) -> Option<Self::Item>
    where
        Self: Sized,
    {
        let len = self.end - self.index;
        if len == 0 {
            return None;
        }

        let index = self.end - 1;
        self.tree.get(index)
    }

    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        if self.index + n + 1 >= self.end {
            return None;
        }

        let value = self.tree.get(self.index);
        if self.index < self.end {
            self.index += 1;
        }

        value
    }
}

impl<'a, T> FusedIterator for ElementIterator<'a, T> {}

impl<'a, T> ExactSizeIterator for ElementIterator<'a, T> {}

impl<'a, T> DoubleEndedIterator for ElementIterator<'a, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.index >= self.end {
            return None;
        }

        let value = self.tree.get(self.end - 1);
        if self.end >= self.index + 1 {
            self.end -= 1;
        }

        value
    }

    fn nth_back(&mut self, n: usize) -> Option<Self::Item> {
        if self.index + n >= self.end {
            return None;
        }

        let value = self.tree.get(self.end - n - 1);
        if self.end >= self.index + n + 1 {
            self.end -= n + 1;
        }

        value
    }
}
