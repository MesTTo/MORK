/// Dense id type accepted by [`UnionFind`].
pub(crate) trait UnionFindId: Copy + Eq {
    fn from_index(index: u32) -> Self;
    fn index(self) -> usize;
}

/// Union-find over dense ids.
#[derive(Clone, Debug)]
pub(crate) struct UnionFind<I> {
    parents: Vec<I>,
    ranks: Vec<u8>,
}

impl<I> Default for UnionFind<I> {
    fn default() -> Self {
        Self {
            parents: Vec::new(),
            ranks: Vec::new(),
        }
    }
}

impl<I: UnionFindId> UnionFind<I> {
    pub(crate) fn new() -> Self {
        Self {
            parents: Vec::new(),
            ranks: Vec::new(),
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.parents.len()
    }

    pub(crate) fn contains(&self, id: I) -> bool {
        id.index() < self.parents.len()
    }

    pub(crate) fn ids(&self) -> impl Iterator<Item = I> + '_ {
        (0..self.parents.len()).map(|index| I::from_index(index as u32))
    }

    pub(crate) fn make_set(&mut self) -> I {
        let id = I::from_index(self.parents.len() as u32);
        self.parents.push(id);
        self.ranks.push(0);
        id
    }

    /// Class root, read-only.
    pub(crate) fn find(&self, mut id: I) -> I {
        while self.parents[id.index()] != id {
            id = self.parents[id.index()];
        }
        id
    }

    /// Class root with path halving along the walk.
    pub(crate) fn find_mut(&mut self, mut id: I) -> I {
        while self.parents[id.index()] != id {
            let parent = self.parents[id.index()];
            let grandparent = self.parents[parent.index()];
            self.parents[id.index()] = grandparent;
            id = parent;
        }
        id
    }

    /// Merges two classes by rank.
    pub(crate) fn union(&mut self, left: I, right: I) -> (I, bool) {
        let mut left = self.find_mut(left);
        let mut right = self.find_mut(right);
        if left == right {
            return (left, false);
        }
        if self.ranks[left.index()] < self.ranks[right.index()] {
            std::mem::swap(&mut left, &mut right);
        }
        self.parents[right.index()] = left;
        if self.ranks[left.index()] == self.ranks[right.index()] {
            self.ranks[left.index()] = self.ranks[left.index()].saturating_add(1);
        }
        (left, true)
    }

    pub(crate) fn validate(&self) -> Result<(), &'static str> {
        if self.parents.len() != self.ranks.len() {
            return Err("union-find arrays differ in length");
        }
        for (index, parent) in self.parents.iter().copied().enumerate() {
            if parent.index() >= self.parents.len() {
                return Err("union-find parent is out of bounds");
            }
            let mut current = I::from_index(index as u32);
            for _ in 0..=self.parents.len() {
                let parent = self.parents[current.index()];
                if parent == current {
                    break;
                }
                current = parent;
            }
            if self.parents[current.index()] != current {
                return Err("union-find parent cycle");
            }
        }
        Ok(())
    }
}
