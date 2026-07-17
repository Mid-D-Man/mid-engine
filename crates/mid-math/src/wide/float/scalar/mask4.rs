// crates/mid-math/src/wide/float/scalar/mask4.rs
//! Scalar fallback float comparison mask (4-lane).

use core::fmt;
use core::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Not};

/// 4-lane float comparison mask — scalar fallback.
/// Lane value: `u32::MAX` = true, `0` = false.
#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(C, align(16))]
pub struct Mask4(pub(crate) [u32; 4]);

impl Mask4 {
    pub const FALSE: Self = Mask4([0; 4]);
    pub const TRUE:  Self = Mask4([u32::MAX; 4]);

    #[inline] pub fn any(self)  -> bool { self.0.iter().any(|&x| x != 0) }
    #[inline] pub fn all(self)  -> bool { self.0.iter().all(|&x| x != 0) }
    #[inline] pub fn none(self) -> bool { self.0.iter().all(|&x| x == 0) }

    #[inline]
    pub fn bitmask(self) -> u32 {
        self.0.iter().enumerate().fold(0u32, |acc, (i, &x)| {
            acc | (if x != 0 { 1u32 } else { 0 }) << i
        })
    }

    /// Index (0-3) of the lowest-numbered true lane, or `None` if none are true.
    #[inline]
    pub fn first_set_lane(self) -> Option<u32> {
        let b = self.bitmask();
        if b == 0 { None } else { Some(b.trailing_zeros()) }
    }

    /// Number of true lanes (0-4).
    #[inline]
    pub fn count_set(self) -> u32 { self.bitmask().count_ones() }

    /// Iterate the indices (0-3) of true lanes, lowest to highest.
    #[inline]
    pub fn iter_set_lanes(self) -> Mask4LaneIter { Mask4LaneIter { bits: self.bitmask() } }

    #[allow(dead_code)]
    #[inline(always)]
    pub(crate) fn from_bools(a: bool, b: bool, c: bool, d: bool) -> Self {
        let lane = |v: bool| if v { u32::MAX } else { 0 };
        Mask4([lane(a), lane(b), lane(c), lane(d)])
    }
}

/// Iterator over the true-lane indices of a [`Mask4`]. See [`Mask4::iter_set_lanes`].
pub struct Mask4LaneIter { bits: u32 }

impl Iterator for Mask4LaneIter {
    type Item = u32;
    #[inline]
    fn next(&mut self) -> Option<u32> {
        if self.bits == 0 { return None; }
        let idx = self.bits.trailing_zeros();
        self.bits &= self.bits - 1; // clear lowest set bit
        Some(idx)
    }
    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.bits.count_ones() as usize;
        (n, Some(n))
    }
}
impl ExactSizeIterator for Mask4LaneIter {}

impl BitAnd for Mask4 { type Output=Self; fn bitand(self, r: Self) -> Self { Mask4([self.0[0]&r.0[0], self.0[1]&r.0[1], self.0[2]&r.0[2], self.0[3]&r.0[3]]) } }
impl BitAndAssign for Mask4 { fn bitand_assign(&mut self, r: Self) { *self = *self & r; } }
impl BitOr for Mask4 { type Output=Self; fn bitor(self, r: Self) -> Self { Mask4([self.0[0]|r.0[0], self.0[1]|r.0[1], self.0[2]|r.0[2], self.0[3]|r.0[3]]) } }
impl BitOrAssign for Mask4 { fn bitor_assign(&mut self, r: Self) { *self = *self | r; } }
impl BitXor for Mask4 { type Output=Self; fn bitxor(self, r: Self) -> Self { Mask4([self.0[0]^r.0[0], self.0[1]^r.0[1], self.0[2]^r.0[2], self.0[3]^r.0[3]]) } }
impl BitXorAssign for Mask4 { fn bitxor_assign(&mut self, r: Self) { *self = *self ^ r; } }
impl Not for Mask4 { type Output=Self; fn not(self) -> Self { Mask4(self.0.map(|x| !x)) } }

impl fmt::Debug for Mask4 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let b = self.bitmask();
        write!(f, "Mask4({},{},{},{})", b&1!=0, b>>1&1!=0, b>>2&1!=0, b>>3&1!=0)
    }
}
