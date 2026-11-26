const PAGE: u64 = 1 << 16;
const VA_END: u64 = 1 << 48;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Transfer {
    pub(crate) source: u64,
    pub(crate) destination: u64,
    pub(crate) bytes: u64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Piece {
    pub(crate) source_page: u32,
    pub(crate) destination_page: u32,
    pub(crate) source_offset: u32,
    pub(crate) destination_offset: u32,
    pub(crate) bytes: u32,
    pub(crate) linear_offset: u64,
}

impl Transfer {
    pub(crate) fn new(source: u64, destination: u64, bytes: u64) -> Option<Self> {
        if source.checked_add(bytes)? > VA_END || destination.checked_add(bytes)? > VA_END {
            return None;
        }
        Some(Self {
            source,
            destination,
            bytes,
        })
    }

    pub(crate) fn pieces(self) -> impl Iterator<Item = Piece> {
        let mut off = 0;
        std::iter::from_fn(move || {
            if off == self.bytes {
                return None;
            }
            let src = self.source + off;
            let dst = self.destination + off;
            let a = src % PAGE;
            let b = dst % PAGE;
            let bytes = (PAGE - a.max(b)).min(self.bytes - off);
            let piece = Piece {
                source_page: (src / PAGE) as u32,
                destination_page: (dst / PAGE) as u32,
                source_offset: a as u32,
                destination_offset: b as u32,
                bytes: bytes as u32,
                linear_offset: off,
            };
            off += bytes;
            Some(piece)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn copy_crosses_two_different_boundaries_and_the_high_dword() {
        let t = Transfer::new((1 << 32) - 3, 2 * PAGE - 1, 7).unwrap();
        let parts = t.pieces().collect::<Vec<_>>();
        assert_eq!(parts.iter().map(|p| p.bytes).collect::<Vec<_>>(), [1, 2, 4]);
        assert_eq!(
            parts.iter().map(|p| p.source_page).collect::<Vec<_>>(),
            [65535, 65535, 65536]
        );
        assert_eq!(
            parts.iter().map(|p| p.destination_page).collect::<Vec<_>>(),
            [1, 2, 2]
        );
    }

    #[test]
    fn pieces_partition_exactly_the_requested_bytes() {
        for a in [0, 1, 2, PAGE - 1, PAGE, VA_END - 3 * PAGE] {
            for b in [0, 3, PAGE - 1, 2 * PAGE - 5, VA_END - 3 * PAGE] {
                for size in [0, 1, 2, 3, 4, PAGE - 1, PAGE, 2 * PAGE + 15] {
                    let mut next = 0;
                    for p in Transfer::new(a, b, size).unwrap().pieces() {
                        assert_eq!(next, p.linear_offset);
                        assert_eq!(
                            a + next,
                            u64::from(p.source_page) * PAGE + u64::from(p.source_offset)
                        );
                        assert_eq!(
                            b + next,
                            u64::from(p.destination_page) * PAGE + u64::from(p.destination_offset)
                        );
                        assert!(
                            p.bytes > 0
                                && u64::from(p.bytes + p.source_offset.max(p.destination_offset))
                                    <= PAGE
                        );
                        next += u64::from(p.bytes);
                    }
                    assert_eq!(next, size);
                }
            }
        }
        assert!(Transfer::new(u64::MAX, 0, 2).is_none());
        assert!(Transfer::new(0, VA_END - 1, 2).is_none());
    }
}
