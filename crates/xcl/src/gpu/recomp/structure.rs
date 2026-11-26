//! Structured control flow from the program's control-flow graph, whatever the console's block
//! layout: every natural loop becomes one `for (;;)`, every merge point a label that the code
//! before it leaves by a flag (`xcl_skip_<pc>`), conditional branches `if` / `else` (the
//! dominator-tree translation of Ramsey, "Beyond Relooper", JFP 2022). Every branch of console
//! code is scalar, so all lanes take the same path and the structure is exact. A graph that is
//! not reducible (a cycle entered at two points), or that ends a block in a way the graph cannot
//! follow (`s_setpc`), is refused: the caller keeps the `pc` chain for it.
//!
//! The steps: [`Cfg::new`] (blocks, dominators, loops), `Cfg::tree` (the dominator-tree
//! statements), `Layout` (statements to the nodes the emitter writes: flags, guards, breaks),
//! and [`Cfg::loop_exits`] (where a loop's iteration guard leaves to, for `x64_masks`).

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use super::{Term, MAX_NEST};

/// The program's blocks and their graph. Block `n` (`n == blocks`) is the program's end.
pub(super) struct Cfg {
    pub start: Vec<u64>,
    pub end: Vec<u64>,
    /// successors; a conditional branch lists the taken target first
    pub succ: Vec<Vec<usize>>,
    pub n: usize,
    rpo: Vec<usize>,
    pub header: Vec<bool>,
    /// loops[h]: the blocks of the natural loop headed by h
    pub loops: BTreeMap<usize, BTreeSet<usize>>,
    /// latches[h]: blocks with a back edge to h
    pub latches: BTreeMap<usize, Vec<usize>>,
    label: Vec<bool>,
    /// labels placed at each block (ascending reverse postorder)
    placed: Vec<Vec<usize>>,
}

/// A statement of the dominator-tree translation.
#[derive(Debug, Clone)]
enum Stmt {
    Code(usize),
    /// block, taken, not taken
    Branch(usize, Vec<Stmt>, Vec<Stmt>),
    Loop(usize, Vec<Stmt>),
    /// the code before label y; y's own code follows the statement
    Labeled(usize, Vec<Stmt>),
    Exit(Exit),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Exit {
    Continue(usize),
    /// to a label (the end of the program: `Cfg::n`)
    Break(usize),
}

/// A flag the structured code leaves a label or continues an outer loop with.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum Flag {
    Skip(usize),
    Cont(usize),
}

/// What the emitter writes.
#[derive(Debug, Clone)]
pub(super) enum Node {
    /// a block whose end is a jump or falls through
    Code(usize),
    /// a block ending with a conditional branch: `if (cond != invert) { then } else { els }`
    If { block: usize, invert: bool, then: Vec<Node>, els: Vec<Node> },
    Loop { header: usize, body: Vec<Node> },
    Set(Flag),
    Clear(Flag),
    Break,
    Continue,
    /// `if (!(flags…)) { body }`
    Guard(Vec<Flag>, Vec<Node>),
    /// `if (flag) { body }`
    IfFlag(Flag, Vec<Node>),
}

impl Cfg {
    /// The graph of the blocks starting at `leaders` (`terms[i]`: how block i ends). Err: a
    /// block end the graph cannot follow, or an irreducible cycle.
    pub fn new(leaders: &[u64], terms: &[Term]) -> Result<Cfg, String> {
        let n = leaders.len();
        let start = leaders.to_vec();
        let end: Vec<u64> = (0..n).map(|i| leaders.get(i + 1).copied().unwrap_or(u64::MAX)).collect();
        let at = |t: u64| leaders.binary_search(&t).map_err(|_| format!("branch to {t:#x}, not a block"));
        let mut succ = Vec::with_capacity(n + 1);
        for i in 0..n {
            let next = if i + 1 < n { i + 1 } else { n };
            succ.push(match terms[i] {
                Term::Next => vec![next],
                Term::Cond(t) if at(t)? == next => vec![next],
                Term::Cond(t) => vec![at(t)?, next],
                Term::Jump(t) => vec![at(t)?],
                Term::End => vec![n],
                Term::Other => return Err(format!("block {:#x}: an end the graph cannot follow", start[i])),
            });
        }
        succ.push(Vec::new());
        let blocks = n + 1;
        // depth-first order from the entry (iterative: retail programs have thousands of blocks)
        let mut post_order = Vec::with_capacity(blocks);
        let mut reachable = vec![false; blocks];
        let mut stack: Vec<(usize, usize)> = vec![(0, 0)];
        reachable[0] = true;
        while let Some(top) = stack.last_mut() {
            let (b, k) = *top;
            if let Some(&s) = succ[b].get(k) {
                top.1 += 1;
                if !reachable[s] {
                    reachable[s] = true;
                    stack.push((s, 0));
                }
            } else {
                post_order.push(b);
                stack.pop();
            }
        }
        let order: Vec<usize> = post_order.iter().rev().copied().collect();
        let mut rpo = vec![usize::MAX; blocks];
        for (k, &b) in order.iter().enumerate() {
            rpo[b] = k;
        }
        let mut preds: Vec<Vec<usize>> = vec![Vec::new(); blocks];
        for &b in &order {
            for &s in &succ[b] {
                preds[s].push(b);
            }
        }
        // dominators (Cooper, Harvey, Kennedy)
        let mut idom = vec![usize::MAX; blocks];
        idom[0] = 0;
        let intersect = |idom: &[usize], mut a: usize, mut b: usize| {
            while a != b {
                while rpo[a] > rpo[b] {
                    a = idom[a];
                }
                while rpo[b] > rpo[a] {
                    b = idom[b];
                }
            }
            a
        };
        loop {
            let mut changed = false;
            for &b in order.iter().skip(1) {
                let mut new = usize::MAX;
                for &p in &preds[b] {
                    if idom[p] != usize::MAX {
                        new = if new == usize::MAX { p } else { intersect(&idom, p, new) };
                    }
                }
                if idom[b] != new {
                    idom[b] = new;
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
        let mut children: Vec<Vec<usize>> = vec![Vec::new(); blocks];
        for &b in order.iter().skip(1) {
            children[idom[b]].push(b);
        }
        let (mut pre, mut post) = (vec![0; blocks], vec![0; blocks]);
        let mut clock = 0;
        let mut stack: Vec<(usize, usize)> = vec![(0, 0)];
        while let Some(top) = stack.last_mut() {
            let (b, k) = *top;
            if let Some(&c) = children[b].get(k) {
                top.1 += 1;
                clock += 1;
                pre[c] = clock;
                stack.push((c, 0));
            } else {
                clock += 1;
                post[b] = clock;
                stack.pop();
            }
        }
        let dominates = |a: usize, b: usize| pre[a] <= pre[b] && post[b] <= post[a];
        let mut header = vec![false; blocks];
        let mut latches: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        let mut fwd = vec![0usize; blocks];
        for &b in &order {
            for &s in &succ[b] {
                if rpo[s] <= rpo[b] {
                    if !dominates(s, b) {
                        return Err(format!("irreducible: {:#x} enters a cycle that {:#x} does not head", start.get(b).copied().unwrap_or(u64::MAX), start.get(s).copied().unwrap_or(u64::MAX)));
                    }
                    header[s] = true;
                    latches.entry(s).or_default().push(b);
                } else {
                    fwd[s] += 1;
                }
            }
        }
        let mut loops: BTreeMap<usize, BTreeSet<usize>> = BTreeMap::new();
        for (&h, ls) in &latches {
            let mut body = BTreeSet::from([h]);
            let mut work: Vec<usize> = ls.clone();
            while let Some(b) = work.pop() {
                if body.insert(b) {
                    work.extend(preds[b].iter().copied());
                }
            }
            loops.insert(h, body);
        }
        let mut nest: Vec<Vec<usize>> = vec![Vec::new(); blocks];
        for (&h, body) in &loops {
            for &b in body {
                nest[b].push(h);
            }
        }
        // labels: merge points, and blocks a loop exits to (placed outside the loop)
        let mut label = vec![false; blocks];
        for &b in order.iter().skip(1) {
            if b == n {
                continue;
            }
            let exit_target = preds[b].iter().any(|&p| nest[p].iter().any(|h| !nest[b].contains(h)));
            label[b] = fwd[b] >= 2 || exit_target;
        }
        // a label goes to the nearest dominator of it that is in no loop the label is not in
        let mut placed: Vec<Vec<usize>> = vec![Vec::new(); blocks];
        for &y in &order {
            if !label[y] {
                continue;
            }
            let mut a = idom[y];
            while a != 0 && nest[a].iter().any(|h| !nest[y].contains(h)) {
                a = idom[a];
            }
            placed[a].push(y);
        }
        for p in &mut placed {
            p.sort_by_key(|&y| rpo[y]);
        }
        Ok(Cfg { start, end, succ, n, rpo, header, loops, latches, label, placed })
    }

    /// The end of a loop's last latch (the range `loop_profile` names it by).
    pub fn loop_end(&self, h: usize) -> u64 {
        self.latches.get(&h).into_iter().flatten().map(|&l| self.end[l]).max().unwrap_or(self.end[h])
    }

    fn tree(&self, x: usize) -> Vec<Stmt> {
        let body = self.within(x, &self.placed[x]);
        if self.header[x] {
            vec![Stmt::Loop(x, body)]
        } else {
            body
        }
    }

    fn within(&self, x: usize, labels: &[usize]) -> Vec<Stmt> {
        match labels.split_last() {
            None => self.node(x),
            Some((&y, rest)) => {
                let mut v = vec![Stmt::Labeled(y, self.within(x, rest))];
                v.extend(self.tree(y));
                v
            }
        }
    }

    fn node(&self, x: usize) -> Vec<Stmt> {
        match self.succ[x].as_slice() {
            [t, f] => vec![Stmt::Branch(x, self.branch(x, *t), self.branch(x, *f))],
            [t] => {
                let mut v = vec![Stmt::Code(x)];
                v.extend(self.branch(x, *t));
                v
            }
            _ => vec![Stmt::Code(x)],
        }
    }

    fn branch(&self, src: usize, tgt: usize) -> Vec<Stmt> {
        if tgt == self.n {
            vec![Stmt::Exit(Exit::Break(self.n))]
        } else if self.rpo[tgt] <= self.rpo[src] {
            vec![Stmt::Exit(Exit::Continue(tgt))]
        } else if self.label[tgt] {
            vec![Stmt::Exit(Exit::Break(tgt))]
        } else {
            self.tree(tgt)
        }
    }

    /// The nodes of the whole program.
    pub fn layout(&self) -> Vec<Node> {
        let mut lay = Layout { frames: vec![Frame::Label(self.n)], depth: 1 };
        let (mut nodes, _) = lay.seq(self.tree(0), Some(Exit::Break(self.n)));
        let mut tested = BTreeSet::new();
        tested_flags(&nodes, &mut tested);
        drop_untested(&mut nodes, &tested);
        nodes
    }

    /// For each loop header: the block a lane continues with when the loop's iteration guard
    /// leaves it (None: the end of the program), as `nodes` are laid out.
    pub fn loop_exits(&self, nodes: &[Node]) -> BTreeMap<usize, Option<usize>> {
        let mut out = BTreeMap::new();
        follows(nodes, None, &mut out);
        out
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Frame {
    Loop(usize),
    Label(usize),
}

struct Layout {
    frames: Vec<Frame>,
    depth: usize,
}

type Open = Option<(BTreeSet<Flag>, Vec<Node>)>;

fn close(out: &mut Vec<Node>, guard: &mut Open) {
    if let Some((flags, body)) = guard.take() {
        out.push(Node::Guard(flags.into_iter().collect(), body));
    }
}

/// `nodes` after the code so far, under a guard while flags are pending.
fn place(out: &mut Vec<Node>, guard: &mut Open, pending: &BTreeSet<Flag>, nodes: Vec<Node>) {
    if nodes.is_empty() {
        return;
    }
    if guard.as_ref().map(|g| &g.0) != Some(pending) {
        close(out, guard);
        if !pending.is_empty() {
            *guard = Some((pending.clone(), Vec::new()));
        }
    }
    match guard {
        Some((_, body)) => body.extend(nodes),
        None => out.extend(nodes),
    }
}

impl Layout {
    fn innermost_loop(&self) -> Option<usize> {
        self.frames.iter().rev().find_map(|f| match f {
            Frame::Loop(h) => Some(*h),
            _ => None,
        })
    }

    /// Loops between here and the label `y`.
    fn loops_inside(&self, y: usize) -> usize {
        self.frames.iter().rev().take_while(|f| **f != Frame::Label(y)).filter(|f| matches!(f, Frame::Loop(_))).count()
    }

    /// A statement list; `follow`: the exit that falling off its end amounts to (None: the
    /// code after it in the output is not that of any exit). The flags still set at its end.
    fn seq(&mut self, stmts: Vec<Stmt>, follow: Option<Exit>) -> (Vec<Node>, BTreeSet<Flag>) {
        let mut work: VecDeque<Stmt> = stmts.into();
        let mut out = Vec::new();
        let mut pending: BTreeSet<Flag> = BTreeSet::new();
        let mut guard: Open = None;
        while let Some(st) = work.pop_front() {
            let follow_here = if work.is_empty() { follow } else { None };
            let guarded = !pending.is_empty() as usize;
            self.depth += guarded;
            let (nodes, raised, resolved) = match st {
                Stmt::Code(b) => (vec![Node::Code(b)], BTreeSet::new(), None),
                Stmt::Exit(e) if Some(e) == follow_here => (Vec::new(), BTreeSet::new(), None),
                Stmt::Exit(e) => {
                    let (n, r) = self.exit(e);
                    (n, r, None)
                }
                Stmt::Branch(b, taken, not_taken) if self.depth >= MAX_NEST => {
                    // too deep to nest further: the smaller branch under an `if` with its exits
                    // explicit, the other one after it
                    let small_taken = count(&taken) <= count(&not_taken);
                    let (nested, rest) = if small_taken { (taken, not_taken) } else { (not_taken, taken) };
                    self.depth += 1;
                    let (n, r) = self.seq(nested, None);
                    self.depth -= 1;
                    for s in rest.into_iter().rev() {
                        work.push_front(s);
                    }
                    (vec![Node::If { block: b, invert: !small_taken, then: n, els: Vec::new() }], r, None)
                }
                Stmt::Branch(b, taken, not_taken) => {
                    self.depth += 1;
                    let (t, tr) = self.seq(taken, follow_here);
                    let (f, fr) = self.seq(not_taken, follow_here);
                    self.depth -= 1;
                    let raised: BTreeSet<Flag> = tr.union(&fr).copied().collect();
                    let node = if t.is_empty() { Node::If { block: b, invert: true, then: f, els: Vec::new() } } else { Node::If { block: b, invert: false, then: t, els: f } };
                    (vec![node], raised, None)
                }
                Stmt::Loop(h, body) => {
                    self.frames.push(Frame::Loop(h));
                    self.depth += 1;
                    let (b, r) = self.seq(body, Some(Exit::Continue(h)));
                    self.depth -= 1;
                    self.frames.pop();
                    let mut nodes = vec![Node::Loop { header: h, body: b }];
                    let mut raised = BTreeSet::new();
                    for f in r {
                        match f {
                            Flag::Cont(c) if self.innermost_loop() == Some(c) => nodes.push(Node::IfFlag(f, vec![Node::Clear(f), Node::Continue])),
                            Flag::Cont(_) => {
                                nodes.push(Node::IfFlag(f, vec![Node::Break]));
                                raised.insert(f);
                            }
                            Flag::Skip(y) => {
                                if self.loops_inside(y) > 0 {
                                    nodes.push(Node::IfFlag(f, vec![Node::Break]));
                                }
                                raised.insert(f);
                            }
                        }
                    }
                    (nodes, raised, None)
                }
                Stmt::Labeled(y, body) => {
                    self.frames.push(Frame::Label(y));
                    let (b, r) = self.seq(body, Some(Exit::Break(y)));
                    self.frames.pop();
                    (b, r, Some(Flag::Skip(y)))
                }
            };
            self.depth -= guarded;
            place(&mut out, &mut guard, &pending, nodes);
            pending.extend(raised.iter().copied());
            if let Some(f) = resolved {
                if pending.remove(&f) {
                    place(&mut out, &mut guard, &pending, vec![Node::Clear(f)]);
                }
            }
        }
        close(&mut out, &mut guard);
        (out, pending)
    }

    fn exit(&self, e: Exit) -> (Vec<Node>, BTreeSet<Flag>) {
        match e {
            Exit::Continue(h) if self.innermost_loop() == Some(h) => (vec![Node::Continue], BTreeSet::new()),
            Exit::Continue(h) => (vec![Node::Set(Flag::Cont(h)), Node::Break], BTreeSet::from([Flag::Cont(h)])),
            Exit::Break(y) => {
                let f = Flag::Skip(y);
                let nodes = if self.loops_inside(y) == 0 { vec![Node::Set(f)] } else { vec![Node::Set(f), Node::Break] };
                (nodes, BTreeSet::from([f]))
            }
        }
    }
}

fn count(stmts: &[Stmt]) -> usize {
    stmts
        .iter()
        .map(|s| match s {
            Stmt::Branch(_, a, b) => 1 + count(a) + count(b),
            Stmt::Loop(_, a) | Stmt::Labeled(_, a) => 1 + count(a),
            _ => 1,
        })
        .sum()
}

fn tested_flags(nodes: &[Node], out: &mut BTreeSet<Flag>) {
    for n in nodes {
        match n {
            Node::Guard(flags, body) => {
                out.extend(flags.iter().copied());
                tested_flags(body, out);
            }
            Node::IfFlag(f, body) => {
                out.insert(*f);
                tested_flags(body, out);
            }
            Node::If { then, els, .. } => {
                tested_flags(then, out);
                tested_flags(els, out);
            }
            Node::Loop { body, .. } => tested_flags(body, out),
            _ => {}
        }
    }
}

/// Setting or clearing a flag nothing tests is dead code.
fn drop_untested(nodes: &mut Vec<Node>, tested: &BTreeSet<Flag>) {
    nodes.retain(|n| !matches!(n, Node::Set(f) | Node::Clear(f) if !tested.contains(f)));
    for n in nodes {
        match n {
            Node::Guard(_, body) | Node::IfFlag(_, body) | Node::Loop { body, .. } => drop_untested(body, tested),
            Node::If { then, els, .. } => {
                drop_untested(then, tested);
                drop_untested(els, tested);
            }
            _ => {}
        }
    }
}

/// The first block `nodes` run with every flag clear, else `after`.
fn first_block(nodes: &[Node], after: Option<usize>) -> Option<usize> {
    for n in nodes {
        match n {
            Node::Code(b) | Node::If { block: b, .. } => return Some(*b),
            Node::Loop { header, .. } => return Some(*header),
            Node::Guard(_, body) => {
                if let Some(b) = first_block(body, None) {
                    return Some(b);
                }
            }
            _ => {}
        }
    }
    after
}

/// Where each loop's guard leaves to: the first block after the loop with every flag clear
/// (the guard leaves with none set); `after`: what follows `nodes`.
fn follows(nodes: &[Node], after: Option<usize>, out: &mut BTreeMap<usize, Option<usize>>) {
    for (k, n) in nodes.iter().enumerate() {
        let next = || first_block(&nodes[k + 1..], after);
        match n {
            Node::Loop { header, body } => {
                out.insert(*header, next());
                follows(body, Some(*header), out);
            }
            Node::If { then, els, .. } => {
                let a = next();
                follows(then, a, out);
                follows(els, a, out);
            }
            Node::Guard(_, body) | Node::IfFlag(_, body) => follows(body, next(), out),
            _ => {}
        }
    }
}
