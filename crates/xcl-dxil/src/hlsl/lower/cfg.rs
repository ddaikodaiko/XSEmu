//! Clean-up of the control-flow graph once the entry function is lowered.
//!
//! The structured lowering gives every branch body, every `break` and every arm of a
//! short-circuit operator a block of its own, also when nothing ends up in it. DXC's
//! SimplifyCFG leaves no such blocks, and the driver's compiler pays for every block it is
//! handed, so they are removed here: one pass over the blocks, no instruction moves.

use crate::ir::{flag, Block, Body, Const, Inst, Module, Op, Ty, Value, ValueKind};

/// Apply the forwarding the lowering recorded ([`Body::replace`]: the phis that turned out
/// trivial) to every operand, and leave a body without any. The passes that follow (flags,
/// liveness, the writer) resolve every operand they read, which is a hash lookup each as long
/// as one forwarding entry exists; here it is one pass over a dense table.
pub(super) fn apply_forwarding(body: &mut Body) {
    if !body.has_replacements() {
        return;
    }
    let mut to = vec![Value::NONE; body.insts.len()];
    for (i, v) in body.replacements() {
        to[i as usize] = v;
    }
    // every chain down to its end
    for i in 0..to.len() {
        if to[i].is_none() {
            continue;
        }
        let mut v = to[i];
        let mut steps = 0;
        while v.is_inst() && !to[v.0 as usize].is_none() && steps <= to.len() {
            v = to[v.0 as usize];
            steps += 1;
        }
        to[i] = v;
    }
    let fwd = |v: &mut Value| {
        if v.is_inst() {
            if let Some(&n) = to.get(v.0 as usize) {
                if !n.is_none() {
                    *v = n;
                }
            }
        }
    };
    for x in body.insts.iter_mut() {
        match x.op {
            Op::Bin(_) | Op::ICmp(_) | Op::FCmp(_) | Op::Store | Op::AtomicRmw(_) => {
                fwd(&mut x.a);
                fwd(&mut x.b);
            }
            Op::Cast(_) | Op::Load | Op::ExtractValue | Op::CondBr | Op::Switch | Op::Call | Op::Gep => fwd(&mut x.a),
            Op::Select | Op::CmpXchg => {
                fwd(&mut x.a);
                fwd(&mut x.b);
                fwd(&mut x.c);
            }
            Op::Phi | Op::Alloca | Op::Br | Op::Ret | Op::Unreachable => {}
        }
    }
    body.operands.iter_mut().for_each(fwd);
    for list in body.phis.iter_mut() {
        for inc in list.iter_mut() {
            fwd(&mut inc.0);
        }
    }
    // the forwarding table is private to the body: move the contents to one that has none
    let mut clean = Body::default();
    clean.insts = std::mem::take(&mut body.insts);
    clean.blocks = std::mem::take(&mut body.blocks);
    clean.operands = std::mem::take(&mut body.operands);
    clean.phis = std::mem::take(&mut body.phis);
    clean.switches = std::mem::take(&mut body.switches);
    clean.aligns = std::mem::take(&mut body.aligns);
    *body = clean;
}

/// Zero in place of every undefined value that an instruction other than a phi or a call
/// would read: the validator rejects those ("Instructions should not read uninitialized
/// value"), and DXC rejects the program where its own folds leave one. What reaches here is
/// what [`super::super::emit::Emit`] did not fold away: a stored value, an index, a branch
/// condition. A phi may have an undefined incoming value, and the arguments of a dx.op are
/// undefined by design where the operation does not use them.
pub(super) fn define_reads(body: &mut Body, m: &mut Module) {
    // one undefined constant per type: a handful
    let undefs: Vec<(Value, Ty)> = m.consts.iter().enumerate().filter(|(_, c)| matches!(c.c, Const::Undef)).map(|(i, c)| (Value::constant(i as u32), c.ty)).collect();
    if undefs.is_empty() {
        return;
    }
    let mut zero = |v: &mut Value| {
        if v.is_const() {
            if let Some(&(_, ty)) = undefs.iter().find(|u| u.0 == *v) {
                *v = m.zero(ty);
            }
        }
    };
    let Body { insts, operands, .. } = body;
    for x in insts.iter_mut() {
        match x.op {
            Op::Bin(_) | Op::ICmp(_) | Op::FCmp(_) | Op::Store | Op::AtomicRmw(_) => {
                zero(&mut x.a);
                zero(&mut x.b);
            }
            Op::Cast(_) | Op::Load | Op::CondBr | Op::Switch => zero(&mut x.a),
            Op::Select | Op::CmpXchg => {
                zero(&mut x.a);
                zero(&mut x.b);
                zero(&mut x.c);
            }
            Op::Gep => {
                zero(&mut x.a);
                operands[x.b.0 as usize..(x.b.0 + x.c.0) as usize].iter_mut().for_each(&mut zero);
            }
            Op::Phi | Op::Call | Op::ExtractValue | Op::Alloca | Op::Br | Op::Ret | Op::Unreachable => {}
        }
    }
}

/// Remove the blocks that hold nothing but an unconditional branch and are entered from one
/// conditional branch: that branch goes to the target itself, and the target's phis take
/// their value from the branching block. Not done when the branching block reaches the
/// target on its other edge too (a phi could not tell the two edges apart), nor for the
/// blocks in `keep` (the preheader, the latch and the exit block of a loop, which keep the
/// loop in the shape it was built in also when they are empty).
///
/// Then [`fold_phi_blocks`], for which a loop's exit block (`exits`) is not kept.
///
/// Returns whether a phi or select of type `handle` is live: DXIL does not allow a resource
/// handle to be chosen that way, and the liveness is at hand here.
pub(super) fn thread_empty_blocks(body: &mut Body, keep: &[u32], exits: &[u32], handle: Ty) -> bool {
    let n = body.blocks.len();
    // what the writer will drop is not there: a block whose instructions are all dead is empty
    let crate::bcwriter::Live { blocks: reachable, insts: live } = crate::bcwriter::liveness(body);
    let merged_handle = body.insts.iter().zip(&live).any(|(x, &l)| l && x.ty == handle && matches!(x.op, Op::Phi | Op::Select));
    let is_live = |i: &&u32| live.get(**i as usize).copied().unwrap_or(true);
    // (stops at the second live instruction: most are live)
    let one_live = |list: &[u32]| {
        let mut it = list.iter().filter(is_live);
        it.next().is_some() && it.next().is_none()
    };
    let mut npred = vec![0u32; n];
    let mut pred = vec![u32::MAX; n];
    for (b, data) in body.blocks.iter().enumerate() {
        let Some(&t) = data.insts.last() else { continue };
        let inst = body.insts[t as usize];
        let mut add = |s: u32| {
            if let Some(c) = npred.get_mut(s as usize) {
                *c += 1;
                pred[s as usize] = b as u32;
            }
        };
        match inst.op {
            Op::Br => add(inst.a.0),
            Op::CondBr => {
                add(inst.b.0);
                add(inst.c.0);
            }
            Op::Switch => {
                add(inst.b.0);
                for &(_, s) in &body.switches[inst.c.0 as usize] {
                    add(s.0);
                }
            }
            _ => {}
        }
    }
    let mut kept = vec![false; n];
    for &k in keep {
        kept[k as usize] = true;
    }
    // a chain of such blocks needs a round per link when the links come in the wrong order
    for _ in 0..4 {
        let mut changed = false;
        for b in 1..n {
            let data = &body.blocks[b];
            if kept[b] || npred[b] != 1 || data.insts.is_empty() || data.phis.iter().any(|i| is_live(&i)) || !one_live(&data.insts) {
                continue;
            }
            let br = body.insts[data.insts[data.insts.len() - 1] as usize];
            let (p, t) = (pred[b], br.a.0);
            if br.op != Op::Br || t as usize == b || p as usize == b || t as usize >= n {
                continue;
            }
            let Some(&pt) = body.blocks[p as usize].insts.last() else { continue };
            let target_phis = body.blocks[t as usize].phis.iter().filter(is_live).count();
            let term = &mut body.insts[pt as usize];
            if term.op != Op::CondBr {
                continue;
            }
            let on_true = term.b.0 == b as u32;
            let other = if on_true { term.c.0 } else { term.b.0 };
            if other == b as u32 {
                continue;
            }
            if other == t {
                // both edges end in the target: the condition decides nothing, unless a phi
                // there tells the edges apart
                if target_phis == 0 {
                    *term = crate::ir::Inst { op: Op::Br, a: Value(t), b: Value::NONE, c: Value::NONE, ..*term };
                    npred[t as usize] -= 1;
                    pred[t as usize] = p;
                    npred[b] = 0;
                    changed = true;
                }
                continue;
            }
            if on_true {
                term.b = Value(t);
            } else {
                term.c = Value(t);
            }
            for &phi in &body.blocks[t as usize].phis {
                let k = body.insts[phi as usize].a.0 as usize;
                for inc in body.phis[k].iter_mut() {
                    if inc.1 == Block(b as u32) {
                        inc.1 = Block(p);
                    }
                }
            }
            if pred[t as usize] == b as u32 {
                pred[t as usize] = p;
            }
            npred[b] = 0;
            changed = true;
        }
        if !changed {
            break;
        }
    }
    for &x in exits {
        kept[x as usize] = false;
    }
    fold_phi_blocks(body, &kept, &live, &reachable);
    merged_handle
}

/// Fold every block that holds only phis and an unconditional branch into the branch's
/// target, when its phis are read by the target's phis alone: the predecessors branch to
/// the target, and each phi there takes, in place of the value that came from the block,
/// what the block's phi took (or the same value once per predecessor). That is what
/// SimplifyCFG does in DXC, which leaves no such block: the merge block of an inner `if`
/// that ends an outer one, an else-if chain, the exit block of a loop with `break`s that
/// is followed by nothing. Not done when a predecessor of the block also enters the target
/// (the target's phis could not tell the two edges apart), nor for the blocks of `keep`
/// (the preheader and the latch of a loop).
///
/// `live` and `reachable` are the writer's liveness of the body before the threading
/// above, which changes neither for anything that is read here.
fn fold_phi_blocks(body: &mut Body, keep: &[bool], live: &[bool], reachable: &[bool]) {
    let n = body.blocks.len();
    let is_live = |i: u32| live.get(i as usize).copied().unwrap_or(true);
    // the target of a block that holds a live phi and, apart from phis, only a branch
    let target = |body: &Body, b: usize| -> Option<usize> {
        let data = &body.blocks[b];
        let mut others = data.insts.iter().filter(|&&i| is_live(i));
        if keep[b] || !reachable[b] || !data.phis.iter().any(|&p| is_live(p)) || others.next().is_none() || others.next().is_some() {
            return None;
        }
        let br = body.insts[*data.insts.last()? as usize];
        let s = br.a.0 as usize;
        (br.op == Op::Br && s != b && s < n && reachable[s]).then_some(s)
    };
    if !(1..n).any(|b| target(body, b).is_some()) {
        return;
    }
    // how many live instructions read each instruction
    let mut uses = vec![0u32; body.insts.len()];
    for (b, data) in body.blocks.iter().enumerate() {
        if !reachable[b] {
            continue;
        }
        for &i in data.phis.iter().chain(&data.insts) {
            if !is_live(i) {
                continue;
            }
            let x = &body.insts[i as usize];
            let mut f = |v: Value| {
                if v.is_inst() {
                    if let Some(u) = uses.get_mut(v.0 as usize) {
                        *u += 1;
                    }
                }
            };
            match x.op {
                Op::Bin(_) | Op::ICmp(_) | Op::FCmp(_) | Op::Store | Op::AtomicRmw(_) => {
                    f(x.a);
                    f(x.b);
                }
                Op::Cast(_) | Op::Load | Op::ExtractValue | Op::CondBr | Op::Switch => f(x.a),
                Op::Select | Op::CmpXchg => {
                    f(x.a);
                    f(x.b);
                    f(x.c);
                }
                Op::Call | Op::Gep => {
                    f(x.a);
                    body.operands_of(x).iter().for_each(|&v| f(v));
                }
                Op::Phi => body.phis[x.a.0 as usize].iter().filter(|inc| reachable[inc.1 .0 as usize]).for_each(|inc| f(inc.0)),
                Op::Alloca | Op::Br | Op::Ret | Op::Unreachable => {}
            }
        }
    }
    // by instruction: 1 + the reads counted in the target's phis, for the phis of the block looked at
    let mut tally = vec![0u32; body.insts.len()];
    // by block: the block it was last seen as a predecessor of
    let mut stamp = vec![u32::MAX; n];
    let mut gone = vec![false; n];
    let mut preds: Vec<u32> = Vec::new();
    // a chain of such blocks needs a round per link when the links come in the wrong order
    for _ in 0..4 {
        let mut changed = false;
        for b in 1..n {
            if gone[b] {
                continue;
            }
            let Some(s) = target(body, b) else { continue };
            // the predecessors are where a live phi takes its values from
            let phi_blocks = |body: &Body, of: usize, out: &mut Vec<u32>| {
                out.clear();
                if let Some(&p) = body.blocks[of].phis.iter().find(|&&p| is_live(p)) {
                    out.extend(body.phis[body.insts[p as usize].a.0 as usize].iter().map(|inc| inc.1 .0).filter(|&from| reachable[from as usize]));
                }
            };
            phi_blocks(body, s, &mut preds);
            let mut ok = !preds.is_empty();
            for &p in &preds {
                stamp[p as usize] = b as u32;
            }
            phi_blocks(body, b, &mut preds);
            ok &= !preds.is_empty() && preds.iter().all(|&p| stamp[p as usize] != b as u32);
            if !ok {
                continue;
            }
            let bphis = std::mem::take(&mut body.blocks[b].phis);
            let sphis = std::mem::take(&mut body.blocks[s].phis);
            for &p in &bphis {
                tally[p as usize] = 1;
            }
            for &q in sphis.iter().filter(|&&q| is_live(q)) {
                for inc in &body.phis[body.insts[q as usize].a.0 as usize] {
                    if inc.1 .0 as usize == b && inc.0.is_inst() && tally[inc.0 .0 as usize] != 0 {
                        tally[inc.0 .0 as usize] += 1;
                    }
                }
            }
            ok = bphis.iter().all(|&p| !is_live(p) || tally[p as usize] - 1 == uses[p as usize]);
            if ok {
                for &q in &sphis {
                    let kq = body.insts[q as usize].a.0 as usize;
                    let Some(j) = body.phis[kq].iter().position(|inc| inc.1 .0 as usize == b) else { continue };
                    let (v, _) = body.phis[kq].swap_remove(j);
                    if v.is_inst() && tally[v.0 as usize] != 0 {
                        let from = std::mem::take(&mut body.phis[body.insts[v.0 as usize].a.0 as usize]);
                        if is_live(q) {
                            for inc in from.iter().filter(|inc| inc.0.is_inst() && reachable[inc.1 .0 as usize]) {
                                uses[inc.0 .0 as usize] += 1;
                            }
                        }
                        body.phis[kq].extend_from_slice(&from);
                        body.phis[body.insts[v.0 as usize].a.0 as usize] = from;
                    } else {
                        body.phis[kq].extend(preds.iter().map(|&p| (v, Block(p))));
                        if is_live(q) && v.is_inst() {
                            uses[v.0 as usize] += preds.len() as u32 - 1;
                        }
                    }
                }
                for &p in bphis.iter().filter(|&&p| is_live(p)) {
                    for inc in body.phis[body.insts[p as usize].a.0 as usize].iter().filter(|inc| inc.0.is_inst() && reachable[inc.1 .0 as usize]) {
                        uses[inc.0 .0 as usize] -= 1;
                    }
                }
                for &p in &preds {
                    let Some(&t) = body.blocks[p as usize].insts.last() else { continue };
                    let term = &mut body.insts[t as usize];
                    let to = |v: &mut Value| {
                        if v.0 as usize == b {
                            *v = Value(s as u32);
                        }
                    };
                    match term.op {
                        Op::Br => to(&mut term.a),
                        Op::CondBr => {
                            to(&mut term.b);
                            to(&mut term.c);
                        }
                        Op::Switch => {
                            to(&mut term.b);
                            for case in body.switches[term.c.0 as usize].iter_mut().filter(|case| case.1 .0 as usize == b) {
                                case.1 = Block(s as u32);
                            }
                        }
                        _ => {}
                    }
                }
                gone[b] = true;
                changed = true;
            }
            for &p in &bphis {
                tally[p as usize] = 0;
            }
            body.blocks[b].phis = bphis;
            body.blocks[s].phis = sphis;
        }
        if !changed {
            break;
        }
    }
}

/// Branch folding and block merging, what SimplifyCFG does to the shape the passes above
/// leave: a conditional branch or switch whose condition is a constant, or whose edges all
/// end in one block, becomes a branch; an empty block on one edge of a conditional branch
/// whose other edge ends where it does becomes a select ([`fold_empty_sides`]), an empty
/// block entered from several ends in its successor ([`fold_empty_joins`]); then a block
/// that is the only successor of its only
/// predecessor joins that predecessor, its phis (one value each) forwarded to that value.
/// Every lane runs both blocks of such a pair or neither, so no wave operation, store or
/// reconvergence point sees a difference; the driver's compiler gets fewer blocks.
pub(super) fn simplify(body: &mut Body, m: &Module, keep: &[u32], exits: &[u32], handle: Ty) {
    let n = body.blocks.len();
    // a phi entry per edge from `p` into each block of `old`, now that `p` branches to `to` alone
    let drop_edges = |body: &mut Body, p: u32, old: &[u32], to: u32| {
        let mut seen: Vec<u32> = Vec::with_capacity(old.len());
        for &x in old {
            if seen.contains(&x) {
                continue;
            }
            seen.push(x);
            for &phi in &body.blocks[x as usize].phis {
                let k = body.insts[phi as usize].a.0 as usize;
                let mut kept = x != to;
                body.phis[k].retain(|inc| {
                    if inc.1 .0 != p {
                        return true;
                    }
                    let keep = !kept;
                    kept = true;
                    keep
                });
            }
        }
    };
    let mut succ: Vec<u32> = Vec::new();
    for b in 0..n {
        let Some(&t) = body.blocks[b].insts.last() else { continue };
        let inst = body.insts[t as usize];
        let int = |v: Value| match v.kind() {
            ValueKind::Const(i) => match m.consts.get(i as usize).map(|c| &c.c) {
                Some(Const::Int(x)) => Some(*x),
                Some(Const::Zero) => Some(0),
                _ => None,
            },
            _ => None,
        };
        let to = match inst.op {
            Op::CondBr if inst.b == inst.c => inst.b.0,
            Op::CondBr if inst.flags & flag::LOOP_UNROLL_DISABLE == 0 => match int(inst.a) {
                Some(x) => if x & 1 != 0 { inst.b.0 } else { inst.c.0 },
                None => continue,
            },
            Op::Switch => {
                let cases = &body.switches[inst.c.0 as usize];
                let first = cases.first().map_or(inst.b, |c| Value(c.1 .0));
                if first == inst.b && cases.iter().all(|c| c.1 .0 == inst.b.0) {
                    inst.b.0
                } else {
                    match int(inst.a) {
                        Some(x) => cases.iter().find(|c| c.0 == x).map_or(inst.b.0, |c| c.1 .0),
                        None => continue,
                    }
                }
            }
            _ => continue,
        };
        succ.clear();
        crate::bcwriter::successors(body, &inst, |s| succ.push(s));
        if to as usize >= n {
            continue;
        }
        body.insts[t as usize] = Inst { op: Op::Br, flags: inst.flags & !flag::HINTS, ty: inst.ty, a: Value(to), b: Value::NONE, c: Value::NONE };
        drop_edges(body, b as u32, &succ, to);
    }

    fold_empty_sides(body, keep, handle);
    fold_empty_joins(body, keep, exits);

    // predecessor edges from the blocks the writer keeps
    let mut work = Vec::new();
    let reachable = crate::bcwriter::reachable_blocks(body, &mut work);
    let mut npred = vec![0u32; n];
    for b in (0..n).filter(|&b| reachable[b]) {
        let Some(&t) = body.blocks[b].insts.last() else { continue };
        crate::bcwriter::successors(body, &body.insts[t as usize], |s| npred[s as usize] += 1);
    }
    let mut forwarded = false;
    let mut single: Vec<(u32, Value)> = Vec::new();
    for b in (0..n).filter(|&b| reachable[b]) {
        loop {
            let Some(&t) = body.blocks[b].insts.last() else { break };
            let br = body.insts[t as usize];
            let s = br.a.0 as usize;
            if br.op != Op::Br || br.flags & !flag::EFFECT != 0 || s == b || s == 0 || s >= n || npred[s] != 1 {
                break;
            }
            // each phi of the block takes the one value of its one edge
            single.clear();
            let ok = body.blocks[s].phis.iter().all(|&phi| {
                let inc = body.phi_incoming(Value::inst(phi));
                let Some(&(v, _)) = inc.first() else { return false };
                single.push((phi, v));
                v != Value::inst(phi) && inc.iter().all(|&(w, from)| w == v && from.0 as usize == b)
            });
            // a successor that already has an entry from `b` would get a second one
            succ.clear();
            if let Some(&st) = body.blocks[s].insts.last() {
                crate::bcwriter::successors(body, &body.insts[st as usize], |x| succ.push(x));
            }
            if !ok || succ.iter().any(|&x| body.blocks[x as usize].phis.iter().any(|&phi| body.phi_incoming(Value::inst(phi)).iter().any(|inc| inc.1 .0 as usize == b))) {
                break;
            }
            for &(phi, v) in &single {
                body.replace(Value::inst(phi), v);
                forwarded = true;
            }
            body.blocks[s].phis.clear();
            succ.sort_unstable();
            succ.dedup();
            for &x in &succ {
                for &phi in &body.blocks[x as usize].phis {
                    let k = body.insts[phi as usize].a.0 as usize;
                    for inc in body.phis[k].iter_mut().filter(|inc| inc.1 .0 as usize == s) {
                        inc.1 = Block(b as u32);
                    }
                }
            }
            let moved = std::mem::take(&mut body.blocks[s].insts);
            let into = &mut body.blocks[b].insts;
            into.pop();
            into.extend_from_slice(&moved);
            npred[s] = 0;
        }
    }
    if forwarded {
        apply_forwarding(body);
    }
}

/// SimplifyCFG's `FoldTwoEntryPHINode` for a side that holds nothing: `P: br c, E, M` with
/// `E: br M` becomes `P: br M`, and each phi of `M` takes `select c, (from E), (from P)`
/// from `P`, the select placed before `P`'s branch. Both values are there before the branch
/// and every lane reaches `M` either way, so nothing runs that did not run before and no
/// lane sees another value. Not done for the blocks of `keep` (a loop's preheader, latch and
/// exit), a branch marked `[branch]` or closing a loop, `dx.break`'s branch (a compare of
/// a load, which the driver must not fold), or a phi of type `handle`, which DXIL does not
/// allow a select of.
fn fold_empty_sides(body: &mut Body, keep: &[u32], handle: Ty) {
    let n = body.blocks.len();
    let crate::bcwriter::Live { blocks: reachable, insts: live } = crate::bcwriter::liveness(body);
    let is_live = |i: u32| live.get(i as usize).copied().unwrap_or(true);
    let mut npred = vec![0u32; n];
    let mut pred = vec![0usize; n];
    for b in (0..n).filter(|&b| reachable[b]) {
        let Some(&t) = body.blocks[b].insts.last() else { continue };
        crate::bcwriter::successors(body, &body.insts[t as usize], |s| {
            npred[s as usize] += 1;
            pred[s as usize] = b;
        });
    }
    let mut kept = vec![false; n];
    for &k in keep {
        kept[k as usize] = true;
    }
    let mut sels: Vec<(u32, Value)> = Vec::new();
    for e in 1..n {
        let data = &body.blocks[e];
        if !reachable[e] || kept[e] || npred[e] != 1 || data.phis.iter().any(|&p| is_live(p)) {
            continue;
        }
        let Some(&et) = data.insts.last() else { continue };
        if data.insts[..data.insts.len() - 1].iter().any(|&i| is_live(i)) {
            continue;
        }
        let br = body.insts[et as usize];
        let mm = br.a.0 as usize;
        if br.op != Op::Br || br.flags & flag::LOOP_UNROLL_DISABLE != 0 || mm == e || mm >= n {
            continue;
        }
        let p = pred[e];
        let Some(&pt) = body.blocks[p].insts.last() else { continue };
        let cb = body.insts[pt as usize];
        let on_true = cb.b.0 as usize == e;
        let other = if on_true { cb.c.0 } else { cb.b.0 } as usize;
        if cb.op != Op::CondBr || other != mm || p == e || kept[p] || cb.flags & (flag::HINT_BRANCH | flag::LOOP_UNROLL_DISABLE) != 0 {
            continue;
        }
        if cb.a.is_inst() {
            let c = body.insts[cb.a.0 as usize];
            if matches!(c.op, Op::ICmp(_)) && c.a.is_inst() && body.insts[c.a.0 as usize].op == Op::Load {
                continue;
            }
        }
        // each phi of the target: one entry from each edge
        sels.clear();
        let ok = body.blocks[mm].phis.iter().all(|&phi| {
            let inc = body.phi_incoming(Value::inst(phi));
            let from = |b: usize| {
                let mut it = inc.iter().filter(|x| x.1 .0 as usize == b);
                let first = it.next().map(|x| x.0);
                if it.next().is_some() { None } else { first }
            };
            match (from(e), from(p)) {
                (Some(ve), Some(vp)) if body.insts[phi as usize].ty != handle || !is_live(phi) => {
                    sels.push((phi, if ve == vp { ve } else { Value::NONE }));
                    true
                }
                _ => false,
            }
        });
        if !ok {
            continue;
        }
        for k in 0..sels.len() {
            let (phi, same) = sels[k];
            let kq = body.insts[phi as usize].a.0 as usize;
            let ve = body.phis[kq].iter().find(|x| x.1 .0 as usize == e).unwrap().0;
            let vp = body.phis[kq].iter().find(|x| x.1 .0 as usize == p).unwrap().0;
            let v = if !same.is_none() || !is_live(phi) {
                if same.is_none() { vp } else { same }
            } else {
                let i = body.insts.len() as u32;
                let (t, f) = if on_true { (ve, vp) } else { (vp, ve) };
                body.insts.push(Inst { op: Op::Select, flags: 0, ty: body.insts[phi as usize].ty, a: cb.a, b: t, c: f });
                let at = body.blocks[p].insts.len() - 1;
                body.blocks[p].insts.insert(at, i);
                Value::inst(i)
            };
            body.phis[kq].retain(|x| x.1 .0 as usize != e);
            for x in body.phis[kq].iter_mut().filter(|x| x.1 .0 as usize == p) {
                x.0 = v;
            }
        }
        let pt = *body.blocks[p].insts.last().unwrap();
        body.insts[pt as usize] = Inst { op: Op::Br, flags: cb.flags & !flag::HINTS, ty: cb.ty, a: Value(mm as u32), b: Value::NONE, c: Value::NONE };
        npred[e] = 0;
        npred[mm] = u32::MAX;
    }
}

/// SimplifyCFG's `TryToSimplifyUncondBranchFromEmptyBlock`: a block that holds nothing but
/// a branch to `T` and is entered from several blocks, none of which also enters `T`, is
/// left out: its predecessors branch to `T`, and each phi of `T` takes the value it took
/// from the block once per edge it gets in its place. Nothing ran in the block, and every lane
/// that entered it went on to `T`, so no lane runs anything else or meets other lanes
/// anywhere else. Not done for a loop's preheader or latch (`keep` without `exits`) nor for a
/// branch that closes a loop.
fn fold_empty_joins(body: &mut Body, keep: &[u32], exits: &[u32]) {
    let n = body.blocks.len();
    let crate::bcwriter::Live { blocks: reachable, insts: live } = crate::bcwriter::liveness(body);
    let is_live = |i: u32| live.get(i as usize).copied().unwrap_or(true);
    let mut preds: Vec<Vec<u32>> = vec![Vec::new(); n];
    for b in (0..n).filter(|&b| reachable[b]) {
        let Some(&t) = body.blocks[b].insts.last() else { continue };
        crate::bcwriter::successors(body, &body.insts[t as usize], |s| preds[s as usize].push(b as u32));
    }
    let mut kept = vec![false; n];
    for &k in keep {
        kept[k as usize] = true;
    }
    for &x in exits {
        kept[x as usize] = false;
    }
    let mut succ: Vec<u32> = Vec::new();
    for e in 1..n {
        let data = &body.blocks[e];
        if !reachable[e] || kept[e] || preds[e].len() < 2 || data.phis.iter().any(|&p| is_live(p)) {
            continue;
        }
        let Some(&et) = data.insts.last() else { continue };
        if data.insts[..data.insts.len() - 1].iter().any(|&i| is_live(i)) {
            continue;
        }
        let br = body.insts[et as usize];
        let t = br.a.0 as usize;
        if br.op != Op::Br || br.flags & flag::LOOP_UNROLL_DISABLE != 0 || t == e || t >= n || !reachable[t] {
            continue;
        }
        // a predecessor that also enters `t` would give a phi there two entries from one block
        let shared = preds[e].iter().any(|&p| {
            succ.clear();
            let pt = *body.blocks[p as usize].insts.last().unwrap();
            crate::bcwriter::successors(body, &body.insts[pt as usize], |x| succ.push(x));
            p as usize == e || succ.contains(&(t as u32))
        });
        let single = body.blocks[t].phis.iter().all(|&phi| body.phi_incoming(Value::inst(phi)).iter().filter(|x| x.1 .0 as usize == e).count() == 1);
        if shared || !single {
            continue;
        }
        let from = std::mem::take(&mut preds[e]);
        for &phi in &body.blocks[t].phis {
            let k = body.insts[phi as usize].a.0 as usize;
            let j = body.phis[k].iter().position(|x| x.1 .0 as usize == e).unwrap();
            let (v, _) = body.phis[k].swap_remove(j);
            body.phis[k].extend(from.iter().map(|&p| (v, Block(p))));
        }
        let mut last = u32::MAX;
        for &p in &from {
            if p == last {
                continue;
            }
            last = p;
            let pt = *body.blocks[p as usize].insts.last().unwrap();
            let term = &mut body.insts[pt as usize];
            let to = |v: &mut Value| {
                if v.0 as usize == e {
                    *v = Value(t as u32);
                }
            };
            match term.op {
                Op::Br => to(&mut term.a),
                Op::CondBr => {
                    to(&mut term.b);
                    to(&mut term.c);
                }
                Op::Switch => {
                    to(&mut term.b);
                    for case in body.switches[term.c.0 as usize].iter_mut().filter(|case| case.1 .0 as usize == e) {
                        case.1 = Block(t as u32);
                    }
                }
                _ => {}
            }
        }
        preds[t].retain(|&p| p as usize != e);
        preds[t].extend_from_slice(&from);
    }
}
