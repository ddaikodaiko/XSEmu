//! One fetch of each descriptor-heap handle per function.
//!
//! A resource variable holds a heap index, and the handle is fetched again where the variable
//! is read ([`super::ResState`]); the value numbering joins the fetches that dominate one
//! another, but the fetches of one heap element in sibling branches stay apart, a pair of
//! calls each (`createHandleFromHeap`, `annotateHandle`). Both calls are `readnone`: they read
//! no memory and have no effect, and their value depends on their operands alone. So one pair
//! per (index, flags, properties) is enough, placed straight after the index is defined,
//! which dominates every fetch that uses it; the others are forwarded to it. Not done for a
//! fetch marked `NonUniformResourceIndex` (its handle differs between lanes, and the driver
//! loops over the lanes at each use) nor for an element fetched once.

use crate::consts::OpCode;
use crate::ir::{flag, Body, Const, Module, Op, Value};
use std::collections::HashMap;

pub(super) fn hoist_heap_handles(body: &mut Body, m: &Module) {
    let crate::bcwriter::Live { blocks: reachable, insts: live } = crate::bcwriter::liveness(body);
    let op_of = |x: &crate::ir::Inst, body: &Body| -> Option<u64> {
        if x.op != Op::Call || x.flags & flag::EFFECT != 0 {
            return None;
        }
        m.as_int(*body.operands_of(x).first()?)
    };
    let zero = |v: Value| matches!(m.as_const(v).map(|c| &c.c), Some(Const::Int(0) | Const::Zero));
    // where each instruction of a reachable block is: (block, position)
    let mut at = vec![(u32::MAX, 0u32); body.insts.len()];
    for (b, data) in body.blocks.iter().enumerate().filter(|(b, _)| reachable[*b]) {
        for &p in &data.phis {
            at[p as usize] = (b as u32, u32::MAX);
        }
        for (k, &i) in data.insts.iter().enumerate() {
            at[i as usize] = (b as u32, k as u32);
        }
    }
    let create = OpCode::CreateHandleFromHeap as u64;
    let annotate = OpCode::AnnotateHandle as u64;
    // the fetches in program order of blocks, so that the first of a group is a natural pick
    let order: Vec<u32> = body.blocks.iter().enumerate().filter(|(b, _)| reachable[*b]).flat_map(|(_, d)| d.insts.iter().copied()).collect();
    let mut first: HashMap<(Value, Vec<Value>), u32> = HashMap::new();
    let mut canon = vec![u32::MAX; body.insts.len()];
    let mut grouped = vec![0u32; body.insts.len()];
    for &i in &order {
        let x = body.insts[i as usize];
        if !live[i as usize] || op_of(&x, body) != Some(create) {
            continue;
        }
        let args = body.operands_of(&x);
        if args.len() != 4 || !zero(args[3]) {
            continue;
        }
        let c = *first.entry((x.a, args.to_vec())).or_insert(i);
        canon[i as usize] = c;
        grouped[c as usize] += 1;
    }
    // the annotations of one fetch, by the fetch they annotate once the fetches are joined
    let mut first_ann: HashMap<(Value, u32, Vec<Value>), u32> = HashMap::new();
    let mut ann_of: HashMap<u32, Vec<u32>> = HashMap::new();
    let mut ann_canon = vec![u32::MAX; body.insts.len()];
    let mut ann_grouped = vec![0u32; body.insts.len()];
    for &i in &order {
        let x = body.insts[i as usize];
        if !live[i as usize] || op_of(&x, body) != Some(annotate) {
            continue;
        }
        let args = body.operands_of(&x);
        let Some(&h) = args.get(1) else { continue };
        if !h.is_inst() || canon[h.0 as usize] == u32::MAX {
            continue;
        }
        let c = canon[h.0 as usize];
        let key = (x.a, c, args.iter().enumerate().filter(|(k, _)| *k != 1).map(|(_, &v)| v).collect());
        let a = *first_ann.entry(key).or_insert_with(|| {
            ann_of.entry(c).or_default().push(i);
            i
        });
        ann_canon[i as usize] = a;
        ann_grouped[a as usize] += 1;
    }
    // a group moves when it joins something: two fetches, or two annotations of one
    let moves: Vec<u32> = first.values().copied().filter(|&c| grouped[c as usize] > 1 || ann_of[&c].iter().any(|&a| ann_grouped[a as usize] > 1)).collect();
    if moves.is_empty() {
        return;
    }
    let mut moved = vec![false; body.insts.len()];
    // inserted after an instruction (Some) or at the start of a block (None, after the allocas)
    let mut after: HashMap<u32, Vec<u32>> = HashMap::new();
    let mut at_start: HashMap<u32, Vec<u32>> = HashMap::new();
    let mut sorted = moves;
    sorted.sort_unstable();
    for c in sorted {
        let index = body.operands_of(&body.insts[c as usize])[1];
        let mut run = vec![c];
        run.extend(ann_of[&c].iter().copied());
        for &i in &run {
            moved[i as usize] = true;
        }
        if index.is_inst() && at[index.0 as usize].1 != u32::MAX {
            after.entry(index.0).or_default().extend(run);
        } else {
            let b = if index.is_inst() { at[index.0 as usize].0 } else { 0 };
            at_start.entry(b).or_default().extend(run);
        }
    }
    for i in 0..body.insts.len() as u32 {
        if canon[i as usize] != u32::MAX && canon[i as usize] != i {
            body.replace(Value::inst(i), Value::inst(canon[i as usize]));
        }
        if ann_canon[i as usize] != u32::MAX && ann_canon[i as usize] != i {
            body.replace(Value::inst(i), Value::inst(ann_canon[i as usize]));
        }
    }
    let touched: Vec<u32> = {
        let mut t: Vec<u32> = after.keys().map(|&i| at[i as usize].0).chain(at_start.keys().copied()).chain((0..body.insts.len()).filter(|&i| moved[i]).map(|i| at[i].0)).collect();
        t.sort_unstable();
        t.dedup();
        t
    };
    for b in touched {
        let old = std::mem::take(&mut body.blocks[b as usize].insts);
        let mut list = Vec::with_capacity(old.len() + 8);
        let lead = old.iter().take_while(|&&i| body.insts[i as usize].op == Op::Alloca).count();
        list.extend_from_slice(&old[..lead]);
        if let Some(run) = at_start.get(&b) {
            list.extend_from_slice(run);
        }
        for &i in &old[lead..] {
            if !moved[i as usize] {
                list.push(i);
            }
            if let Some(run) = after.get(&i) {
                list.extend_from_slice(run);
            }
        }
        body.blocks[b as usize].insts = list;
    }
    super::cfg::apply_forwarding(body);
}
