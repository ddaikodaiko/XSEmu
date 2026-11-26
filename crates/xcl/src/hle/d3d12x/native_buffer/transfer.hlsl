cbuffer TransferConstants : register(b0) {
    uint locator;
    uint guestPage;
    uint fallbackView;
    uint fallbackPageOffset;
    uint planOffset;
    uint guestOffset;
    uint scratchOffset;
    uint byteCount;
};

RWByteAddressBuffer plan : register(u0);
RWByteAddressBuffer scratch : register(u1);

#define NATIVE_ROOT "RootFlags(CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED), RootConstants(num32BitConstants=8,b0), UAV(u0), UAV(u1)"

[RootSignature(NATIVE_ROOT)]
[numthreads(1, 1, 1)]
void Capture() {
    ByteAddressBuffer selection = ResourceDescriptorHeap[locator];
    uint2 p = xcl_lookup_gpu_pte(uint2(guestPage << 16u, guestPage >> 16u), selection.Load(0));
    uint2 target = uint2(0u, 0u);
    if (p.x == XCL_PTE_ABSENT) {
        target = uint2(fallbackView, fallbackPageOffset);
    } else if (p.x == XCL_PTE_MAPPED) {
        target = uint2(p.y & 0xfffffu, ((p.y >> 20u) & 127u) << 16u);
    }
    plan.Store2(planOffset, target);
}

void StoreMasked(RWByteAddressBuffer dst, uint address, uint mask, uint bits) {
    if (mask == 0xffffffffu) {
        dst.Store(address, bits);
        return;
    }
    uint oldValue = dst.Load(address);
    [allow_uav_condition] for (;;) {
        uint observed;
        dst.InterlockedCompareExchange(address, oldValue, (oldValue & ~mask) | bits, observed);
        if (observed == oldValue) return;
        oldValue = observed;
    }
}

void TransferWord(uint index, bool gather) {
    uint2 target = plan.Load2(planOffset);
    if (!gather && target.x == 0u) return;
    uint first = (gather ? scratchOffset : guestOffset) & 3u;
    uint outWord = 4u * index;
    if (outWord >= first + byteCount) return;
    uint bytes = 0u, mask = 0u;
    if (outWord >= first && outWord - first + 4u <= byteCount && ((scratchOffset ^ guestOffset) & 3u) == 0u) {
        uint a = target.y + (guestOffset & ~3u) + outWord;
        uint rawOffset = (scratchOffset & ~3u) + outWord;
        if (gather && target.x == 0u) {
            scratch.Store(rawOffset, 0u);
        } else {
            RWByteAddressBuffer resource = ResourceDescriptorHeap[NonUniformResourceIndex(target.x)];
            if (gather) scratch.Store(rawOffset, resource.Load(a));
            else resource.Store(a, scratch.Load(rawOffset));
        }
        return;
    }
    [unroll] for (uint j = 0u; j < 4u; ++j) {
        uint pos = outWord + j;
        if (pos < first || pos-first >= byteCount) continue;
        uint rel = pos-first;
        uint a = gather ? target.y + guestOffset + rel : scratchOffset + rel;
        uint value = 0u;
        if (!gather) {
            value = scratch.Load(a & ~3u);
        } else if (target.x != 0u) {
            RWByteAddressBuffer src = ResourceDescriptorHeap[NonUniformResourceIndex(target.x)];
            value = src.Load(a & ~3u);
        }
        bytes |= ((value >> (8u * (a & 3u))) & 255u) << (j * 8u);
        mask |= 255u << (j * 8u);
    }
    if (gather) {
        StoreMasked(scratch, (scratchOffset & ~3u) + outWord, mask, bytes);
    } else {
        RWByteAddressBuffer dst = ResourceDescriptorHeap[NonUniformResourceIndex(target.x)];
        StoreMasked(dst, target.y + (guestOffset & ~3u) + outWord, mask, bytes);
    }
}

[RootSignature(NATIVE_ROOT)]
[numthreads(64, 1, 1)]
void Gather(uint3 id : SV_DispatchThreadID) { TransferWord(id.x, true); }

[RootSignature(NATIVE_ROOT)]
[numthreads(64, 1, 1)]
void Scatter(uint3 id : SV_DispatchThreadID) { TransferWord(id.x, false); }
