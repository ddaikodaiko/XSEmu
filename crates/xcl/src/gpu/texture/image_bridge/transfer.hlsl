#ifndef IMAGE_BRIDGE_ROOT
#define IMAGE_BRIDGE_ROOT
#endif

cbuffer BridgeDispatch : register(b0) {
    uint firstGuestVpn;
    uint pageCount;
    uint directoryRootSrv;
    uint umaBaseHi;
    uint umaFirstUav;
    uint baseX;
    uint baseY;
    uint baseZ;
};

ByteAddressBuffer plan : register(t0);
RWByteAddressBuffer captured : register(u0);
RWByteAddressBuffer footprint : register(u1);

static const uint IMAGE_PAGE_PHYSICAL = 1;
static const uint IMAGE_PAGE_UMA = 2;
static const uint IMAGE_PAGE_NATIVE = 3;

#ifdef IMAGE_BRIDGE_SNAPSHOT
#include "image_page_resolver.hlsli"

IMAGE_BRIDGE_ROOT
[numthreads(64, 1, 1)]
void SnapshotImagePages(uint3 id : SV_DispatchThreadID) {
    if (baseX >= pageCount || id.x >= pageCount - baseX) return;
    uint page = baseX + id.x;
    uint4 owner = 0;
    if (page <= 0xffffffffu - firstGuestVpn)
        owner = ResolveImagePage(firstGuestVpn + page, directoryRootSrv, umaBaseHi, umaFirstUav, baseY);
    captured.Store4(page * 16, owner);
}
#else
uint axisOffset(uint value, uint word) {
    uint offset = 0;
    for (uint bits = value & 0xffffu; bits != 0; bits &= bits - 1)
        offset ^= plan.Load(4 * (word + firstbitlow(bits)));
    return offset;
}

uint64_t guestOffset(uint3 p) {
    uint2 xy = p.xy + plan.Load2(4 * 4);
    uint4 logs = plan.Load4(4 * 6);
    uint4 tile = plan.Load4(4 * 10);
    uint64_t block = (uint64_t)(xy.y >> logs.y) * tile.x + (xy.x >> logs.x);
    uint inner = tile.y ^ axisOffset(xy.x, 20) ^ axisOffset(xy.y, 36) ^ axisOffset(p.z + tile.w, 52);
    return (uint64_t)plan.Load(4 * 14) + (uint64_t)(p.z >> logs.z) * tile.z + (block << logs.w) + inner;
}

void storePart(RWByteAddressBuffer dst, uint at, uint n, uint value) {
    if (n == 4) {
        dst.Store(at, value);
        return;
    }
    uint shift = (at & 3) * 8;
    uint mask = ((1u << (n * 8)) - 1) << shift;
    uint field = (value << shift) & mask;
    uint expected = 0;
    uint observed;
    [loop] do {
        dst.InterlockedCompareExchange(at & ~3u, expected, (expected & ~mask) | field, observed);
        if (observed == expected) break;
        expected = observed;
    } while (true);
}

uint4 capturedPage(uint firstPage, uint64_t guest) {
    uint64_t page = firstPage + (guest >> 16);
    if (page < pageCount)
        return captured.Load4((uint)page * 16);
    return 0;
}

uint convertBgra16(uint value, uint conversion, bool expand) {
    uint4 bits = conversion == 1 ? uint4(5, 5, 5, 1) : uint4(4, 4, 4, 4);
    uint4 limits = (1u << bits) - 1;
    uint4 shift = uint4(bits.b + bits.g, bits.b, 0, bits.b + bits.g + bits.r);
    uint4 rgbaShift = uint4(0, 8, 16, 24);
    uint4 lanes;
    if (expand) {
        lanes = ((value >> shift) & limits) * 255;
        lanes = ((lanes + (limits >> 1)) / limits) << rgbaShift;
    } else {
        lanes = ((value >> rgbaShift) & 255) * limits;
        lanes = ((lanes + 127) / 255) << shift;
    }
    return lanes.r | lanes.g | lanes.b | lanes.a;
}

uint sharedToHalf(uint mantissa, uint exponent) {
    if (!mantissa) return 0;
    uint top = firstbithigh(mantissa);
    if (top + exponent < 10) return mantissa << exponent;
    return ((top + exponent - 9) << 10) | ((mantissa << (10 - top)) & 1023u);
}

uint2 expandRgb9e5(uint bits) {
    uint e = bits >> 27;
    uint r = sharedToHalf(bits & 511, e);
    uint g = sharedToHalf((bits >> 9) & 511, e);
    uint b = sharedToHalf((bits >> 18) & 511, e);
    return uint2(r | (g << 16), b | 0x3c000000u);
}

uint sharedMantissa(uint halfBits, uint exponent) {
    uint halfExp = halfBits >> 10;
    uint significant = (halfBits & 1023u) + (halfExp == 0 ? 0u : 1024u);
    uint shift = exponent - (halfExp == 0 ? 0u : halfExp - 1u);
    if (!shift) return significant;
    uint integral = significant >> shift;
    uint tail = significant & ((1u << shift) - 1);
    uint midpoint = 1u << (shift - 1);
    return integral + (uint)(tail > midpoint || (tail == midpoint && (integral & 1u) != 0));
}

uint packRgb9e5(uint2 rgba) {
    uint3 h = uint3(rgba.x & 65535, rgba.x >> 16, rgba.y & 65535);
    h = select(h > 0x7c00u, 0u, min(h, 0x7bfcu));
    uint peak = max(h.r, max(h.g, h.b));
    uint e = peak >= 1024 ? (peak >> 10) + 1 : (uint)(peak >= 512);
    if (sharedMantissa(peak, e) == 512) ++e;
    return sharedMantissa(h.r, e) | (sharedMantissa(h.g, e) << 9) |
           (sharedMantissa(h.b, e) << 18) | (e << 27);
}

void transferElement(uint3 id, bool gather) {
    uint4 size = plan.Load4(0);
    uint3 base = uint3(baseX, baseY, baseZ);
    if (any(base >= size.xyz) || any(id >= size.xyz - base)) return;
    uint3 p = base + id;
    uint bytes = 1u << size.w;
    uint4 host = plan.Load4(4 * 16);
    uint hostBytes = host.w == 0 ? bytes : (host.w == 3 ? 8 : 4);
    uint hostAt = host.x + p.z * host.z + p.y * host.y + p.x * hostBytes;
    uint64_t guest = guestOffset(p);
    uint firstPage = plan.Load(4 * 15);
    if (host.w != 0) {
        uint4 owner = capturedPage(firstPage, guest);
        if (owner.w == IMAGE_PAGE_NATIVE) return;
        if (owner.w != IMAGE_PAGE_PHYSICAL && owner.w != IMAGE_PAGE_UMA) {
            if (gather) {
                if (host.w == 3) footprint.Store2(hostAt, uint2(0, 0));
                else footprint.Store(hostAt, 0);
            }
            return;
        }
        RWByteAddressBuffer canon = ResourceDescriptorHeap[NonUniformResourceIndex(owner.x)];
        uint at = owner.y + ((uint)guest & 0xffffu);
        if (host.w == 3) {
            if (gather) footprint.Store2(hostAt, expandRgb9e5(canon.Load(at)));
            else canon.Store(at, packRgb9e5(footprint.Load2(hostAt)));
        } else if (gather) {
            footprint.Store(hostAt, convertBgra16(canon.Load(at & ~3u) >> ((at & 2u) * 8), host.w, true));
        } else {
            storePart(canon, at, 2, convertBgra16(footprint.Load(hostAt), host.w, false));
        }
        return;
    }
    while (bytes != 0) {
        uint inPage = (uint)guest & 0xffffu;
        uint4 owner = capturedPage(firstPage, guest);
        uint chunk = min(bytes, 0x10000u - inPage);
        uint raw = owner.y + inPage;
        guest += chunk;
        bytes -= chunk;
        if (owner.w == IMAGE_PAGE_NATIVE) {
            hostAt += chunk;
            continue;
        }
        while (chunk != 0) {
            uint n = min(chunk, min(4u - (hostAt & 3u), 4u - (raw & 3u)));
            uint value = 0;
            if (owner.w == IMAGE_PAGE_PHYSICAL || owner.w == IMAGE_PAGE_UMA) {
                RWByteAddressBuffer canon = ResourceDescriptorHeap[NonUniformResourceIndex(owner.x)];
                if (gather)
                    value = canon.Load(raw & ~3u) >> ((raw & 3u) * 8);
                else
                    storePart(canon, raw, n, footprint.Load(hostAt & ~3u) >> ((hostAt & 3u) * 8));
            }
            if (gather)
                storePart(footprint, hostAt, n, value);
            hostAt += n;
            raw += n;
            chunk -= n;
        }
    }
}

IMAGE_BRIDGE_ROOT
[numthreads(64, 1, 1)]
void GatherImage(uint3 id : SV_DispatchThreadID) {
    transferElement(id, true);
}

IMAGE_BRIDGE_ROOT
[numthreads(64, 1, 1)]
void ScatterImage(uint3 id : SV_DispatchThreadID) {
    transferElement(id, false);
}
#endif
