cbuffer SampleFootprint : register(b0) {
    uint width;
    uint height;
    uint sampleCount;
};

Texture2DMS<float4> msColor : register(t0);
StructuredBuffer<float4> logicalInput : register(t1);
RWStructuredBuffer<float4> logicalOutput : register(u0);

uint logicalIndex(uint2 pixel, uint sampleId) {
    return (pixel.y * width + pixel.x) * sampleCount + sampleId;
}

[numthreads(8, 8, 1)]
void ExportColorSamples(uint3 id : SV_DispatchThreadID) {
    uint w, h, samples;
    msColor.GetDimensions(w, h, samples);
    if (any(uint3(w, h, samples) != uint3(width, height, sampleCount)) ||
        any(id >= uint3(w, h, samples))) return;
    logicalOutput[logicalIndex(id.xy, id.z)] = msColor.Load(id.xy, id.z);
}

float4 SampleFullscreenVertex(uint vertex : SV_VertexID) : SV_Position {
    float2 xy = float2((vertex << 1) & 2, vertex & 2);
    return float4(xy * float2(2, -2) + float2(-1, 1), 0, 1);
}

float4 RestoreColorSample(float4 position : SV_Position, uint sampleId : SV_SampleIndex) : SV_Target0 {
    uint2 pixel = uint2(position.xy);
    if (any(pixel >= uint2(width, height)) || sampleId >= sampleCount) discard;
    return logicalInput[logicalIndex(pixel, sampleId)];
}
