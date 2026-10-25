#pragma warning(disable: 41012)

struct GpuChunkDraw
{
    int4     origin_and_quads; // xyz = chunk world origin (blocks), w = quad count (0 = free slot)
    uint*    vertex_buffer;
    uint64_t reserved;
};

struct DrawIndexedIndirectCommand
{
    uint index_count;
    uint instance_count;
    uint first_index;
    int  vertex_offset;
    uint first_instance;
};

struct CullStats
{
    uint draw_count;
    uint quad_count;
};

struct CullPushConstants
{
    float4                      planes[6];
    GpuChunkDraw*               chunk_draws;
    DrawIndexedIndirectCommand* commands;
    CullStats*                  stats;
    uint                        slot_count;
    uint                        flags;
};

static const uint CULL_FLAG_FRUSTUM = 1u;
static const float CHUNK_EXTENT     = 32.0;

[[vk::push_constant]]
CullPushConstants pc;

[shader("compute")]
[numthreads(64, 1, 1)]
void cullMain(uint3 dispatch_id : SV_DispatchThreadID)
{
    const uint slot = dispatch_id.x;

    bool visible = false;
    uint quads   = 0u;

    if (slot < pc.slot_count)
    {
        const int4 origin_and_quads = pc.chunk_draws[slot].origin_and_quads;
        quads   = uint(origin_and_quads.w);
        visible = quads > 0u;

        if (visible && (pc.flags & CULL_FLAG_FRUSTUM) != 0u)
        {
            const float3 aabb_min = float3(origin_and_quads.xyz);
            const float3 aabb_max = aabb_min + CHUNK_EXTENT;

            [unroll]
            for (uint i = 0u; i < 6u; ++i)
            {
                const float4 plane = pc.planes[i];
                const float3 p     = float3(
                    plane.x > 0.0 ? aabb_max.x : aabb_min.x,
                    plane.y > 0.0 ? aabb_max.y : aabb_min.y,
                    plane.z > 0.0 ? aabb_max.z : aabb_min.z
                );
                if (dot(plane.xyz, p) + plane.w < 0.0) visible = false;
            }
        }
    }

    const uint wave_visible = WaveActiveCountBits(visible);
    if (wave_visible == 0u) return;

    const uint lane_offset = WavePrefixCountBits(visible);
    const uint wave_quads  = WaveActiveSum(visible ? quads : 0u);

    uint base = 0u;
    if (WaveIsFirstLane())
    {
        InterlockedAdd(pc.stats->draw_count, wave_visible, base);
        InterlockedAdd(pc.stats->quad_count, wave_quads);
    }
    base = WaveReadLaneFirst(base);

    if (visible)
    {
        DrawIndexedIndirectCommand cmd;
        cmd.index_count    = quads * 6u;
        cmd.instance_count = 1u;
        cmd.first_index    = 0u;
        cmd.vertex_offset  = 0;
        cmd.first_instance = slot; // Read back in the vertex shader as the instance index
        pc.commands[base + lane_offset] = cmd;
    }
}
