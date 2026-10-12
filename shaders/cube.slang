struct CubeVertex
{
    float3 position;
    float  ao;
    float3 normal;
    uint   face_id;
    float3 color;
    float  roughness;
    float2 uv;
    float2 pad;
};

struct PushConstants
{
    column_major float4x4 mvp;
    float4                model_offset_and_time;
    float4                camera_pos;
    CubeVertex*           vertex_buffer;
};

[[vk::push_constant]]
PushConstants pc;

struct VSOutput
{
    float4 position  : SV_Position;
    float3 world_pos : TEXCOORD0;
    float3 normal    : TEXCOORD1;
    float3 color     : TEXCOORD2;
    float2 uv        : TEXCOORD3;
    float  ao        : TEXCOORD4;
};

[shader("vertex")]
VSOutput vertexMain(uint vertex_id : SV_VertexID)
{
    CubeVertex v = pc.vertex_buffer[vertex_id];

    float3 world_pos = v.position + pc.model_offset_and_time.xyz;

    VSOutput out_vert;
    out_vert.position  = mul(pc.mvp, float4(world_pos, 1.0));
    out_vert.world_pos = world_pos;
    out_vert.normal    = v.normal;
    out_vert.color     = v.color;
    out_vert.uv        = v.uv;
    out_vert.ao        = v.ao;
    return out_vert;
}

[shader("fragment")]
float4 fragmentMain(VSOutput input) : SV_Target0
{
    float3 N = normalize(input.normal);
    float3 L = normalize(float3(0.45, 0.85, 0.35));
    float3 V = normalize(pc.camera_pos.xyz - input.world_pos);
    float3 H = normalize(L + V);

    // 16x16 grid pattern on each face
    float2 texel_coord = floor(input.uv * 16.0);
    float  checker     = fmod(texel_coord.x + texel_coord.y, 2.0);
    float  texel_tint  = lerp(0.94, 1.06, checker);

    // Face border highlight
    float2 edge_dist = min(input.uv, 1.0 - input.uv);
    float  border    = smoothstep(0.0, 0.04, min(edge_dist.x, edge_dist.y));
    float  edge_shade = lerp(0.78, 1.0, border);

    // Sun and ambient lighting
    float ndotl   = max(dot(N, L), 0.0);
    float sky_hemi = 0.5 * N.y + 0.5;
    float3 ambient = lerp(float3(0.22, 0.24, 0.30), float3(0.55, 0.62, 0.75), sky_hemi);
    float  spec    = pow(max(dot(N, H), 0.0), 32.0) * 0.15;

    float3 base_color = input.color * texel_tint * edge_shade;
    float3 lit_color  = base_color * (ambient + ndotl * float3(1.0, 0.96, 0.88)) * input.ao + spec;

    return float4(lit_color, 1.0);
}
