struct GpuBlockMaterial
{
    float4 face_color_and_roughness[6];
    float4 emission_and_opacity;
};

struct PushConstants
{
    column_major float4x4 mvp;
    float4                chunk_origin_and_time;
    float4                camera_pos;
    uint*                 vertex_buffer;
    GpuBlockMaterial*     material_buffer;
};

[[vk::push_constant]]
PushConstants pc;

static const float3 FACE_NORMALS[6] = {
    float3(-1.0,  0.0,  0.0), // Left (-X)
    float3( 1.0,  0.0,  0.0), // Right (+X)
    float3( 0.0, -1.0,  0.0), // Down (-Y)
    float3( 0.0,  1.0,  0.0), // Up (+Y)
    float3( 0.0,  0.0, -1.0), // Back (-Z)
    float3( 0.0,  0.0,  1.0)  // Front (+Z)
};

static const float AO_CURVE[4] = {
    0.32, // 3 neighbor blocks occluding corner
    0.55, // 2 neighbor blocks occluding corner
    0.78, // 1 neighbor block occluding corner
    1.00  // unoccluded corner
};

// Distance threshold beyond which face textures are replaced by a unified face color
static const float TEXTURE_CUTOFF_DIST = 128.0;

struct VSOutput
{
    float4 position                 : SV_Position;
    float3 world_pos                : TEXCOORD0;
    float3 normal                   : TEXCOORD1;
    float3 color                    : TEXCOORD2;
    float2 uv                       : TEXCOORD3;
    float  ao                       : TEXCOORD4;
    float  roughness                : TEXCOORD5;
    float  emission                 : TEXCOORD6;
    nointerpolation uint visibility : TEXCOORD7;
};

[shader("vertex")]
VSOutput vertexMain(uint vertex_id : SV_VertexID)
{
    uint packed = pc.vertex_buffer[vertex_id];

    uint vx       =  packed         & 0x3Fu;
    uint vy       = (packed >> 6u)  & 0x3Fu;
    uint vz       = (packed >> 12u) & 0x3Fu;
    uint norm_id  = min((packed >> 18u) & 0x07u, 5u);
    uint ao_level = (packed >> 21u) & 0x03u;
    uint uv_id    = (packed >> 23u) & 0x03u;
    uint block_id = (packed >> 25u) & 0x7Fu;

    float3 local_pos = float3(float(vx), float(vy), float(vz));
    float3 world_pos = local_pos + pc.chunk_origin_and_time.xyz;

    GpuBlockMaterial mat = pc.material_buffer[block_id];
    float4 color_rough   = mat.face_color_and_roughness[norm_id];

    VSOutput out_vert;
    out_vert.position   = mul(pc.mvp, float4(world_pos, 1.0));
    out_vert.world_pos  = world_pos;
    out_vert.normal     = FACE_NORMALS[norm_id];
    out_vert.color      = color_rough.xyz;
    out_vert.uv         = float2(float(uv_id & 1u), float((uv_id >> 1u) & 1u));
    out_vert.ao         = AO_CURVE[ao_level];
    out_vert.roughness  = color_rough.w;
    out_vert.emission   = mat.emission_and_opacity.x;
    out_vert.visibility = uint(mat.emission_and_opacity.z + 0.5);
    return out_vert;
}

[shader("fragment")]
float4 fragmentMain(VSOutput input) : SV_Target0
{
    // Distance from camera to chunk center
    float3 chunk_center = pc.chunk_origin_and_time.xyz + float3(16.0, 16.0, 16.0);
    float  chunk_dist   = length(chunk_center - pc.camera_pos.xyz);

    // Unified face color
    float3 base_color = input.color;

    if (chunk_dist < TEXTURE_CUTOFF_DIST)
    {
        // 16x16 checkerboard pattern on each face
        float2 texel_coord = floor(clamp(input.uv, 0.0, 0.9999) * 16.0);
        float  checker     = fmod(texel_coord.x + texel_coord.y, 2.0);

        // Cutout blocks
        if (input.visibility == 1u && checker < 0.5)
        {
            discard;
        }

        float  texel_tint = lerp(0.96, 1.04, checker);
        float2 edge_dist  = min(input.uv, 1.0 - input.uv);
        float  border     = smoothstep(0.0, 0.035, min(edge_dist.x, edge_dist.y));
        float  edge_shade = lerp(0.86, 1.0, border);

        base_color *= (texel_tint * edge_shade);
    }

    float3 N = normalize(input.normal);
    float3 L = normalize(float3(0.42, 0.84, 0.36));
    float3 V = normalize(pc.camera_pos.xyz - input.world_pos);
    float3 H = normalize(L + V);

    // Directional sun and ambient lighting
    float  ndotl    = max(dot(N, L), 0.0);
    float  sky_hemi = 0.5 * N.y + 0.5;
    float3 ambient  = lerp(float3(0.24, 0.26, 0.34), float3(0.56, 0.64, 0.78), sky_hemi);

    // Blinn-Phong specular highlight
    float shininess     = lerp(96.0, 8.0, input.roughness);
    float spec_strength = lerp(0.45, 0.04, input.roughness);
    float spec          = pow(max(dot(N, H), 0.0), shininess) * spec_strength * ndotl;

    float3 direct_lit = (ambient + ndotl * float3(1.00, 0.96, 0.88)) * input.ao;
    float3 emissive   = base_color * (input.emission * 1.6);

    float3 lit_color = base_color * direct_lit + emissive + spec;
    return float4(lit_color, 1.0);
}
