layout(local_size_x = 4, local_size_y = 1, local_size_z = 1) in;

layout(bindless_image) uniform writeonly image2D w;
layout(bindless_image) uniform writeonly image2D b; 

uniform float seed0;
uniform float seed1;

uniform float cols_cnt;
uniform float O_SZ;

shared vec4 rand_vals;
shared vec4 bias_vals;

float random (vec2 st) {
    return fract(sin(dot(st.xy, vec2(12.9898,78.233))) * 43758.5453123);
}

void main(){
    vec2 seed = vec2(seed0 + gl_GlobalInvocationID.x, seed1 + gl_GlobalInvocationID.y);

    if(gl_GlobalInvocationID.x < cols_cnt){
        rand_vals[gl_LocalInvocationID.x] = random(seed);
    }else{
        rand_vals[gl_LocalInvocationID.x] = 0;
    }

    if(gl_GlobalInvocationID.x < O_SZ)
        bias_vals[gl_LocalInvocationID.x] = BIAS;
    else
        bias_vals[gl_LocalInvocationID.x] = 0;
    memoryBarrierShared();

    imageStore(w, ivec2(gl_WorkGroupID.x, gl_WorkGroupID.y), rand_vals);
    imageStore(b, ivec2(gl_WorkGroupID.x, 0), bias_vals);
}