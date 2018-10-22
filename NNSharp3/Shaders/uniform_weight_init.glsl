layout(local_size_x = X, local_size_y = Y, local_size_z = Z) in;

layout(bindless_image) uniform restrict writeonly imageBuffer w;
layout(bindless_image) uniform restrict writeonly imageBuffer b; 

uniform float seed0;
uniform float seed1;

float random (vec2 st) {
    return fract(sin(dot(st.xy, vec2(12.9898,78.233))) * 43758.5453123);
}

float random_gauss(vec2 st){
    return MEAN + SIGMA * sqrt(-2 * log(random(st))) * sin(2 * 3.141569f * random(st.yx));
}

void main(){
    vec2 seed = vec2(seed0 / (gl_GlobalInvocationID.x + 1), seed1 / (gl_GlobalInvocationID.y + 1));

    imageStore(w, int(F(gl_GlobalInvocationID.x, gl_GlobalInvocationID.y)), vec4(random_gauss(seed)));
    imageStore(b, int(gl_GlobalInvocationID.x), vec4(BIAS));
}