layout(local_size_x = 4, local_size_y = 1, local_size_z = 1) in;

layout(bindless_image) uniform writeonly image2D w;

uniform float val;
uniform float cols_cnt;

shared vec4 w_val;

void main(){

    if(gl_GlobalInvocationID.x < cols_cnt){
        w_val[gl_LocalInvocationID.x] = val;
    }else{
        w_val[gl_LocalInvocationID.x] = 0;
    }
    memoryBarrierShared();

    imageStore(w, ivec2(gl_WorkGroupID.x, gl_WorkGroupID.y), w_val);
}