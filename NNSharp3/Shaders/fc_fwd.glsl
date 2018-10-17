layout(local_size_x = X, local_size_y = Y, local_size_z = Z) in;

layout(IMG_FMT, bindless_image) uniform readonly image2D w;
layout(IMG_FMT, bindless_image) uniform readonly image2D b;

layout(IMG_FMT, bindless_image) uniform readonly image2D i; 

layout(bindless_image) uniform writeonly image2D o;
layout(bindless_image) uniform writeonly image2D a;

//O_SZ
//I_SZ

//Read the vector into shared memory

//columns = x = I_SZ
//rows = y = O_SZ

shared FLOAT4_T sum_var;
shared FLOAT4_T activ_var;

void main(){

    FLOAT_T sum = 0;
    for(int idx = 0; idx < I_SZ / 4; idx ++)
        sum += dot(imageLoad(w, ivec2(idx, gl_GlobalInvocationID.x)), imageLoad(i, ivec2(idx, 0)));

    sum += imageLoad(b, ivec2(gl_WorkGroupID.x, 0))[gl_LocalInvocationID.x];

    sum_var[gl_LocalInvocationID.x] = sum;

    switch(ACTIV_FN_IDX){
        case 0:
            activ_var[gl_LocalInvocationID.x] = 1.0f / (1.0f + exp(sum));
            break;
        case 1:
            activ_var[gl_LocalInvocationID.x] = tanh(sum);
            break;
        case 2:
            activ_var[gl_LocalInvocationID.x] = step(0.0hf, sum);
            break;
    }

    memoryBarrierShared();

    imageStore(o, ivec2(gl_WorkGroupID.x, 0), sum_var);
    imageStore(a, ivec2(gl_WorkGroupID.x, 0), activ_var);


    //o_tmp = w * i + b
    //o = o_tmp
    //a = f(ACTIV_FN_IDX, o_tmp)
}