layout(local_size_x = X, local_size_y = Y, local_size_z = Z) in;

layout(IMG_FMT, bindless_image) uniform restrict readonly imageBuffer w;
layout(IMG_FMT, bindless_image) uniform restrict readonly imageBuffer i; 
layout(IMG_FMT, bindless_image) uniform restrict readonly imageBuffer b;


layout(bindless_image) uniform restrict writeonly imageBuffer o;
layout(bindless_image) uniform restrict writeonly imageBuffer a;

//O_SZ
//I_SZ

//Read the vector into shared memory

//columns = x = I_SZ
//rows = y = O_SZ

void main(){

    FLOAT_T sum = 0;
    for(int idx = 0; idx < I_SZ; idx ++)
        sum += imageLoad(w, int(F(gl_GlobalInvocationID.x, idx))).r * imageLoad(i, int(idx)).r;

    sum += imageLoad(b, int(gl_GlobalInvocationID.x)).r;

    FLOAT_T activ_var;
    switch(ACTIV_FN_IDX){
        case 0:
            activ_var = 1.0f / (1.0f + 1.0f/exp(sum));
            break;
        case 1:
            activ_var = tanh(sum);
            break;
        case 2:
            activ_var = step(0.0hf, sum) * sum;
            break;
    }

    imageStore(o, int(gl_GlobalInvocationID.x), vec4(sum));
    imageStore(a, int(gl_GlobalInvocationID.x), vec4(activ_var));


    //o_tmp = w * i + b
    //o = o_tmp
    //a = f(ACTIV_FN_IDX, o_tmp)
}