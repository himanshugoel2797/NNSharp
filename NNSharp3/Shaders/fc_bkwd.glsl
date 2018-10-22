layout(local_size_x = X, local_size_y = Y, local_size_z = Z) in;

layout(IMG_FMT, bindless_image) uniform restrict readonly imageBuffer w;
layout(IMG_FMT, bindless_image) uniform restrict readonly imageBuffer b;

layout(IMG_FMT, bindless_image) uniform restrict readonly imageBuffer o;
layout(IMG_FMT, bindless_image) uniform restrict readonly imageBuffer a;

layout(IMG_FMT, bindless_image) uniform restrict readonly imageBuffer eo;//previous layer error
layout(IMG_FMT, bindless_image) uniform restrict readonly imageBuffer i; //previous layer activation

//expected output
layout(IMG_FMT, bindless_image) uniform restrict imageBuffer errO;

//columns = x = I_SZ
//rows = y = O_SZ

void main(){

    FLOAT_T o_i = imageLoad(o, int(gl_GlobalInvocationID.x)).r;
    FLOAT_T a_i = imageLoad(a, int(gl_GlobalInvocationID.x)).r;
    FLOAT_T eo_i = imageLoad(eo, int(gl_GlobalInvocationID.x)).r; 
    FLOAT_T b_i = imageLoad(b, int(gl_GlobalInvocationID.x)).r;

    //Compute transpose(w) * err
    //column(w) dot err
    FLOAT_T trans_w_err = imageLoad(errO, int(gl_GlobalInvocationID.x)).r;

    //Compute activation function derivative
    FLOAT_T activ_deriv;
    switch(ACTIV_FN_IDX){
        case 0:
            activ_deriv = a_i * (1.0f - a_i);
            break;
        case 1:
            activ_deriv = 1.0f - (a_i * a_i);
            break;
        case 2:
            activ_deriv = step(0.0hf, a_i);
            break;
    }
    
    FLOAT_T err = trans_w_err * activ_deriv;

    //Update the error
    imageStore(errO, int(gl_GlobalInvocationID.x), vec4(err));

    //Update the weights
    /*for(int i = 0; i < I_SZ / 4; i++){
        FLOAT4_T a0_i = imageLoad(i, ivec2(i, 0));

        for(int j = 0; j < 4; j++){
            FLOAT4_T w_i = imageLoad(w, ivec2(gl_GlobalInvocationID.x, i * 4 + j));
            FLOAT4_T w0_i_err = a0_i[j] * err;
            imageStore(wO, ivec2(gl_GlobalInvocationID.x, i * 4 + j), w_i - learning_rate * w0_i_err);
        }
    }*/
}