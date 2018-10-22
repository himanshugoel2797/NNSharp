layout(local_size_x = X, local_size_y = Y, local_size_z = Z) in;

layout(IMG_FMT, bindless_image) uniform restrict readonly imageBuffer w;
layout(IMG_FMT, bindless_image) uniform restrict readonly imageBuffer b;


layout(IMG_FMT, bindless_image) uniform restrict readonly imageBuffer o;
layout(IMG_FMT, bindless_image) uniform restrict readonly imageBuffer a;

layout(IMG_FMT, bindless_image) uniform restrict readonly imageBuffer i; //previous layer activation

//expected output
layout(IMG_FMT, bindless_image) uniform restrict readonly imageBuffer eo;

uniform float learning_rate;

layout(bindless_image) uniform restrict writeonly imageBuffer errO;

//columns = x = I_SZ
//rows = y = O_SZ

void main(){

    FLOAT_T o_i = imageLoad(o, int(gl_GlobalInvocationID.x)).r;
    FLOAT_T a_i = imageLoad(a, int(gl_GlobalInvocationID.x)).r;
    FLOAT_T eo_i = imageLoad(eo, int(gl_GlobalInvocationID.x)).r; 
    FLOAT_T b_i = imageLoad(b, int(gl_GlobalInvocationID.x)).r;

    //Compute loss function derivative
    FLOAT_T loss_deriv;
    switch(LOSS_FN_IDX){
        case 0:
            loss_deriv = (a_i - eo_i);
            break;
    }
    
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
    
    FLOAT_T err = activ_deriv * loss_deriv;

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