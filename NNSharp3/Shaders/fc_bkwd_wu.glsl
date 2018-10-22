layout(local_size_x = X, local_size_y = Y, local_size_z = Z) in;


layout(IMG_FMT, bindless_image) uniform restrict readonly imageBuffer err;
layout(IMG_FMT, bindless_image) uniform restrict readonly imageBuffer i; //previous layer activation

layout(IMG_FMT, bindless_image) uniform restrict imageBuffer w;
layout(IMG_FMT, bindless_image) uniform restrict imageBuffer b;

uniform float learning_rate;


//columns = x = I_SZ
//rows = y = O_SZ

void main(){

    FLOAT_T b_i = imageLoad(b, int(gl_GlobalInvocationID.x)).r; 
    FLOAT_T w_i = imageLoad(w, int(F(gl_GlobalInvocationID.x, gl_GlobalInvocationID.y))).r; 
    FLOAT_T err_i = imageLoad(err, int(gl_GlobalInvocationID.x)).r; 
    
    //Update the bias
    imageStore(b, int(gl_GlobalInvocationID.x), vec4(b_i - learning_rate * err_i));


    //Update the weights
    for(int idx = 0; idx < I_SZ; idx++){
        FLOAT_T a0_i = imageLoad(i, int(idx)).r;
        FLOAT_T w_i = imageLoad(w, int(F(gl_GlobalInvocationID.x, idx))).r;
        FLOAT_T w0_i_err = a0_i * err_i;
        imageStore(w, int(F(gl_GlobalInvocationID.x, idx)), vec4(w_i - learning_rate * w0_i_err));
    }
}