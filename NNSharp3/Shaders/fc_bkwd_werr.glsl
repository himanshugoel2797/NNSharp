layout(local_size_x = X, local_size_y = Y, local_size_z = Z) in;

layout(IMG_FMT, bindless_image) uniform restrict readonly imageBuffer w;
layout(IMG_FMT, bindless_image) uniform restrict readonly imageBuffer eo;//previous layer error

//expected output
layout(bindless_image) uniform restrict writeonly imageBuffer errO;

//columns = y = I_SZ
//rows = x = O_SZ

void main(){

    //Compute transpose(w) * err
    //column(w) dot err
    //Update the weights
    
    FLOAT_T sum = 0;
    for(int idx = 0; idx < O_SZ; idx ++)
        sum += imageLoad(w, int(F(idx, gl_GlobalInvocationID.x))).r * imageLoad(eo, int(idx)).r;
    
    //Update the error
    imageStore(errO, int(gl_GlobalInvocationID.x), vec4(sum));


}