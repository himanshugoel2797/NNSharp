layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout(bindless_image) uniform restrict writeonly imageBuffer w;

uniform float val;
void main(){
    imageStore(w, int(gl_LocalInvocationIndex), vec4(val));
}