using NNSharp3.Math;
using OpenGL;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp3
{
    public class Device
    {
        private DeviceContext deviceContext;
        private IntPtr ctx;

        private static Device device;
        public static Device GetDevice()
        {
            if (device == null)
                device = new Device();

            return device;
        }

        Device()
        {
            deviceContext = DeviceContext.Create();
            ctx = deviceContext.CreateContext(IntPtr.Zero);

            deviceContext.MakeCurrent(ctx);
        }

        public void GLInfo()
        {
            Gl.Get(Gl.MAX_COMPUTE_WORK_GROUP_INVOCATIONS, out int compute_workgrp_invocations);

            Gl.Get(Gl.MAX_TEXTURE_BUFFER_SIZE, out int tex_buf_sz);

            Gl.Get(Gl.MAX_SHADER_STORAGE_BLOCK_SIZE, out int max_block_sz);

            Gl.Get(Gl.MAX_COMPUTE_WORK_GROUP_COUNT, 0, out int workgrp_cnt_x);
            Gl.Get(Gl.MAX_COMPUTE_WORK_GROUP_COUNT, 1, out int workgrp_cnt_y);
            Gl.Get(Gl.MAX_COMPUTE_WORK_GROUP_COUNT, 2, out int workgrp_cnt_z);

            Gl.Get(Gl.MAX_TEXTURE_SIZE, out int tex_size);

            string gl_info =
$@"Renderer: {Gl.CurrentRenderer}
Version: {Gl.CurrentVersion}
Compute Info:
    Max Compute Work Group Invocations: {compute_workgrp_invocations}
    Max Dimensions: {workgrp_cnt_x}, {workgrp_cnt_y}, {workgrp_cnt_z}
    Max Texture Size: {tex_size}

Memory Limits:
    Max Shader Buffer Size: {max_block_sz / (1024.0f * 1024.0f)} MiB
    Max Texture Buffer Size: {tex_buf_sz / (1024.0f * 1024.0f)} MiB
";

            Console.WriteLine(gl_info);
        }
    }
}
