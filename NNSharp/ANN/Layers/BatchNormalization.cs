using NNSharp.ANN.Kernels;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp.ANN.Layers
{
    [Serializable]
    public class BatchNormalization : IWeightInitializable
    {
        //Forward: Take an image as input and convolve it with the specified number of filters
        //Backward: Apply deconvolution to update the filters
        private int inputSz = 0;

        public BatchNormalization()
        {
        }

        public void Reset()
        {
        }

        public Vector Error(Vector prev_delta, bool update_cur)
        {
            return null;
        }

        public Vector Forward(Vector input)
        {
            return null;
        }

        public void Learn(IOptimizer optimizer)
        {
        }
        
        public void SetInputSize(int sz)
        {
            inputSz = sz;
        }

        public void SetWeights(IWeightInitializer weightInitializer)
        {
            
        }
    }
}
