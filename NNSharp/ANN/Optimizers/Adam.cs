using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp.ANN.Optimizers
{
    [Serializable]
    public class Adam : IOptimizer
    {
        private float r;
        private NeuralNetwork net;

        public Adam(NeuralNetwork net)
        {
            this.net = net;
        }

        public void Optimize(Matrix w, Matrix nabla_w)
        {
            Matrix.MSubSelf(nabla_w, w, r);
        }

        public void Optimize(Vector b, Vector nabla_b)
        {
            Vector.MSubSelf(nabla_b, b, r);
        }

        public void SetLearningRate(float v)
        {
            r = v;
        }

        public void Update(float curError)
        {

        }
    }
}
