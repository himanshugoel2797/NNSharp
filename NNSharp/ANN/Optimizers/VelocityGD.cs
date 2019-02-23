using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp.ANN.Optimizers
{
    [Serializable]
    public class VelocityGD : IOptimizer
    {
        private float rate, rate_margin;
        private float prevError = float.NaN;
        private float n_rate = 0.00001f;

        public VelocityGD(float rate_margin)
        {
            this.rate_margin = rate_margin;
        }

        public void Update(float curError)
        {
            if (prevError != float.NaN)
            {
                if (n_rate == 0 | n_rate > 0.1f) n_rate = rate;

                //Compute the error velocity and update the learning rate
                float curVelocity = (prevError - curError) / prevError;
                if (curVelocity < 0.005f && curVelocity > 0.0f)
                {
                    n_rate = 2f * n_rate;
                    Console.WriteLine("Rate Updated: " + n_rate);
                }
                else if (curVelocity < -0.01f)
                {
                    n_rate = n_rate / 2f;
                    Console.WriteLine("Rate Updated: " + n_rate);
                }
            }
            prevError = curError;
        }

        public void Optimize(Matrix w, Matrix nabla_w)
        {
            Matrix.MSubSelf(nabla_w, w, n_rate);
        }

        public void Optimize(Vector b, Vector nabla_b)
        {
            Vector.MSubSelf(nabla_b, b, n_rate);
        }

        public void SetLearningRate(float v)
        {
            rate = v;
            n_rate = v;
        }
    }
}
