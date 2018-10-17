using NNSharp2.ComputationalGraph.Compiler;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp2.Math
{
    public class Constant : MathTypeBase
    {
        public Constant(double val) : base(1, 1, val)
        {
            Operation = ComputationalGraph.NodeOperationType.ConstantDeclaration;
        }

        public override MathTypeBase Add(MathTypeBase b)
        {
            throw new NotImplementedException();
        }

        public override MathTypeBase Multiply(MathTypeBase b)
        {
            throw new NotImplementedException();
        }

        protected override MathTypeBase CurGradient(MathTypeBase wrt)
        {
            throw new NotImplementedException();
        }
    }
}
