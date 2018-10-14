using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp2.ComputationalGraph.Compiler
{
    public abstract class MathTypeBase
    {
        public int[] Dimensions { get; private set; }
        public double[] OpValues { get; private set; }

        public List<MathTypeBase> Operands { get; private set; }
        public List<MathTypeBase> OutputNodes { get; private set; }

        public NodeOperationType Operation { get; internal set; }

        #region Constructors
        public MathTypeBase(int w, int h)
        {
            Operands = new List<MathTypeBase>();
            OutputNodes = new List<MathTypeBase>();
            Dimensions = new int[] { w, h };
        }

        public MathTypeBase(MathTypeBase b)
        {
            Dimensions = new int[] { b.Dimensions[0], b.Dimensions[1] };
            OpValues = new double[b.OpValues.Length];
            Array.Copy(b.OpValues, 0, OpValues, 0, OpValues.Length);

            Operands = new List<MathTypeBase>(b.Operands);
            OutputNodes = new List<MathTypeBase>(b.OutputNodes);

            Operation = b.Operation;
        }

        public MathTypeBase(int h) : this(1, h) { }

        public MathTypeBase(int w, int h, double v) : this(w, h)
        {
            OpValues = new double[] { v };
        }

        public MathTypeBase(int h, double v) : this(h)
        {
            OpValues = new double[] { v };
        }
        #endregion


    }
}
