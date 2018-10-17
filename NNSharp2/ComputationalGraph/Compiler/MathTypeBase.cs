using NNSharp2.Math;
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

        public string VariableName { get; private set; }

        public NodeOperationType Operation { get; internal set; }

        #region Constructors
        public MathTypeBase(int w, int h, string varName = null)
        {
            Operands = new List<MathTypeBase>();
            OutputNodes = new List<MathTypeBase>();
            Dimensions = new int[] { w, h };
            VariableName = varName;
        }

        public MathTypeBase(MathTypeBase b)
        {
            Dimensions = new int[] { b.Dimensions[0], b.Dimensions[1] };
            OpValues = new double[b.OpValues.Length];
            Array.Copy(b.OpValues, 0, OpValues, 0, OpValues.Length);

            Operands = new List<MathTypeBase>(b.Operands);
            OutputNodes = new List<MathTypeBase>(b.OutputNodes);
            VariableName = b.VariableName;
            Operation = b.Operation;
        }

        public MathTypeBase(int h, string varName = null) : this(1, h, varName) { }

        public MathTypeBase(int w, int h, double v) : this(w, h)
        {
            OpValues = new double[] { v };
        }

        public MathTypeBase(int h, double v) : this(h)
        {
            OpValues = new double[] { v };
        }

        protected MathTypeBase(int w, int h, NodeOperationType op, params MathTypeBase[] param) : this(w, h)
        {
            Operation = op;
            Operands.AddRange(param);

            for (int i = 0; i < param.Length; i++)
            {
                param[i].OutputNodes.Add(this);
            }
        }
        #endregion

        #region Clone
        private static MathTypeBase Clone(MathTypeBase v)
        {
            MathTypeBase v_clone = null;
            if (v is Matrix)
                v_clone = new Matrix(null, v.Dimensions[0], v.Dimensions[1]);
            else if (v is Vector)
                v_clone = new Vector(null, v.Dimensions[0], v.Dimensions[1]);
            else if (v is Constant)
                v_clone = new Constant(v.OpValues[0]);

            v_clone.OpValues = new double[v.OpValues.Length];
            Array.Copy(v.OpValues, 0, v_clone.OpValues, 0, v.OpValues.Length);
            v_clone.VariableName = v.VariableName;
            v_clone.Operation = v.Operation;

            for(int i = 0; i < v.Operands.Count; i++)
            {
                var child_clone = Clone(v.Operands[i]);
                v_clone.Operands.Add(child_clone);
                child_clone.OutputNodes.Add(v_clone);
            }

            return v_clone;
        }

        public MathTypeBase Clone()
        {
            return Clone(this);
        }
        #endregion

        #region Gradient Operators
        protected abstract MathTypeBase CurGradient(MathTypeBase wrt);
        public (bool, MathTypeBase) Gradient(MathTypeBase wrt)
        {
            if (wrt == this)
            {
                if (this is Constant)
                    return (true, new Constant(1));

                if (this is Vector)
                    return (true, Matrix.Diagonal(new Vector(Dimensions[0], Dimensions[1], 1)));

                if (this is Matrix)
                    return (true, new Matrix(Dimensions[0], Dimensions[1], 1));
            }

            if (Operation == NodeOperationType.ConstantDeclaration)
                return (false, new Constant(0));

            if (Operation == NodeOperationType.ConstantVectorDeclaration)
                return (false, Matrix.Diagonal(new Vector(Dimensions[0], Dimensions[1], 0.0d)));

            if (Operation == NodeOperationType.ConstantMatrixDeclaration)
                return (false, new Matrix(Dimensions[0], Dimensions[1], 0.0d));

            if (Operation == NodeOperationType.VectorDeclaration)
                return (false, Matrix.Diagonal(new Vector(Dimensions[0], Dimensions[1], 0.0d)));

            if (Operation == NodeOperationType.MatrixDeclaration)
                return (false, new Matrix(Dimensions[0], Dimensions[1], 0.0d));

            //Compute derivative of current operation wrt first operand * derivative of first operand + derivative of current operation wrt second operand * derivative of second operand
            var dv0 = this.CurGradient(Operands[0]);
            bool dv0_valid = false;
            MathTypeBase dv0_grad = null;

            if (Operands[0] != wrt)
            {
                (dv0_valid, dv0_grad) = Operands[0].Gradient(wrt);

                if (dv0_valid)
                    dv0 = dv0.Multiply(dv0_grad);
            }
            else
                dv0_valid = true;

            if (Operands.Count == 2)
            {
                var dv1 = this.CurGradient(Operands[1]);
                bool dv1_valid = false;
                MathTypeBase dv1_grad = null;

                if (Operands[1] != wrt)
                {
                    (dv1_valid, dv1_grad) = Operands[1].Gradient(wrt);

                    if (dv1_valid)
                        dv1 = dv1.Multiply(dv1_grad);

                    if (dv0_valid && dv1_valid)
                        return (true, dv0.Add(dv1));
                    else if (!dv0_valid && dv1_valid)
                        return (true, dv1);
                }
                else
                {
                    if (dv0_valid)
                        return (true, dv0.Add(dv1));
                    else
                        return (true, dv1);
                }
            }

            if (dv0_valid)
                return (true, dv0);

            return (false, null);
        }
        #endregion

        #region Operators
        public abstract MathTypeBase Multiply(MathTypeBase b);
        public abstract MathTypeBase Add(MathTypeBase b);
        #endregion

        #region Comparison Operators
        public static bool operator ==(MathTypeBase x, MathTypeBase y)
        {
            if (x.Operation != y.Operation)
                return false;

            if (x.VariableName != y.VariableName)
                return false;

            if (x.OpValues != null && y.OpValues != null)
            {
                if (x.OpValues.Length != y.OpValues.Length)
                    return false;

                for (int i = 0; i < x.OpValues.Length; i++)
                    if (x.OpValues[i] != y.OpValues[i])
                        return false;
            }

            if (x.OpValues == null && y.OpValues != null)
                return false;

            if (x.OpValues != null && y.OpValues == null)
                return false;

            if (x.Operands.Count != y.Operands.Count)
                return false;

            for (int i = 0; i < x.Operands.Count; i++)
            {
                if (x.Operands[i] != y.Operands[i])
                    return false;
            }

            return true;
        }

        public static bool operator !=(MathTypeBase x, MathTypeBase y)
        {
            return !(x == y);
        }
        #endregion

        #region ToString
        public override string ToString()
        {
            switch (Operation)
            {
                case NodeOperationType.Gradient:
                    return $"grad({Operands[0]})";
                case NodeOperationType.MatrixDeclaration:
                    return VariableName;
                case NodeOperationType.DiagonalMatrixDeclaration:
                    return $"diag({Operands[0]})";
                case NodeOperationType.VectorDeclaration:
                    return VariableName;
                case NodeOperationType.ConstantDeclaration:
                    return OpValues[0].ToString();
                case NodeOperationType.ConstantVectorDeclaration:
                    return $"vec({OpValues[0]})";
                case NodeOperationType.ConstantMatrixDeclaration:
                    return $"mat({OpValues[0]})";
                case NodeOperationType.Assignment:
                    return $"{VariableName} = {Operands[0]}";
                case NodeOperationType.Transpose:
                    return $"transpose({Operands[0]})";
                case NodeOperationType.MatrixProduct:
                    return $"mat_prod({Operands[0]}, {Operands[1]})";
                case NodeOperationType.TensorProduct:
                    return $"tensor_prod({Operands[0]}, {Operands[1]})";
                case NodeOperationType.HadamardProduct:
                    return $"hadamard_prod({Operands[0]}, {Operands[1]})";
                case NodeOperationType.Add:
                    return $"{Operands[0]} + {Operands[1]}";
                case NodeOperationType.Subtract:
                    return $"{Operands[0]} - {Operands[1]}";
                case NodeOperationType.Multiply:
                    return $"{Operands[0]} * {Operands[1]}";
                case NodeOperationType.Divide:
                    return $"{Operands[0]} / {Operands[1]}";
                case NodeOperationType.Tanh:
                    return $"tanh({Operands[0]})";
                case NodeOperationType.Power:
                    return $"({Operands[0]}) ^ {Operands[1]}";
                default:
                    throw new Exception();
                    break;
            }

            return base.ToString();
        }
        #endregion
    }
}
