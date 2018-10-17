using NNSharp2.ComputationalGraph;
using NNSharp2.ComputationalGraph.Compiler;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using SMath = System.Math;

namespace NNSharp2.Math
{
    public class Vector : MathTypeBase
    {
        public int Width { get { return Dimensions[0]; } }
        public int Height { get { return Dimensions[1]; } }

        public Vector(string varName, int w, int h) : base(w, h, varName)
        {
            Operation = NodeOperationType.VectorDeclaration;
            if (w != 1 && h != 1) throw new Exception();
        }

        public Vector(int w, int h, double constVal) : base(w, h, constVal)
        {
            Operation = NodeOperationType.ConstantVectorDeclaration;
            if (w != 1 && h != 1) throw new Exception();
        }

        internal Vector(int w, int h, NodeOperationType op, params MathTypeBase[] param) : base(w, h, op, param)
        {
            if (w != 1 && h != 1) throw new Exception();
        }

        #region Elementwise Operators
        public static Vector Hadamard(Vector a, Vector b)
        {
            if (a.Width != b.Width)
                throw new ArgumentException();

            if (a.Height != b.Height)
                throw new ArgumentException();

            if (a.Operation == NodeOperationType.ConstantVectorDeclaration && a.OpValues[0] == 0)
                return new Vector(a.Width, a.Height, 0.0d);

            if (b.Operation == NodeOperationType.ConstantVectorDeclaration && b.OpValues[0] == 0)
                return new Vector(a.Width, a.Height, 0.0d);

            if (a.Operation == NodeOperationType.ConstantVectorDeclaration && a.OpValues[0] == 1)
                return b;

            if (b.Operation == NodeOperationType.ConstantVectorDeclaration && b.OpValues[0] == 1)
                return a;

            return new Vector(a.Width, a.Height, NodeOperationType.HadamardProduct, a, b);
        }

        public static Vector operator +(Vector a, Vector b)
        {
            if (a.Width != b.Width)
                throw new ArgumentException();

            if (a.Height != b.Height)
                throw new ArgumentException();

            if (a.Operation == NodeOperationType.ConstantVectorDeclaration && a.OpValues[0] == 0)
                return b;

            if (b.Operation == NodeOperationType.ConstantVectorDeclaration && b.OpValues[0] == 0)
                return a;

            return new Vector(a.Width, a.Height, NodeOperationType.Add, a, b);
        }

        public static Vector operator -(Vector a, Vector b)
        {
            if (a.Width != b.Width)
                throw new ArgumentException();

            if (a.Height != b.Height)
                throw new ArgumentException();

            return new Vector(a.Width, a.Height, NodeOperationType.Subtract, a, b);
        }

        public static Vector Tanh(Vector a)
        {
            return new Vector(a.Width, a.Height, NodeOperationType.Tanh, a);
        }

        public static Vector Power(Vector a, double pwr)
        {
            if (pwr == 1)
                return a;

            return new Vector(a.Width, a.Height, NodeOperationType.Power, a, new Constant(pwr));
        }
        #endregion

        #region Constant Multiply
        public static Vector operator *(double b, Vector a)
        {
            return a * b;
        }

        public static Vector operator *(Vector a, double b)
        {
            return Vector.Hadamard(a, new Vector(a.Width, a.Height, b));
            //return new Matrix(NodeOperationType.Multiply, a.node, new Node(NodeOperationType.ConstantDeclaration, b));
        }
        #endregion

        public static Matrix TensorProduct(Vector a, Vector b)
        {
            return new Matrix(SMath.Max(a.Height, a.Width), SMath.Max(b.Height, b.Width), NodeOperationType.TensorProduct, a, b);
        }

        public static Matrix ReverseProduct(Matrix a)
        {//Transpose 'a', multiply with preceeding term multiply by identity vector then take diagonal
            return new Matrix(a.Width, a.Width, NodeOperationType.ReverseMatrixProduct, a);
        }

        #region Gradient Operators
        protected override MathTypeBase CurGradient(MathTypeBase wrt)
        {
            switch (Operation)
            {
                case NodeOperationType.MatrixDeclaration:

                    break;
                case NodeOperationType.VectorDeclaration:
                    break;
                case NodeOperationType.ConstantDeclaration:
                    break;
                case NodeOperationType.ConstantVectorDeclaration:
                    break;
                case NodeOperationType.ConstantMatrixDeclaration:
                    break;
                case NodeOperationType.Transpose:
                    break;
                case NodeOperationType.MatrixProduct:
                    {
                        if (wrt == Operands[0])
                        {
                            return TensorProduct((Operands[1] as Vector).Transpose(), new Vector(1, Operands[0].Dimensions[1], 1));
                        }
                        else if (wrt == Operands[1])
                        {
                            return ReverseProduct(Operands[0] as Matrix);
                        }
                    }
                    break;
                case NodeOperationType.TensorProduct:
                    {
                        //if (wrt == Operands[0])

                    }
                    break;
                case NodeOperationType.HadamardProduct:
                    if (wrt == Operands[0])
                        return Matrix.Diagonal(Operands[1] as Vector);
                    else if (wrt == Operands[1])
                        return Matrix.Diagonal(Operands[0] as Vector);
                    break;
                case NodeOperationType.Add:
                    return Matrix.Diagonal(new Vector(Width, Height, 1));
                    break;
                case NodeOperationType.Subtract:
                    if (wrt == Operands[0])
                        return Matrix.Diagonal(new Vector(Width, Height, 1));
                    else if (wrt == Operands[1])
                        return Matrix.Diagonal(new Vector(Width, Height, -1));
                    break;
                case NodeOperationType.Multiply:
                    break;
                case NodeOperationType.Divide:
                    break;
                case NodeOperationType.DiagonalMatrixDeclaration:
                    break;
                case NodeOperationType.Tanh:
                    return Matrix.Diagonal(new Vector(Width, Height, 1) + Vector.Power(this, 2));
                case NodeOperationType.Power:
                    return Matrix.Diagonal(Vector.Hadamard(new Vector(Width, Height, Operands[1].OpValues[0]), Vector.Power(Operands[0] as Vector, Operands[1].OpValues[0] - 1)));
                case NodeOperationType.Gradient:
                case NodeOperationType.Assignment:
                default:
                    throw new Exception("Unexpected operation.");
                    break;
            }
            return null;
        }
        #endregion


        #region Operators
        public override MathTypeBase Multiply(MathTypeBase b)
        {
            /*if (b is Matrix)
                return this * (b as Matrix);

            if (b is Vector)
                return this * (b as Vector);
                */
            throw new Exception();
        }

        public override MathTypeBase Add(MathTypeBase b)
        {
            if (b is Vector)
                return this + (b as Vector);

            throw new Exception();
        }
        #endregion

        /*
        #region Constant Divide
        public static Matrix operator /(Matrix a, double b)
        {
            return new Matrix(NodeOperationType.Multiply, a.node, new Node(NodeOperationType.ConstantDeclaration, 1 / b));
        }

        public static Matrix operator /(double b, Matrix a)
        {
            return new Matrix(NodeOperationType.Divide, new Node(NodeOperationType.ConstantDeclaration, b), a.node);
        }

        #endregion

        #region Constant Add
        public static Matrix operator +(Matrix a, double b)
        {
            return new Matrix(NodeOperationType.Add, a.node, new Node(NodeOperationType.ConstantDeclaration, b));
        }

        public static Matrix operator +(double b, Matrix a)
        {
            return new Matrix(NodeOperationType.Add, a.node, new Node(NodeOperationType.ConstantDeclaration, b));
        }
        #endregion

        #region Constant Subtract
        public static Matrix operator -(Matrix a, double b)
        {
            return new Matrix(NodeOperationType.Subtract, a.node, new Node(NodeOperationType.ConstantDeclaration, b));
        }

        public static Matrix operator -(double b, Matrix a)
        {
            return new Matrix(NodeOperationType.Subtract, new Node(NodeOperationType.ConstantDeclaration, b), a.node);
        }
        #endregion
        */
        /*
        public static Matrix Power(Matrix a, double pwr)
        {
            return new Matrix(a.Width, a.Height, NodeOperationType.Power, a.node, new Node(NodeOperationType.ConstantDeclaration, pwr));
        }

        public static Matrix Tanh(Matrix a)
        {
            return new Matrix(a.Width, a.Height, NodeOperationType.Tanh, a.node);
        }

        public static Matrix Log(Matrix a)
        {
            return new Matrix(a.Width, a.Height, NodeOperationType.Log, a.node);
        }

        public static Matrix Exp(Matrix a)
        {
            return new Matrix(a.Width, a.Height, NodeOperationType.Exp, a.node);
        }*/

        public Vector Transpose()
        {
            return new Vector(Height, Width, NodeOperationType.Transpose, this);
        }

    }
}
