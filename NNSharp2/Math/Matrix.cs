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
    public class Matrix : MathTypeBase
    {
        public int Width { get { return Dimensions[0]; } }
        public int Height { get { return Dimensions[1]; } }

        public Matrix(string varName, int w, int h) : base(w, h, varName)
        {
            Operation = NodeOperationType.MatrixDeclaration;
        }

        public Matrix(int w, int h, double constVal) : base(w, h, constVal)
        {
            Operation = NodeOperationType.ConstantMatrixDeclaration;
        }

        public static Matrix Diagonal(Vector vec)
        {
            int dim = SMath.Max(vec.Width, vec.Height);
            return new Matrix(dim, dim, NodeOperationType.DiagonalMatrixDeclaration, vec);
        }

        internal Matrix(int w, int h, NodeOperationType op, params MathTypeBase[] param) : base(w, h, op, param) { }

        #region Matrix Multiplication Operators
        public static Matrix operator *(Matrix a, Matrix b)
        {
            if (a.Width != b.Height)
                throw new ArgumentException();

            if (a.Operation == NodeOperationType.DiagonalMatrixDeclaration && b.Operation == NodeOperationType.DiagonalMatrixDeclaration)
                return new Matrix(b.Width, a.Height, NodeOperationType.DiagonalMatrixDeclaration, Vector.Hadamard((a.Operands[0] as Vector), (b.Operands[0] as Vector)));

            if (a.Operation == NodeOperationType.DiagonalMatrixDeclaration && a.Operands[0].Operation == NodeOperationType.ConstantVectorDeclaration && a.Operands[0].OpValues[0] == 1)
                return b;

            if (b.Operation == NodeOperationType.DiagonalMatrixDeclaration && b.Operands[0].Operation == NodeOperationType.ConstantVectorDeclaration && b.Operands[0].OpValues[0] == 1)
                return a;

            return new Matrix(b.Width, a.Height, NodeOperationType.MatrixProduct, a, b);
        }

        public static Vector operator *(Matrix a, Vector b)
        {
            if (a.Width != b.Height)
                throw new ArgumentException();

            return new Vector(b.Width, a.Height, NodeOperationType.MatrixProduct, a, b);
        }
        #endregion

        #region Elementwise Operators
        public static Matrix Hadamard(Matrix a, Matrix b)
        {
            if (a.Width != b.Width)
                throw new ArgumentException();

            if (a.Height != b.Height)
                throw new ArgumentException();

            return new Matrix(a.Width, a.Height, NodeOperationType.HadamardProduct, a, b);
        }

        public static Matrix operator +(Matrix a, Matrix b)
        {
            if (a.Width != b.Width)
                throw new ArgumentException();

            if (a.Height != b.Height)
                throw new ArgumentException();

            return new Matrix(a.Width, a.Height, NodeOperationType.Add, a, b);
        }

        public static Matrix operator -(Matrix a, Matrix b)
        {
            if (a.Width != b.Width)
                throw new ArgumentException();

            if (a.Height != b.Height)
                throw new ArgumentException();

            return new Matrix(a.Width, a.Height, NodeOperationType.Subtract, a, b);
        }

        public static Matrix Tanh(Matrix a)
        {
            return new Matrix(a.Width, a.Height, NodeOperationType.Tanh, a);
        }

        public static Matrix Power(Matrix a, double pwr)
        {
            return new Matrix(a.Width, a.Height, NodeOperationType.Power, a, new Constant(pwr));
        }
        #endregion

        #region Constant Multiply
        public static Matrix operator *(double b, Matrix a)
        {
            return a * b;
        }

        public static Matrix operator *(Matrix a, double b)
        {
            return Matrix.Hadamard(a, new Matrix(a.Width, a.Height, b));
            //return new Matrix(NodeOperationType.Multiply, a.node, new Node(NodeOperationType.ConstantDeclaration, b));
        }
        #endregion

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
                    break;
                case NodeOperationType.TensorProduct:
                    break;
                case NodeOperationType.HadamardProduct:
                    break;
                case NodeOperationType.Add:
                    break;
                case NodeOperationType.Subtract:
                    break;
                case NodeOperationType.Multiply:
                    break;
                case NodeOperationType.Divide:
                    break;
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
            if (b is Matrix)
                return this * (b as Matrix);

            if (b is Vector)
                return this * (b as Vector);

            throw new Exception();
        }

        public override MathTypeBase Add(MathTypeBase b)
        {
            if (b is Matrix)
                return this + (b as Matrix);

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

        public Matrix Transpose()
        {
            return new Matrix(Height, Width, NodeOperationType.Transpose, this);
        }

    }
}
