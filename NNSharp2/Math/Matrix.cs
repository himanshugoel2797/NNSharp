using NNSharp2.ComputationalGraph;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp2.Math
{
    public class Matrix
    {
        internal Node node { get; private set; }

        public int Width { get; private set; }
        public int Height { get; private set; }

        public Matrix(string varName, int w, int h) : this(varName, w, h, NodeOperationType.MatrixDeclaration) { }

        public Matrix(int w, int h, double constVal)
        {
            Width = w;
            Height = h;

            node = new Node(NodeOperationType.ConstantMatrixDeclaration, constVal)
            {
                Dimension = new int[] { w, h }
            };
        }

        internal Matrix(int w, int h, NodeOperationType operationType, params Node[] inputs)
        {
            Width = w;
            Height = h;

            node = new Node(operationType)
            {
                Dimension = new int[] { w, h }
            };

            node.IncomingEdges.AddRange(inputs);
            for (int i = 0; i < inputs.Length; i++)
                inputs[i].OutgoingEdges.Add(node);
        }

        internal Matrix(string varName, int w, int h, NodeOperationType operationType)
        {
            Width = w;
            Height = h;

            node = new Node(operationType, varName)
            {
                Dimension = new int[] { w, h }
            };
        }

        internal Matrix(Node n)
        {
            node = n;
            Width = n.Dimension[0];
            Height = n.Dimension[1];
        }

        public Matrix Gradient(Matrix wrt)
        {
            //Building the gradient tree here, rather than waiting for the compiler
            //Find gradients of all incoming paths
            var m = new Matrix(node.Gradient(wrt.node));
            return m;
        }

        public static Matrix operator *(Matrix a, Matrix b)
        {
            if (a.Width != b.Height)
                throw new ArgumentException();

            return new Matrix(b.Width, a.Height, NodeOperationType.Multiply, a.node, b.node);
        }

        public static Matrix operator /(Matrix b, Matrix a)
        {
            if (a.Width != b.Width)
                throw new ArgumentException();

            if (a.Height != b.Height)
                throw new ArgumentException();

            return new Matrix(a.Width, a.Height, NodeOperationType.Divide, b.node, a.node);
        }

        public static Matrix operator -(Matrix a, Matrix b)
        {
            if (a.Width != b.Width)
                throw new ArgumentException();

            if (a.Height != b.Height)
                throw new ArgumentException();

            return new Matrix(a.Width, a.Height, NodeOperationType.Subtract, a.node, b.node);
        }

        public static Matrix operator +(Matrix a, Matrix b)
        {
            if (a.Width != b.Width)
                throw new ArgumentException();

            if (a.Height != b.Height)
                throw new ArgumentException();

            return new Matrix(a.Width, a.Height, NodeOperationType.Add, a.node, b.node);
        }

        public static Matrix Hadamard(Matrix a, Matrix b)
        {
            if (a.Width != b.Width)
                throw new ArgumentException();

            if (a.Height != b.Height)
                throw new ArgumentException();

            return new Matrix(a.Width, a.Height, NodeOperationType.Hadamard, a.node, b.node);
        }

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
        }

        public Matrix Transpose()
        {
            return new Matrix(Height, Width, NodeOperationType.Transpose, node);
        }

    }
}
