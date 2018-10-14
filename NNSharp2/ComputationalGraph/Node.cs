using NNSharp2.Math;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp2.ComputationalGraph
{
    public class Node
    {
        public List<Node> IncomingEdges { get; private set; }
        public List<Node> OutgoingEdges { get; private set; }

        public string VariableName { get; internal set; }
        public NodeOperationType Operation { get; internal set; }
        public double[] OpSpecificValue { get; internal set; }
        public int[] Dimension { get; set; }

        public Node(NodeOperationType operationType, params double[] value)
        {
            Operation = operationType;
            OpSpecificValue = value;

            IncomingEdges = new List<Node>();
            OutgoingEdges = new List<Node>();
        }

        public Node(NodeOperationType declOp, string varName, params double[] value) : this(declOp, value)
        {
            VariableName = varName;
        }

        public Node(Node n)
        {
            Operation = n.Operation;
            IncomingEdges = new List<Node>(n.IncomingEdges);
            OutgoingEdges = new List<Node>(n.OutgoingEdges);
            VariableName = n.VariableName;
            OpSpecificValue = new double[n.OpSpecificValue.Length];
            Dimension = new int[n.Dimension.Length];

            Array.Copy(n.OpSpecificValue, OpSpecificValue, OpSpecificValue.Length);
            Array.Copy(n.Dimension, Dimension, Dimension.Length);
        }

        internal Node NodeGradient(int wrt)
        {
            switch (Operation)
            {
                case NodeOperationType.Tanh:
                    {
                        Matrix x = new Matrix(IncomingEdges[wrt]);
                        return (new Matrix(IncomingEdges[wrt].Dimension[0], IncomingEdges[wrt].Dimension[1], 1.0d) - Matrix.Power(Matrix.Tanh(x), 2)).node;
                    }
                case NodeOperationType.Log:
                    {
                        Matrix x = new Matrix(IncomingEdges[wrt]);
                        return (new Matrix(IncomingEdges[wrt].Dimension[0], IncomingEdges[wrt].Dimension[1], 1.0d) / x).node;
                    }
                case NodeOperationType.Exp:
                    {
                        Matrix x = new Matrix(IncomingEdges[0]);
                        return Matrix.Exp(x).node;
                    }
                case NodeOperationType.Add:
                    {
                        return new Matrix(IncomingEdges[wrt].Dimension[0], IncomingEdges[wrt].Dimension[1], 1).node;
                    }
                case NodeOperationType.Hadamard:
                    {
                        Matrix u = new Matrix(IncomingEdges[0]);
                        Matrix v = new Matrix(IncomingEdges[1]);


                        //z = u . v
                        //dz = u . dv + du . v
                        //dz / du = v
                        //dz / dv = u

                        if (wrt == 0)
                            return IncomingEdges[1];
                        else if (wrt == 1)
                            return IncomingEdges[0];

                        throw new Exception();
                        //return (Matrix.Hadamard(u, dv) + Matrix.Hadamard(du, v)).node;
                    }
                case NodeOperationType.Multiply:
                    {
                        if (wrt == 0)
                            return IncomingEdges[1];
                        else if (wrt == 1)
                            return IncomingEdges[0];

                        throw new Exception();
                    }
                case NodeOperationType.Divide:
                    {
                        throw new Exception();
                        //Matrix u = new Matrix(IncomingEdges[0]);
                        //Matrix v = new Matrix(IncomingEdges[1]);

                        //Matrix du = new Matrix(subNodes[0]);
                        //Matrix dv = new Matrix(subNodes[1]);

                        //return ((v * du - u * dv) / Matrix.Power(v, 2)).node;
                    }
                case NodeOperationType.Subtract:
                    {
                        if (wrt == 0)
                            return new Matrix(IncomingEdges[wrt].Dimension[0], IncomingEdges[wrt].Dimension[1], 1).node;
                        else if (wrt == 0)
                            return new Matrix(IncomingEdges[wrt].Dimension[0], IncomingEdges[wrt].Dimension[1], -1).node;

                        throw new Exception();
                    }
                case NodeOperationType.Power:
                    {
                        //Matrix db = new Matrix(subNodes[1]);

                        Matrix a = new Matrix(IncomingEdges[wrt]);
                        //Matrix b = new Matrix(IncomingEdges[1]);

                        return (IncomingEdges[1].OpSpecificValue[0] * Matrix.Power(a, IncomingEdges[1].OpSpecificValue[0] - 1)).node;
                    }
                default:
                    throw new Exception();
            }
        }

        internal Node Gradient(Node wrt)
        {
            if (wrt == this)
            {
                //return a constant value '1'
                return new Node(NodeOperationType.ConstantMatrixDeclaration, 1)
                {
                    Dimension = Dimension
                };
            }

            if (this.Operation == NodeOperationType.ConstantMatrixDeclaration | this.Operation == NodeOperationType.MatrixDeclaration)
            {
                return new Node(NodeOperationType.ConstantMatrixDeclaration, 0)
                {
                    Dimension = Dimension
                };
            }

            if (this.Operation == NodeOperationType.ConstantDeclaration)
            {
                return new Node(NodeOperationType.ConstantDeclaration, 0);
            }

            //List<Node> subNodes = new List<Node>();
            Matrix sum = new Matrix(NodeGradient(0)) * new Matrix(IncomingEdges[0].Gradient(wrt));

            for (int i = 1; i < IncomingEdges.Count; i++)
            {
                sum += new Matrix(NodeGradient(i)) * new Matrix(IncomingEdges[i].Gradient(wrt));
            }

            return sum.node;

            /*
            switch (Operation)
            {
                case NodeOperationType.Tanh:
                    {
                        Matrix x = new Matrix(IncomingEdges[0]);
                        Matrix dx = new Matrix(subNodes[0]);
                        return Matrix.Hadamard((new Matrix(IncomingEdges[0].Dimension[0], IncomingEdges[0].Dimension[1], 1.0d) - Matrix.Power(Matrix.Tanh(x), 2)), dx).node;
                    }
                case NodeOperationType.Log:
                    {
                        Matrix x = new Matrix(IncomingEdges[0]);
                        Matrix dx = new Matrix(subNodes[0]);
                        return Matrix.Hadamard((new Matrix(IncomingEdges[0].Dimension[0], IncomingEdges[0].Dimension[1], 1.0d) / x), dx).node;
                    }
                case NodeOperationType.Exp:
                    {
                        Matrix x = new Matrix(IncomingEdges[0]);
                        Matrix dx = new Matrix(subNodes[0]);
                        return Matrix.Hadamard(Matrix.Exp(x), dx).node;
                    }
                case NodeOperationType.Add:
                    {
                        Matrix da = new Matrix(subNodes[0]);
                        Matrix db = new Matrix(subNodes[1]);

                        return (da + db).node;
                    }
                case NodeOperationType.Hadamard:
                    {
                        Matrix u = new Matrix(IncomingEdges[0]);
                        Matrix v = new Matrix(IncomingEdges[1]);

                        Matrix du = new Matrix(subNodes[0]);
                        Matrix dv = new Matrix(subNodes[1]);

                        return (Matrix.Hadamard(u, dv) + Matrix.Hadamard(du, v)).node;
                    }
                case NodeOperationType.Multiply:
                    {
                        Matrix u = new Matrix(IncomingEdges[0]);
                        Matrix v = new Matrix(IncomingEdges[1]);

                        Matrix du = new Matrix(subNodes[0]);
                        Matrix dv = new Matrix(subNodes[1]);

                        return (du * v + u * dv).node;
                    }
                case NodeOperationType.Divide:
                    {
                        throw new Exception();
                        Matrix u = new Matrix(IncomingEdges[0]);
                        Matrix v = new Matrix(IncomingEdges[1]);

                        Matrix du = new Matrix(subNodes[0]);
                        Matrix dv = new Matrix(subNodes[1]);

                        return ((v * du - u * dv) / Matrix.Power(v, 2)).node;
                    }
                case NodeOperationType.Subtract:
                    {
                        Matrix da = new Matrix(subNodes[0]);
                        Matrix db = new Matrix(subNodes[1]);

                        return (da - db).node;
                    }
                case NodeOperationType.Power:
                    {
                        Matrix da = new Matrix(subNodes[0]);
                        //Matrix db = new Matrix(subNodes[1]);

                        Matrix a = new Matrix(IncomingEdges[0]);
                        //Matrix b = new Matrix(IncomingEdges[1]);

                        return Matrix.Hadamard(IncomingEdges[1].OpSpecificValue[0] * Matrix.Power(a, IncomingEdges[1].OpSpecificValue[0] - 1), da).node;
                    }
            }
            return null;
            */
        }

        public static bool operator ==(Node x, Node y)
        {
            if (x.Operation != y.Operation)
                return false;

            if (x.VariableName != y.VariableName)
                return false;

            if (x.OpSpecificValue != null && y.OpSpecificValue != null)
            {
                if (x.OpSpecificValue.Length != y.OpSpecificValue.Length)
                    return false;

                for (int i = 0; i < x.OpSpecificValue.Length; i++)
                    if (x.OpSpecificValue[i] != y.OpSpecificValue[i])
                        return false;
            }

            if (x.OpSpecificValue == null && y.OpSpecificValue != null)
                return false;

            if (x.OpSpecificValue != null && y.OpSpecificValue == null)
                return false;

            if (x.IncomingEdges.Count != y.IncomingEdges.Count)
                return false;

            for (int i = 0; i < x.IncomingEdges.Count; i++)
            {
                if (x.IncomingEdges[i] != y.IncomingEdges[i])
                    return false;
            }

            return true;
        }

        public static bool operator !=(Node x, Node y)
        {
            return !(x == y);
        }

        public override string ToString()
        {
            if (IncomingEdges.Count == 0)
            {
                switch (Operation)
                {
                    case NodeOperationType.MatrixDeclaration:
                    case NodeOperationType.IntermediateDeclaration:
                        return VariableName;
                    case NodeOperationType.ConstantDeclaration:
                        return OpSpecificValue[0].ToString();
                    case NodeOperationType.ConstantMatrixDeclaration:
                        return "[" + Dimension[0] + ", " + Dimension[1] + ", " + OpSpecificValue[0].ToString() + "]";
                    default:
                        throw new Exception();
                }
            }

            if (IncomingEdges.Count == 1)
            {
                string t = IncomingEdges[0].ToString();
                if (!t.EndsWith(")")) t = t + ")";
                if (!t.StartsWith("(")) t = "(" + t;

                switch (Operation)
                {
                    case NodeOperationType.Log:
                        return "log" + t + "";
                    case NodeOperationType.Tanh:
                        return "tanh" + t + "";
                    case NodeOperationType.Transpose:
                        return "transpose" + t + "";
                    case NodeOperationType.Exp:
                        return "(e ^ " + t + " )";
                    default:
                        throw new Exception();
                }
            }

            if (IncomingEdges.Count == 2)
            {
                string l = IncomingEdges[0].ToString();
                string r = IncomingEdges[1].ToString();

                switch (Operation)
                {
                    case NodeOperationType.Add:
                        return "(" + l + " + " + r + ")";
                    case NodeOperationType.Subtract:
                        return "(" + l + " - " + r + ")";
                    case NodeOperationType.Multiply:
                        return "(" + l + " * " + r + ")";
                    case NodeOperationType.Divide:
                        return "(" + l + " / " + r + ")";
                    case NodeOperationType.Power:
                        return "(" + l + " ^ " + r + ")";
                    case NodeOperationType.Hadamard:
                        return "(" + l + " hadamard " + r + ")";
                    default:
                        throw new Exception();
                }
            }

            return "";
        }
    }
}
