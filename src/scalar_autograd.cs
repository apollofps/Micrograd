using System;
using System.Collections.Generic;

namespace Micrograd
{
    public class Value
    {
        public double Data { get; private set; }
        public double Grad { get; set; }
        private HashSet<Value> _prev;
        private string _op;
        private Action _backwardPropagation;

        public Value(double data, HashSet<Value> children = null, string op = "")
        {
            Data = data;
            Grad = 0;
            _prev = children ?? new HashSet<Value>();
            _backwardPropagation = () => { };
            _op = op;
        }

        // Addition Operator
        public static Value operator +(Value a, Value b)
        {
            b = b ?? new Value(0);
            var outValue = new Value(a.Data + b.Data, new HashSet<Value> { a, b }, "+");

            outValue._backwardPropagation = () =>
            {
                a.Grad += outValue.Grad;
                b.Grad += outValue.Grad;
            };

            return outValue;
        }

        // Multiplication Operator
        public static Value operator *(Value a, Value b)
        {
            b = b ?? new Value(1);
            var outValue = new Value(a.Data * b.Data, new HashSet<Value> { a, b }, "*");

            // Chain rule for gradients
            outValue._backwardPropagation = () =>
            {
                a.Grad += b.Data * outValue.Grad;
                b.Grad += a.Data * outValue.Grad;
            };

            return outValue;
        }

        // Subtraction Operator
        public static Value operator -(Value a, Value b) => a + (-b);

        // Negation Operator
        public static Value operator -(Value a) => a * new Value(-1);

        // Power Function
        public Value Pow(double exponent)
        {
            var outValue = new Value(Math.Pow(Data, exponent), new HashSet<Value> { this }, $"**{exponent}");

            outValue._backwardPropagation = () =>
            {
                Grad += (exponent * Math.Pow(Data, exponent - 1)) * outValue.Grad;
            };

            return outValue;
        }

        // Exponential Function
        public Value Exp()
        {
            var outValue = new Value(Math.Exp(Data), new HashSet<Value> { this }, "exp");

            outValue._backwardPropagation = () =>
            {
                Grad += outValue.Data * outValue.Grad;
            };

            return outValue;
        }

        // Hyperbolic Tangent Function
        public Value Tanh()
        {
            double tanhValue = Math.Tanh(Data);
            var outValue = new Value(tanhValue, new HashSet<Value> { this }, "tanh");

            // Derivative of tanh(x) is (1 - tanh^2(x))
            outValue._backwardPropagation = () =>
            {
                Grad += (1 - tanhValue * tanhValue) * outValue.Grad;
            };

            return outValue;
        }

        // Backpropagation (Gradient Computation)
        public void Backward()
        {
            var topologicalGraph = new List<Value>();
            var visitedNodes = new HashSet<Value>();

            void BuildTopologicalGraph(Value v)
            {
                if (!visitedNodes.Contains(v))
                {
                    visitedNodes.Add(v);
                    foreach (var child in v._prev)
                        BuildTopologicalGraph(child);
                    topologicalGraph.Add(v);
                }
            }

            BuildTopologicalGraph(this);

            Grad = 1;
            for (int i = topologicalGraph.Count - 1; i >= 0; i--)
            {
                topologicalGraph[i]._backwardPropagation();
            }
        }

        public override string ToString() => $"Value(data={Data}, grad={Grad})";
    }
}
