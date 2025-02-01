using System;
using System.Collections.Generic;
using System.Linq;

namespace Micrograd
{
    public class MLP : Module
    {
        private List<Layer> layers;

        public MLP(int nin, List<int> nouts)
        {
            List<int> sz = new List<int> { nin };
            sz.AddRange(nouts);
            layers = Enumerable.Range(0, nouts.Count)
                               .Select(i => new Layer(sz[i], sz[i + 1], i != nouts.Count - 1))
                               .ToList();
        }

        public List<Value> Call(List<Value> x)
        {
            foreach (var layer in layers)
            {
                x = layer.Call(x);
            }
            return x;
        }

        public override List<Value> Parameters() => layers.SelectMany(layer => layer.Parameters()).ToList();

        public override string ToString() => $"MLP of [{string.Join(", ", layers)}]";
    }
}