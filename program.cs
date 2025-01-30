using System;
using Micrograd; 
class Program
{
    static void Main()
    {
        // Inputs
        var x1 = new Value(2.0);
        var x2 = new Value(0.0);
        
        // Weights
        var w1 = new Value(-3);
        var w2 = new Value(1);
        
        // Bias
        var b = new Value(6.881);

        // Forward Pass: Compute Linear Combination
        var n = x1 * w1 + x2 * w2 + b;
        var o = n.Tanh();

        // Backward Pass (Autograd)
        o.Backward(); // No need to assign it to 'o'

        // Output Results
        Console.WriteLine("Results:");
        Console.WriteLine($"x1: {x1}");
        Console.WriteLine($"x2: {x2}");
        Console.WriteLine($"w1: {w1}");
        Console.WriteLine($"w2: {w2}");
        Console.WriteLine($"b: {b}");
        Console.WriteLine($"n: {n}");
        Console.WriteLine($"o: {o}");
    }
}
