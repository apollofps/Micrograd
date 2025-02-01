using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Micrograd;  // Make sure you reference your Micrograd library that contains Value and MLP classes.

class Program
{
    // Use a single Random instance seeded with 1337 for reproducibility.
    static Random rng = new Random(1337);

    static void Main(string[] args)
    {
        // Load moons data from a CSV file.
        // The CSV file should have a header and three columns: x1, x2, label.
        // The label is expected to be either -1.0 or 1.0.
        var (X, y) = LoadMoonsData("moons_dataset.csv");

        // Initialize the model: 2 inputs, two hidden layers of 16 neurons each, and 1 output.
        var model = new MLP(2, new List<int> { 16, 16, 1 });

        // Open a log file to record training progress.
        string logFile = "training_logs.csv";
        using (StreamWriter writer = new StreamWriter(logFile))
        {
            writer.WriteLine("Step,Loss,Accuracy");

            // Run training for 100 steps.
            for (int step = 0; step < 100; step++)
            {
                // Compute loss and accuracy over the full dataset.
                var (totalLoss, accuracy) = ComputeLoss(X, y, model, batchSize: null);

                // Zero out gradients, perform backpropagation.
                model.ZeroGrad();
                totalLoss.Backward();

                // Update parameters with a learning rate that decays linearly from 1.0 to 0.1.
                double learningRate = 1.0 - 0.9 * step / 100.0;
                foreach (var p in model.Parameters())
                {
                    p.Data -= learningRate * p.Grad;
                }

                writer.WriteLine($"{step},{totalLoss.Data},{accuracy}");
                Console.WriteLine($"Step {step}: Loss = {totalLoss.Data}, Accuracy = {accuracy}%");
            }
        }
        
        // Export the decision boundary data to CSV.
        SaveDecisionBoundary(X, model, "decision_boundary.csv");

        Console.WriteLine("Training complete. Logs saved to training_logs.csv.");
        Console.WriteLine("Decision boundary data saved to decision_boundary.csv.");
    }

    /// <summary>
    /// Loads the moons dataset from a CSV file.
    /// Expects three columns: x1, x2, and label.
    /// The label is assumed to be in the format -1.0 or 1.0.
    /// </summary>
    static (List<List<Value>>, List<int>) LoadMoonsData(string filePath)
    {
        var X = new List<List<Value>>();
        var y = new List<int>();

        using (var reader = new StreamReader(filePath))
        {
            string header = reader.ReadLine(); // Skip header
            while (!reader.EndOfStream)
            {
                var line = reader.ReadLine();
                var values = line.Split(',');

                // Create input features.
                X.Add(new List<Value>
                {
                    new Value(double.Parse(values[0])),
                    new Value(double.Parse(values[1]))
                });

                // Parse the label.
                y.Add((int)double.Parse(values[2]));
            }
        }

        return (X, y);
    }

    /// <summary>
    /// Computes the SVM max–margin loss over a batch of samples.
    /// If batchSize is provided, a random batch is sampled; otherwise, the full dataset is used.
    /// </summary>
    static (Value, double) ComputeLoss(List<List<Value>> X, List<int> y, MLP model, int? batchSize = null)
    {
        List<List<Value>> Xb;
        List<int> yb;

        if (batchSize.HasValue)
        {
            Xb = new List<List<Value>>();
            yb = new List<int>();
            for (int i = 0; i < batchSize.Value; i++)
            {
                int idx = rng.Next(X.Count);
                Xb.Add(X[idx]);
                yb.Add(y[idx]);
            }
        }
        else
        {
            Xb = X;
            yb = y;
        }

        List<Value> losses = new List<Value>();
        List<Value> predictions = new List<Value>();

        // For each sample, compute the model score and the SVM loss.
        for (int i = 0; i < Xb.Count; i++)
        {
            var x = Xb[i];
            var target = new Value(yb[i]); // target is -1 or 1

            // Forward pass: obtain the model’s output score.
            var score = model.Call(x)[0];
            predictions.Add(score);

            // SVM max–margin loss: loss = ReLU(1 – target * score)
            var margin = new Value(1) - target * score;
            var loss = margin.ReLU();
            losses.Add(loss);
        }

        // Average the data loss over the batch.
        var dataLoss = losses.Aggregate((a, b) => a + b) / new Value(losses.Count);
        // L2 regularization with alpha = 1e-4.
        var alpha = new Value(1e-4);
        var regLoss = alpha * model.Parameters().Select(p => p * p).Aggregate((a, b) => a + b);
        var totalLoss = dataLoss + regLoss;

        // Compute accuracy: prediction is 1 if score > 0, otherwise -1.
        double accuracy = CalculateAccuracy(predictions, yb);

        return (totalLoss, accuracy);
    }

    /// <summary>
    /// Calculates accuracy given model output scores and target labels.
    /// </summary>
    static double CalculateAccuracy(List<Value> predictions, List<int> targets)
    {
        int correct = 0;
        for (int i = 0; i < predictions.Count; i++)
        {
            int predictedLabel = predictions[i].Data > 0 ? 1 : -1;
            if (predictedLabel == targets[i])
                correct++;
        }
        return (double)correct / predictions.Count * 100;
    }

    /// <summary>
    /// Generates a mesh grid over the range of the input data, evaluates the model on each grid point,
    /// and saves the decision (1 if the model's output is positive, 0 otherwise) along with the grid coordinates.
    /// </summary>
    static void SaveDecisionBoundary(List<List<Value>> X, MLP model, string filePath)
    {
        double h = 0.25;
        double x_min = X.Min(row => row[0].Data) - 1;
        double x_max = X.Max(row => row[0].Data) + 1;
        double y_min = X.Min(row => row[1].Data) - 1;
        double y_max = X.Max(row => row[1].Data) + 1;

        using (StreamWriter writer = new StreamWriter(filePath))
        {
            writer.WriteLine("x,y,decision");

            for (double y_val = y_min; y_val <= y_max; y_val += h)
            {
                for (double x_val = x_min; x_val <= x_max; x_val += h)
                {
                    var input = new List<Value> { new Value(x_val), new Value(y_val) };
                    var score = model.Call(input)[0];
                    int decision = score.Data > 0 ? 1 : 0;
                    writer.WriteLine($"{x_val},{y_val},{decision}");
                }
            }
        }
    }
}
