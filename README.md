# Micrograd

Welcome to **Micrograd** – a mini-autograd engine built in C#! This project is lit because it lets you train your own neural nets from scratch, flexing on tasks like binary classification with the moons dataset. Why C# you might ask, Idk why

### What's Poppin'
- **Autograd Engine:** Built from scratch to automatically compute gradients so you can backprop like a boss.  
- **Neural Network Components:** Rockin' neurons and layers (peep [Neuron.cs](Micrograd/src/Neuron.cs) and [MLP.cs](Micrograd/src/MLP.cs) for the deets).  
- **Training Vibes:** Loads up the moons dataset ([moons_dataset.csv](Micrograd/moons_dataset.csv)) and spits out training logs, loss curves, and decision boundaries. Check out [program.cs](Micrograd/program.cs) for the main action.

### Extra Swag
- **Hyperlinks for the Nerds:**  
  - [Project File](Micrograd/Micrograd.csproj) – where the magic is set up.
  - [Autograd Scalar](Micrograd/Micrograd.csproj) – take a crash course of you high school calculus before clicking this   
  - [Module Base](Micrograd/src/Module.cs) – keepin' it real with parameter zeroing. 
  - [Neuron Details](Micrograd/src/Neuron.cs) – Building blocks for our neural nets.  
  - [MLP Layers](Micrograd/src/MLP.cs) – The guts of our network architecture.  
  - [Main Runner](Micrograd/program.cs) – Where it all kicks off. 
- **Dope Visuals:**  
  ![Neural Network Viz](https://www.altexsoft.com/static/content-image/2024/6/ec55ad04-11ca-44f3-8eec-9399936c26ff.png)

### Decision line from our little binary classification experiment trained using micrograd.Ps: Blood, Sweat and tears were split.
![Decision Split](Micrograd/img/output.png)

### How to Roll with It
1. Clone the repo and open it in VS Code.
2. Run `dotnet run` in the [terminal](https://github.com/) to train the model.
3. Peep the logs in `training_logs.csv` and the sweet decision boundary in `decision_boundary.csv`.


