# GradWire: A NumPy-based Automatic Differentiation Engine

GradWire is a Python library that implements automatic differentiation (autodiff) using the reverse mode, also known as backpropagation. It allows users to define computations as a dynamic graph and then automatically compute gradients of any node with respect to any other nodes in the graph. This project is built from scratch using only Python and NumPy, providing a clear look into the mechanics of autodiff engines found in modern deep learning frameworks.

## Features

*   **Dynamic Computation Graphs:** Define complex mathematical expressions that are dynamically converted into a computation graph.
*   **Reverse-Mode Automatic Differentiation:** Efficiently calculate gradients of a scalar output with respect to many inputs.
*   **NumPy Backend:** All numerical computations are performed using NumPy, allowing for vectorized operations on arrays.
*   **Operator Overloading:** Intuitive graph construction using standard Python operators (`+`, `*`) for `GradWire` nodes.
*   **Supported Operations:**
    *   Addition (Node + Node, Node + Constant)
    *   Multiplication (Node * Node, Node * Constant)
    *   Matrix Multiplication (`matmul_op`) with optional transposition of inputs.
*   **Higher-Order Gradients:** Compute gradients of gradients (e.g., second-order derivatives).
*   **Symbolic Gradient Representation:** The `gradients()` function returns new `Node` objects representing the symbolic derivative expressions, which can then be evaluated.
*   **Clear Execution Model:** An `Executor` class evaluates specified nodes in the graph given a `feed_dict` for input values.

## How it Works: A Glimpse

GradWire constructs a computation graph where each node represents either a variable or the result of an operation.

**1. Forward Pass (Graph Construction & Evaluation):**
When you define an expression like `y = x1 * x2 + 5`, GradWire builds a graph:
*   `x1` and `x2` are `PlaceholderOp` nodes.
*   `temp = x1 * x2` becomes a `MulOp` node with `x1` and `x2` as inputs.
*   `y = temp + 5` becomes an `AddByConstOp` node with `temp` as input.

The `Executor` traverses this graph in topological order, computing the numerical value of each node.

**Example from logs (`test_add_by_const`):**
```
--- Running test_add_by_const ---
[test_add_by_const] Node 'x2' (x2): Op=PlaceholderOp, Inputs=[], ConstAttr=None
[test_add_by_const] Node 'y' ((x2+5)): Op=AddByConstOp, Inputs=['x2'], ConstAttr=5
...
[test_add_by_const] x2_val (feed_dict): [2. 2. 2.]
[test_add_by_const] Running executor...
[test_add_by_const] y_val (computed): [7. 7. 7.]
```

**2. Backward Pass (Gradient Computation):**
The `gradients(output_node, list_of_wrt_nodes)` function performs reverse-mode autodiff:
*   It starts from the `output_node` and assigns it a gradient of `1` (symbolically, an `OnesLikeOp` node).
*   It traverses the graph backward (reverse topological order).
*   For each node, it uses the chain rule to propagate and accumulate gradients:
    *   The gradient of an operation's input is the gradient of its output multiplied by the local derivative of the output with respect to that input.
*   This process also builds a *new graph* representing the symbolic derivative expressions.

**Example from logs (`test_add_mul_mix_1` where `y = x1 + x2 * x3 * x1`):**
The gradient `dy/dx2` (simplified) is `(dy/dy) * (x1*x3)`. The log shows the constructed gradient node:
```
[test_add_mul_mix_1] Node 'grad_x2_node' (((OnesLike((x1+((x2*x3)*x1)))*x1)*x3)): Op=MulOp, Inputs=['(OnesLike((x1+((x2*x3)*x1)))*x1)', 'x3'], ConstAttr=None
```
This node represents `( ( (dL/dy) * x1_node ) * x3_node )`. When evaluated, it gives the numerical gradient.

The symbolic representation of gradients can become quite complex for nested expressions, as seen in the `test_grad_of_grad` or `test_add_mul_mix_3` logs for `grad_x2_node`. However, the `Executor` handles their evaluation correctly.

## Project Structure

```
GradWire/
├── gradwire/               # The GradWire library package
│   ├── __init__.py         # Initializes the package and exports API
│   ├── core.py             # Defines Node and base Op classes
│   ├── ops.py              # Defines specific operations (AddOp, MulOp, MatMulOp, etc.)
│   ├── executor.py         # Defines the Executor for graph evaluation
│   └── gradients.py        # Implements the reverse-mode autodiff logic
│
├── tests/                  # Unit and integration tests
│   ├── __init__.py
│   └── test_engine.py      # Test suite for the GradWire engine
│
├── setup.sh                # Script to set up the virtual environment and install dependencies
├── requirements.txt        # Project dependencies (NumPy)
└── README.md               # This file
```

## Installation and Setup

1.  **Clone the repository (if applicable) or ensure all files are in place.**
2.  **Prerequisites:** Python 3.6+
3.  **Set up the environment:**
    Open your terminal in the project root directory and run the setup script:
    ```bash
    bash setup.sh
    ```
    This will create a Python virtual environment named `venv` and install NumPy.

4.  **Activate the virtual environment:**
    ```bash
    source venv/bin/activate
    ```
    On Windows, it might be `venv\Scripts\activate`.

## Usage Example

```python
import gradwire as gw
import numpy as np

# Define variables
x1 = gw.Variable(name="x1")
x2 = gw.Variable(name="x2")

# Define a computation
# y = (x1 * x2) + x1
term1 = x1 * x2
y = term1 + x1

# Get symbolic gradients of y with respect to x1 and x2
grad_y_wrt_x1_node, grad_y_wrt_x2_node = gw.gradients(y, [x1, x2])

# Create an executor to evaluate nodes
executor = gw.Executor([y, grad_y_wrt_x1_node, grad_y_wrt_x2_node])

# Provide numerical values for input variables
x1_val = np.array([2.0])
x2_val = np.array([3.0])

# Run the computation
y_val, grad_y_x1_val, grad_y_x2_val = executor.run(
    feed_dict={x1: x1_val, x2: x2_val}
)

print(f"x1 value: {x1_val}")
print(f"x2 value: {x2_val}")
print(f"y = (x1 * x2) + x1  => y_value: {y_val}") # Expected: (2*3)+2 = 8
print(f"Gradient dy/dx1: {grad_y_x1_val}")       # Expected: x2 + 1 = 3 + 1 = 4
print(f"Gradient dy/dx2: {grad_y_x2_val}")       # Expected: x1 = 2
```

## Running Tests

After setting up and activating the virtual environment, navigate to the project root directory and run:

```bash
python -m unittest discover tests
```
For more verbose output:
```bash
python -m unittest discover -v tests
```
All 10 tests should pass, covering various operations, combinations, and higher-order gradients. The test output can be made extremely verbose by uncommenting `print` statements within `tests/test_engine.py` to trace graph construction and numerical evaluation.

## Future Enhancements (Potential)

*   Broader range of mathematical operations (e.g., division, power, exp, log, trigonometric functions).
*   Support for more complex neural network layers (e.g., Dense, Convolutional).
*   Broadcasting rules for NumPy array operations.
*   Basic optimization passes on the graph (e.g., constant folding).
*   More sophisticated error handling and shape inference.

