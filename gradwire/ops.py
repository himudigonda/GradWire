import numpy as np
from .core import Op, Node # Import base classes

class AddOp(Op):
    """Op to element-wise add two nodes."""
    def __call__(self, node_A, node_B):
        new_node = super().__call__() # Op.__call__ creates Node and sets op
        new_node.inputs = [node_A, node_B]
        new_node.name = "(%s+%s)" % (node_A.name, node_B.name)
        return new_node

    def compute(self, node, input_vals):
        assert len(input_vals) == 2, "AddOp compute takes 2 input values"
        return input_vals[0] + input_vals[1]

    def gradient(self, node, output_grad):
        # output_grad is a Node. The gradients returned must also be Nodes.
        # When output_grad is evaluated, it will be a numerical array.
        # The gradient operations here should be symbolic.
        # If output_grad is dL/dY and Y = A+B, then dL/dA = dL/dY * 1, dL/dB = dL/dY * 1
        return [output_grad, output_grad]

class AddByConstOp(Op):
    """Op to element-wise add a node by a constant."""
    def __call__(self, node_A, const_val):
        new_node = super().__call__()
        new_node.inputs = [node_A]
        new_node.const_attr = const_val
        new_node.name = "(%s+%s)" % (node_A.name, str(const_val))
        return new_node

    def compute(self, node, input_vals):
        assert len(input_vals) == 1, "AddByConstOp compute takes 1 input value"
        return input_vals[0] + node.const_attr

    def gradient(self, node, output_grad):
        # If Y = A + C, dL/dA = dL/dY * 1
        return [output_grad]

class MulOp(Op):
    """Op to element-wise multiply two nodes."""
    def __call__(self, node_A, node_B):
        new_node = super().__call__()
        new_node.inputs = [node_A, node_B]
        new_node.name = "(%s*%s)" % (node_A.name, node_B.name)
        return new_node

    def compute(self, node, input_vals):
        assert len(input_vals) == 2, "MulOp compute takes 2 input values"
        return input_vals[0] * input_vals[1]

    def gradient(self, node, output_grad):
        # If Y = A*B, dL/dA = dL/dY * B, dL/dB = dL/dY * A
        # node.inputs[0] is A, node.inputs[1] is B
        # output_grad is dL/dY (a Node)
        # We need to return Node objects: (dL/dY * B) and (dL/dY * A)
        # This uses the overloaded __mul__ on Node objects.
        grad_A = output_grad * node.inputs[1]
        grad_B = output_grad * node.inputs[0]
        return [grad_A, grad_B]

class MulByConstOp(Op):
    """Op to element-wise multiply a node by a constant."""
    def __call__(self, node_A, const_val):
        new_node = super().__call__()
        new_node.inputs = [node_A]
        new_node.const_attr = const_val
        new_node.name = "(%s*%s)" % (node_A.name, str(const_val))
        return new_node

    def compute(self, node, input_vals):
        assert len(input_vals) == 1, "MulByConstOp compute takes 1 input value"
        return input_vals[0] * node.const_attr

    def gradient(self, node, output_grad):
        # If Y = A*C, dL/dA = dL/dY * C
        # node.const_attr is C
        # output_grad is dL/dY (a Node)
        # Return a Node: (dL/dY * C)
        grad_A = output_grad * node.const_attr # Node * constant
        return [grad_A]

class MatMulOp(Op):
    """Op to matrix multiply two nodes."""
    def __call__(self, node_A, node_B, trans_A=False, trans_B=False):
        new_node = super().__call__()
        new_node.inputs = [node_A, node_B]
        new_node.matmul_attr_trans_A = trans_A
        new_node.matmul_attr_trans_B = trans_B
        new_node.name = "MatMul(%s,%s,A_T=%s,B_T=%s)" % (
            node_A.name, node_B.name, str(trans_A), str(trans_B)
        )
        return new_node

    def compute(self, node, input_vals):
        assert len(input_vals) == 2, "MatMulOp compute takes 2 input values"
        mat_A = input_vals[0]
        mat_B = input_vals[1]

        if node.matmul_attr_trans_A:
            mat_A = np.transpose(mat_A)
        if node.matmul_attr_trans_B:
            mat_B = np.transpose(mat_B)
        
        return np.matmul(mat_A, mat_B)

    def gradient(self, node, output_grad):
        # Y = A @ B (simplified, ignoring initial transpose for formula)
        # dL/dA = dL/dY @ B^T
        # dL/dB = A^T @ dL/dY
        # output_grad is dL/dY (a Node)
        # node.inputs[0] is A_node, node.inputs[1] is B_node
        # node.matmul_attr_trans_A and node.matmul_attr_trans_B are transposes for forward pass.

        A_node, B_node = node.inputs
        
        # Gradient w.r.t. A (first input)
        # If forward was Y = A @ B, grad_A = dY @ B.T
        # If forward was Y = A.T @ B, grad_A_orig = (dY @ B.T).T = B @ dY.T
        # If forward was Y = A @ B.T, grad_A = dY @ (B.T).T = dY @ B
        # If forward was Y = A.T @ B.T, grad_A_orig = (dY @ (B.T).T).T = (dY @ B).T = B.T @ dY.T

        if not node.matmul_attr_trans_A: # A was not transposed in forward
            # dL/dA = dL/dY @ B_eff^T
            # B_eff is B if !trans_B, B.T if trans_B
            # So B_eff^T is B.T if !trans_B, B if trans_B
            grad_A = matmul_op(output_grad, B_node, trans_A=False, trans_B=not node.matmul_attr_trans_B)
        else: # A was transposed in forward (A_orig.T)
            # dL/dA_orig = (dL/dY @ B_eff^T)^T = B_eff @ (dL/dY)^T
            # B_eff is B if !trans_B, B.T if trans_B
            grad_A = matmul_op(B_node, output_grad, trans_A=node.matmul_attr_trans_B, trans_B=True)


        # Gradient w.r.t. B (second input)
        # If forward was Y = A @ B, grad_B = A.T @ dY
        # If forward was Y = A.T @ B, grad_B = (A.T).T @ dY = A @ dY
        # If forward was Y = A @ B.T, grad_B_orig = (A.T @ dY).T = dY.T @ A
        # If forward was Y = A.T @ B.T, grad_B_orig = ((A.T).T @ dY).T = (A @ dY).T = dY.T @ A.T

        if not node.matmul_attr_trans_B: # B was not transposed in forward
            # dL/dB = A_eff^T @ dL/dY
            # A_eff is A if !trans_A, A.T if trans_A
            # So A_eff^T is A.T if !trans_A, A if trans_A
            grad_B = matmul_op(A_node, output_grad, trans_A=not node.matmul_attr_trans_A, trans_B=False)
        else: # B was transposed in forward (B_orig.T)
            # dL/dB_orig = (A_eff^T @ dL/dY)^T = (dL/dY)^T @ A_eff
            # A_eff is A if !trans_A, A.T if trans_A
            grad_B = matmul_op(output_grad, A_node, trans_A=True, trans_B=node.matmul_attr_trans_A)
            
        return [grad_A, grad_B]


class PlaceholderOp(Op):
    """Op to feed value to a node."""
    def __call__(self): # Name is set by Variable()
        new_node = super().__call__()
        # No inputs for a placeholder from other nodes in the graph definition
        return new_node

    def compute(self, node, input_vals):
        # This should not be called. Placeholder values are fed directly by Executor.
        assert False, "PlaceholderOp compute should not be called. Values are provided by feed_dict."

    def gradient(self, node, output_grad):
        # Placeholders are leaf nodes in terms of graph construction for forward pass.
        # Their gradient is accumulated from output_grad but not propagated further back through ops.
        return None # No inputs to propagate gradient to

class ZerosLikeOp(Op):
    """Op that represents a constant np.zeros_like."""
    def __call__(self, node_A):
        new_node = super().__call__()
        new_node.inputs = [node_A] # Depends on node_A for shape
        new_node.name = "ZerosLike(%s)" % node_A.name
        return new_node

    def compute(self, node, input_vals):
        assert len(input_vals) == 1, "ZerosLikeOp compute takes 1 input value (for shape)"
        # input_vals[0] is the evaluated value of node_A
        assert isinstance(input_vals[0], np.ndarray), "Input to ZerosLikeOp must be a NumPy array for shape"
        return np.zeros_like(input_vals[0])

    def gradient(self, node, output_grad):
        # Gradient of a constant (zeros_like is constant once shape is known) is zero.
        # It has one input (node_A, for shape). Grad w.r.t node_A is 0.
        return [zeroslike_op(node.inputs[0])] # Symbolic zero of the same shape as node_A

class OnesLikeOp(Op):
    """Op that represents a constant np.ones_like."""
    def __call__(self, node_A):
        new_node = super().__call__()
        new_node.inputs = [node_A] # Depends on node_A for shape
        new_node.name = "OnesLike(%s)" % node_A.name
        return new_node

    def compute(self, node, input_vals):
        assert len(input_vals) == 1, "OnesLikeOp compute takes 1 input value (for shape)"
        assert isinstance(input_vals[0], np.ndarray), "Input to OnesLikeOp must be a NumPy array for shape"
        return np.ones_like(input_vals[0])

    def gradient(self, node, output_grad):
        # Gradient of a constant is zero.
        return [zeroslike_op(node.inputs[0])] # Symbolic zero of the same shape as node_A

# --- Create global singleton instances of operators ---
add_op = AddOp()
mul_op = MulOp()
add_byconst_op = AddByConstOp()
mul_byconst_op = MulByConstOp()
matmul_op = MatMulOp()
placeholder_op = PlaceholderOp()
oneslike_op = OnesLikeOp()
zeroslike_op = ZerosLikeOp()

# --- Convenience function for Variables ---
def Variable(name=""):
    """User defined variables in an expression.
        e.g. x = Variable(name = "x")
    Acts as a named placeholder.
    """
    node = placeholder_op() # Use the singleton PlaceholderOp
    node.name = name
    return node
