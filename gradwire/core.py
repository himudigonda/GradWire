import numpy as np


class Node(object):
    """Node in a computation graph."""

    def __init__(self):
        """Constructor, new node is indirectly created by Op object __call__ method.

        Instance variables
        ------------------
        self.inputs: the list of input nodes.
        self.op: the associated op object,
            e.g. add_op object if this node is created by adding two other nodes.
        self.const_attr: the add or multiply constant,
            e.g. self.const_attr=5 if this node is created by x+5.
        self.name: node name for debugging purposes.
        """
        self.inputs = []
        self.op = None
        self.const_attr = None  # Used for ops like AddByConst, MulByConst
        self.name = ""
        # For MatMulOp specifically
        self.matmul_attr_trans_A = False
        self.matmul_attr_trans_B = False

    def __add__(self, other):
        """Adding two nodes or a node and a constant returns a new node."""
        from .ops import add_op, add_byconst_op  # Local import for ops

        if isinstance(other, Node):
            new_node = add_op(self, other)
        else:
            # Add by a constant
            new_node = add_byconst_op(self, other)
        return new_node

    def __mul__(self, other):
        """Multiplying two nodes or a node and a constant returns a new node."""
        from .ops import mul_op, mul_byconst_op  # Local import for ops

        if isinstance(other, Node):
            new_node = mul_op(self, other)
        else:
            # Multiply by a constant
            new_node = mul_byconst_op(self, other)
        return new_node

    # Allow left-hand-side add and multiply.
    __radd__ = __add__
    __rmul__ = __mul__

    def __str__(self):
        """Allow print to display node name."""
        return self.name

    __repr__ = __str__


class Op(object):
    """Op represents operations performed on nodes."""

    def __call__(self, *args, **kwargs):
        """
        Base __call__ for Op. Specific ops should override this to
        properly set inputs and other attributes on the new_node.
        This base method primarily creates the Node and assigns self as op.
        """
        new_node = Node()
        new_node.op = self
        return new_node

    def compute(self, node, input_vals):
        """
        Given values of input nodes, compute the output value.

        Parameters
        ----------
        node: node that performs the compute.
        input_vals: values of input nodes.

        Returns
        -------
        An output value of the node.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement compute method."
        )

    def gradient(self, node, output_grad):
        """
        Given value of output gradient, compute gradient contributions to each input node.

        Parameters
        ----------
        node: node that performs the gradient.
        output_grad: value of output gradient summed from children nodes' contributions.
                     This is a Node object representing the symbolic gradient.

        Returns
        -------
        A list of gradient contributions (Node objects) to each input node respectively.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement gradient method."
        )
