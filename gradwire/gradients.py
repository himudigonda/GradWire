import numpy as np
from .executor import find_topo_sort  # Re-use from executor
from .ops import oneslike_op, zeroslike_op, add_op  # Need op instances

# from .core import Node # Not strictly needed unless for type hinting


def gradients(output_node, node_list):
    """
    Take gradient of output_node with respect to each node in node_list.

    Parameters
    ----------
    output_node: The node we are taking the derivative of (e.g., loss).
    node_list: A list of nodes with respect to which we want gradients.

    Returns
    -------
    A list of Node objects representing the symbolic gradients, one for each
    node in node_list respectively.
    """

    # node_to_output_grads_list:
    #   Maps a node to a LIST of symbolic gradient Nodes flowing INTO it
    #   from its children in the forward graph (i.e., its consumers during forward pass).
    #   These are the d(output_node)/d(child_output) * d(child_output)/d(node) terms.
    node_to_output_grads_list = {}

    # Initialize gradient of output_node with respect to itself as 1.
    # This is a symbolic '1' with the same shape as output_node.
    node_to_output_grads_list[output_node] = [oneslike_op(output_node)]

    # node_to_output_grad:
    #   Maps a node to a SINGLE symbolic gradient Node representing the
    #   total d(output_node)/d(node). This is the sum of all paths from
    #   output_node back to this node.
    node_to_output_grad = {}

    # Traverse graph in reverse topological order.
    reverse_topo_order = reversed(find_topo_sort([output_node]))

    for node in reverse_topo_order:
        # Sum up all incoming gradient contributions for the current 'node'.
        # These contributions are already symbolic Node objects.
        assert node in node_to_output_grads_list, (
            f"Node {node} from reverse_topo_order not found in gradient contribution list. "
            f"This should not happen if output_node is handled correctly."
        )

        current_grad_terms = node_to_output_grads_list[node]
        if len(current_grad_terms) == 1:
            total_incoming_grad_for_node = current_grad_terms[0]
        else:
            # sum_node_list will create a chain of AddOp nodes if multiple terms
            total_incoming_grad_for_node = sum_node_list(current_grad_terms)

        node_to_output_grad[node] = total_incoming_grad_for_node

        if (
            not node.inputs
        ):  # If node has no inputs (e.g., Placeholder, constant from op like ZerosLike)
            continue  # No further gradients to propagate "through" this node's op

        # Propagate gradients to the inputs of the current 'node'.
        # node.op.gradient(node, total_incoming_grad_for_node) computes:
        # [d(output_node)/d(node) * d(node)/d(input_1),
        #  d(output_node)/d(node) * d(node)/d(input_2), ...]
        # Each term is a symbolic Node.
        input_specific_grads = node.op.gradient(node, total_incoming_grad_for_node)

        if input_specific_grads is None:  # PlaceholderOp.gradient returns None
            continue

        assert len(input_specific_grads) == len(node.inputs), (
            f"Number of gradients from op {node.op} ({len(input_specific_grads)})"
            f" does not match number of inputs ({len(node.inputs)}) for node {node}."
        )

        for i, grad_for_input_node in enumerate(input_specific_grads):
            input_node = node.inputs[i]
            if input_node not in node_to_output_grads_list:
                node_to_output_grads_list[input_node] = []
            node_to_output_grads_list[input_node].append(grad_for_input_node)

    # Collect results for the specifically requested gradient nodes.
    grad_results_for_node_list = []
    for req_node in node_list:
        if req_node in node_to_output_grad:
            grad_results_for_node_list.append(node_to_output_grad[req_node])
        else:
            # If req_node was not reached during backward pass from output_node,
            # its gradient d(output_node)/d(req_node) is zero.
            # Create a symbolic zero with the shape of req_node.
            grad_results_for_node_list.append(zeroslike_op(req_node))

    return grad_results_for_node_list


def sum_node_list(node_list_to_sum):
    """
    Custom sum function to build a graph of additions from a list of nodes.
    Avoids issues with Python's sum() potentially not using Node.__add__ correctly
    in all contexts or creating too many intermediate Python objects.
    """
    from functools import reduce

    if not node_list_to_sum:
        # This should ideally be handled by the caller or represent a zero node
        # of appropriate shape. For now, raise error.
        raise ValueError("sum_node_list cannot sum an empty list.")

    # operator.add on Node objects calls Node.__add__, which uses add_op.
    # This creates a chain of AddOp nodes: (n1+n2)+n3 ...
    return reduce(lambda x, y: add_op(x, y), node_list_to_sum)
