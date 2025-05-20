import numpy as np

# from .core import Node # Not strictly needed unless for type hinting


class Executor:
    """Executor computes values for a given subset of nodes in a computation graph."""

    def __init__(self, eval_node_list):
        """
        Parameters
        ----------
        eval_node_list: list of nodes whose values need to be computed.
        """
        self.eval_node_list = eval_node_list

    def run(self, feed_dict):
        """
        Computes values of nodes in eval_node_list given computation graph.

        Parameters
        ----------
        feed_dict: dict mapping variable nodes (Placeholders) to their numerical values.

        Returns
        -------
        A list of numerical values for nodes in eval_node_list.
        """
        node_to_val_map = dict(feed_dict)  # Initialize with fed values

        # Traverse graph in topological sort order and compute values for all nodes.
        # The HASH of nodes (their memory address) is used for dict keys and set membership.
        topo_order = find_topo_sort(self.eval_node_list)

        for node in topo_order:
            if (
                node in node_to_val_map
            ):  # Value already provided (e.g., feed_dict) or computed
                continue

            # Collect input values from already computed nodes
            input_vals = []
            for input_node in node.inputs:
                assert (
                    input_node in node_to_val_map
                ), f"Input node {input_node} value not found for computing {node}"
                input_vals.append(node_to_val_map[input_node])

            # Compute node value using its operation
            # node.op should be an instance of a subclass of Op
            # node.op.compute expects the node itself and the list of its input_values
            node_val = node.op.compute(node, input_vals)
            node_to_val_map[node] = node_val

        # Collect results for the specifically requested evaluation nodes
        node_val_results = [node_to_val_map[node] for node in self.eval_node_list]
        return (
            node_val_results
            if len(node_val_results) > 1
            else node_val_results[0] if len(node_val_results) == 1 else []
        )


def find_topo_sort(node_list):
    """
    Given a list of nodes, return a topological sort list of nodes
    ending in them (including all their dependencies).
    """
    visited = set()
    topo_order = []
    for node in node_list:
        _topo_sort_dfs(node, visited, topo_order)
    return topo_order


def _topo_sort_dfs(node, visited, topo_order):
    """Post-order DFS helper for topological sort."""
    if node in visited:
        return
    visited.add(node)
    for n in node.inputs:
        _topo_sort_dfs(n, visited, topo_order)
    topo_order.append(node)
