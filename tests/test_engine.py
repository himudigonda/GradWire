# tests/test_engine.py

import unittest
import gradwire as gw
import numpy as np

# Helper function to print node details (optional, can be inlined)
def log_node_details(test_name, node_var_name, node_obj):
    if node_obj is None:
        print(f"[{test_name}] {node_var_name}: None")
        return
    input_names = [inp.name for inp in node_obj.inputs] if hasattr(node_obj, 'inputs') else "N/A"
    op_name = node_obj.op.__class__.__name__ if hasattr(node_obj, 'op') and node_obj.op else "N/A"
    const_attr = node_obj.const_attr if hasattr(node_obj, 'const_attr') else "N/A"
    print(f"[{test_name}] Node '{node_var_name}' ({node_obj.name}): Op={op_name}, Inputs={input_names}, ConstAttr={const_attr}")

class TestGradWireEngine(unittest.TestCase):

    def test_identity(self):
        test_name = "test_identity"
        print(f"\n--- Running {test_name} ---")
        x2 = gw.Variable(name="x2")
        log_node_details(test_name, "x2", x2)
        y = x2
        log_node_details(test_name, "y (after y=x2 assignment)", y) # y should be same object as x2

        print(f"[{test_name}] Calculating gradients for y w.r.t [x2]...")
        grad_x2_node, = gw.gradients(y, [x2])
        log_node_details(test_name, "grad_x2_node", grad_x2_node)

        print(f"[{test_name}] Creating executor for [y, grad_x2_node]...")
        executor = gw.Executor([y, grad_x2_node])
        x2_val = 2 * np.ones(3)
        print(f"[{test_name}] x2_val (feed_dict): {x2_val}")

        print(f"[{test_name}] Running executor...")
        results = executor.run(feed_dict={x2: x2_val})
        y_val = results[0]
        grad_x2_val = results[1]
        print(f"[{test_name}] y_val (computed): {y_val}")
        print(f"[{test_name}] grad_x2_val (computed): {grad_x2_val}")

        expected_y_val = x2_val
        expected_grad_x2_val = np.ones_like(x2_val)
        print(f"[{test_name}] y_val (expected): {expected_y_val}")
        print(f"[{test_name}] grad_x2_val (expected): {expected_grad_x2_val}")

        self.assertIsInstance(y, gw.Node)
        self.assertTrue(np.array_equal(y_val, expected_y_val))
        self.assertTrue(np.array_equal(grad_x2_val, expected_grad_x2_val))
        print(f"--- {test_name} PASSED ---")

    def test_add_by_const(self):
        test_name = "test_add_by_const"
        print(f"\n--- Running {test_name} ---")
        x2 = gw.Variable(name="x2")
        log_node_details(test_name, "x2", x2)
        y = 5 + x2
        log_node_details(test_name, "y", y)

        print(f"[{test_name}] Calculating gradients for y w.r.t [x2]...")
        grad_x2_node, = gw.gradients(y, [x2])
        log_node_details(test_name, "grad_x2_node", grad_x2_node)

        print(f"[{test_name}] Creating executor for [y, grad_x2_node]...")
        executor = gw.Executor([y, grad_x2_node])
        x2_val = 2 * np.ones(3)
        print(f"[{test_name}] x2_val (feed_dict): {x2_val}")

        print(f"[{test_name}] Running executor...")
        y_val, grad_x2_val = executor.run(feed_dict={x2: x2_val})
        print(f"[{test_name}] y_val (computed): {y_val}")
        print(f"[{test_name}] grad_x2_val (computed): {grad_x2_val}")

        expected_y_val = x2_val + 5
        expected_grad_x2_val = np.ones_like(x2_val)
        print(f"[{test_name}] y_val (expected): {expected_y_val}")
        print(f"[{test_name}] grad_x2_val (expected): {expected_grad_x2_val}")

        self.assertIsInstance(y, gw.Node)
        self.assertTrue(np.array_equal(y_val, expected_y_val))
        self.assertTrue(np.array_equal(grad_x2_val, expected_grad_x2_val))
        print(f"--- {test_name} PASSED ---")

    def test_mul_by_const(self):
        test_name = "test_mul_by_const"
        print(f"\n--- Running {test_name} ---")
        x2 = gw.Variable(name="x2")
        log_node_details(test_name, "x2", x2)
        y = 5 * x2
        log_node_details(test_name, "y", y)

        print(f"[{test_name}] Calculating gradients for y w.r.t [x2]...")
        grad_x2_node, = gw.gradients(y, [x2])
        log_node_details(test_name, "grad_x2_node", grad_x2_node)

        print(f"[{test_name}] Creating executor for [y, grad_x2_node]...")
        executor = gw.Executor([y, grad_x2_node])
        x2_val = 2 * np.ones(3)
        print(f"[{test_name}] x2_val (feed_dict): {x2_val}")

        print(f"[{test_name}] Running executor...")
        y_val, grad_x2_val = executor.run(feed_dict={x2: x2_val})
        print(f"[{test_name}] y_val (computed): {y_val}")
        print(f"[{test_name}] grad_x2_val (computed): {grad_x2_val}")

        expected_y_val = x2_val * 5
        expected_grad_x2_val = np.ones_like(x2_val) * 5
        print(f"[{test_name}] y_val (expected): {expected_y_val}")
        print(f"[{test_name}] grad_x2_val (expected): {expected_grad_x2_val}")

        self.assertIsInstance(y, gw.Node)
        self.assertTrue(np.array_equal(y_val, expected_y_val))
        self.assertTrue(np.array_equal(grad_x2_val, expected_grad_x2_val))
        print(f"--- {test_name} PASSED ---")

    def test_add_two_vars(self):
        test_name = "test_add_two_vars"
        print(f"\n--- Running {test_name} ---")
        x2 = gw.Variable(name="x2")
        log_node_details(test_name, "x2", x2)
        x3 = gw.Variable(name="x3")
        log_node_details(test_name, "x3", x3)
        y = x2 + x3
        log_node_details(test_name, "y", y)

        print(f"[{test_name}] Calculating gradients for y w.r.t [x2, x3]...")
        grad_x2_node, grad_x3_node = gw.gradients(y, [x2, x3])
        log_node_details(test_name, "grad_x2_node", grad_x2_node)
        log_node_details(test_name, "grad_x3_node", grad_x3_node)

        print(f"[{test_name}] Creating executor for [y, grad_x2_node, grad_x3_node]...")
        executor = gw.Executor([y, grad_x2_node, grad_x3_node])
        x2_val = 2 * np.ones(3)
        x3_val = 3 * np.ones(3)
        print(f"[{test_name}] x2_val (feed_dict): {x2_val}")
        print(f"[{test_name}] x3_val (feed_dict): {x3_val}")

        print(f"[{test_name}] Running executor...")
        y_val, grad_x2_val, grad_x3_val = executor.run(feed_dict={x2: x2_val, x3: x3_val})
        print(f"[{test_name}] y_val (computed): {y_val}")
        print(f"[{test_name}] grad_x2_val (computed): {grad_x2_val}")
        print(f"[{test_name}] grad_x3_val (computed): {grad_x3_val}")

        expected_y_val = x2_val + x3_val
        expected_grad_x2_val = np.ones_like(x2_val)
        expected_grad_x3_val = np.ones_like(x3_val)
        print(f"[{test_name}] y_val (expected): {expected_y_val}")
        print(f"[{test_name}] grad_x2_val (expected): {expected_grad_x2_val}")
        print(f"[{test_name}] grad_x3_val (expected): {expected_grad_x3_val}")

        self.assertIsInstance(y, gw.Node)
        self.assertTrue(np.array_equal(y_val, expected_y_val))
        self.assertTrue(np.array_equal(grad_x2_val, expected_grad_x2_val))
        self.assertTrue(np.array_equal(grad_x3_val, expected_grad_x3_val))
        print(f"--- {test_name} PASSED ---")

    def test_mul_two_vars(self):
        test_name = "test_mul_two_vars"
        print(f"\n--- Running {test_name} ---")
        x2 = gw.Variable(name="x2")
        log_node_details(test_name, "x2", x2)
        x3 = gw.Variable(name="x3")
        log_node_details(test_name, "x3", x3)
        y = x2 * x3
        log_node_details(test_name, "y", y)

        print(f"[{test_name}] Calculating gradients for y w.r.t [x2, x3]...")
        grad_x2_node, grad_x3_node = gw.gradients(y, [x2, x3])
        log_node_details(test_name, "grad_x2_node", grad_x2_node)
        log_node_details(test_name, "grad_x3_node", grad_x3_node)

        print(f"[{test_name}] Creating executor for [y, grad_x2_node, grad_x3_node]...")
        executor = gw.Executor([y, grad_x2_node, grad_x3_node])
        x2_val = 2 * np.ones(3)
        x3_val = 3 * np.ones(3)
        print(f"[{test_name}] x2_val (feed_dict): {x2_val}")
        print(f"[{test_name}] x3_val (feed_dict): {x3_val}")

        print(f"[{test_name}] Running executor...")
        y_val, grad_x2_val, grad_x3_val = executor.run(feed_dict={x2: x2_val, x3: x3_val})
        print(f"[{test_name}] y_val (computed): {y_val}")
        print(f"[{test_name}] grad_x2_val (computed): {grad_x2_val}")
        print(f"[{test_name}] grad_x3_val (computed): {grad_x3_val}")

        expected_y_val = x2_val * x3_val
        expected_grad_x2_val = x3_val
        expected_grad_x3_val = x2_val
        print(f"[{test_name}] y_val (expected): {expected_y_val}")
        print(f"[{test_name}] grad_x2_val (expected): {expected_grad_x2_val}")
        print(f"[{test_name}] grad_x3_val (expected): {expected_grad_x3_val}")

        self.assertIsInstance(y, gw.Node)
        self.assertTrue(np.array_equal(y_val, expected_y_val))
        self.assertTrue(np.array_equal(grad_x2_val, expected_grad_x2_val))
        self.assertTrue(np.array_equal(grad_x3_val, expected_grad_x3_val))
        print(f"--- {test_name} PASSED ---")

    def test_add_mul_mix_1(self):
        test_name = "test_add_mul_mix_1"
        print(f"\n--- Running {test_name} ---")
        x1 = gw.Variable(name="x1")
        log_node_details(test_name, "x1", x1)
        x2 = gw.Variable(name="x2")
        log_node_details(test_name, "x2", x2)
        x3 = gw.Variable(name="x3")
        log_node_details(test_name, "x3", x3)

        # y = x1 + x2 * x3 * x1
        # Graph construction: term_mult = x2 * x3; term_mult_x1 = term_mult * x1; y = x1 + term_mult_x1
        term_mult = x2 * x3
        log_node_details(test_name, "term_mult (x2*x3)", term_mult)
        term_mult_x1 = term_mult * x1
        log_node_details(test_name, "term_mult_x1 ((x2*x3)*x1)", term_mult_x1)
        y = x1 + term_mult_x1
        log_node_details(test_name, "y (x1 + (x2*x3*x1))", y)


        print(f"[{test_name}] Calculating gradients for y w.r.t [x1, x2, x3]...")
        grad_x1_node, grad_x2_node, grad_x3_node = gw.gradients(y, [x1, x2, x3])
        log_node_details(test_name, "grad_x1_node", grad_x1_node)
        log_node_details(test_name, "grad_x2_node", grad_x2_node)
        log_node_details(test_name, "grad_x3_node", grad_x3_node)

        print(f"[{test_name}] Creating executor for [y, grad_x1_node, grad_x2_node, grad_x3_node]...")
        executor = gw.Executor([y, grad_x1_node, grad_x2_node, grad_x3_node])
        x1_val = 1 * np.ones(3)
        x2_val = 2 * np.ones(3)
        x3_val = 3 * np.ones(3)
        print(f"[{test_name}] x1_val (feed_dict): {x1_val}")
        print(f"[{test_name}] x2_val (feed_dict): {x2_val}")
        print(f"[{test_name}] x3_val (feed_dict): {x3_val}")

        print(f"[{test_name}] Running executor...")
        results = executor.run(feed_dict={x1: x1_val, x2: x2_val, x3: x3_val})
        y_val, grad_x1_val, grad_x2_val, grad_x3_val = results
        print(f"[{test_name}] y_val (computed): {y_val}")
        print(f"[{test_name}] grad_x1_val (computed): {grad_x1_val}")
        print(f"[{test_name}] grad_x2_val (computed): {grad_x2_val}")
        print(f"[{test_name}] grad_x3_val (computed): {grad_x3_val}")

        expected_y_val = x1_val + x2_val * x3_val * x1_val
        expected_grad_x1_val = np.ones_like(x1_val) + x2_val * x3_val
        expected_grad_x2_val = x3_val * x1_val
        expected_grad_x3_val = x2_val * x1_val
        print(f"[{test_name}] y_val (expected): {expected_y_val}")
        print(f"[{test_name}] grad_x1_val (expected): {expected_grad_x1_val}")
        print(f"[{test_name}] grad_x2_val (expected): {expected_grad_x2_val}")
        print(f"[{test_name}] grad_x3_val (expected): {expected_grad_x3_val}")

        self.assertIsInstance(y, gw.Node)
        self.assertTrue(np.array_equal(y_val, expected_y_val))
        self.assertTrue(np.array_equal(grad_x1_val, expected_grad_x1_val))
        self.assertTrue(np.array_equal(grad_x2_val, expected_grad_x2_val))
        self.assertTrue(np.array_equal(grad_x3_val, expected_grad_x3_val))
        print(f"--- {test_name} PASSED ---")

    def test_add_mul_mix_2(self):
        test_name = "test_add_mul_mix_2"
        print(f"\n--- Running {test_name} ---")
        x1 = gw.Variable(name="x1")
        log_node_details(test_name, "x1", x1)
        x2 = gw.Variable(name="x2")
        log_node_details(test_name, "x2", x2)
        x3 = gw.Variable(name="x3")
        log_node_details(test_name, "x3", x3)
        x4 = gw.Variable(name="x4")
        log_node_details(test_name, "x4", x4)

        # y = x1 + x2 * x3 * x4
        term_mult_1 = x2 * x3
        log_node_details(test_name, "term_mult_1 (x2*x3)", term_mult_1)
        term_mult_2 = term_mult_1 * x4
        log_node_details(test_name, "term_mult_2 ((x2*x3)*x4)", term_mult_2)
        y = x1 + term_mult_2
        log_node_details(test_name, "y (x1 + (x2*x3*x4))", y)

        print(f"[{test_name}] Calculating gradients for y w.r.t [x1, x2, x3, x4]...")
        grad_x1_node, grad_x2_node, grad_x3_node, grad_x4_node = gw.gradients(y, [x1, x2, x3, x4])
        log_node_details(test_name, "grad_x1_node", grad_x1_node)
        log_node_details(test_name, "grad_x2_node", grad_x2_node)
        log_node_details(test_name, "grad_x3_node", grad_x3_node)
        log_node_details(test_name, "grad_x4_node", grad_x4_node)

        print(f"[{test_name}] Creating executor...")
        executor = gw.Executor([y, grad_x1_node, grad_x2_node, grad_x3_node, grad_x4_node])
        x1_val = 1 * np.ones(3)
        x2_val = 2 * np.ones(3)
        x3_val = 3 * np.ones(3)
        x4_val = 4 * np.ones(3)
        print(f"[{test_name}] x1_val: {x1_val}, x2_val: {x2_val}, x3_val: {x3_val}, x4_val: {x4_val}")

        print(f"[{test_name}] Running executor...")
        results = executor.run(feed_dict={x1: x1_val, x2: x2_val, x3: x3_val, x4: x4_val})
        y_val, grad_x1_val, grad_x2_val, grad_x3_val, grad_x4_val = results
        print(f"[{test_name}] y_val (computed): {y_val}")
        # ... (log other computed grads)

        expected_y_val = x1_val + x2_val * x3_val * x4_val
        expected_grad_x1_val = np.ones_like(x1_val)
        expected_grad_x2_val = x3_val * x4_val
        expected_grad_x3_val = x2_val * x4_val
        expected_grad_x4_val = x2_val * x3_val
        print(f"[{test_name}] y_val (expected): {expected_y_val}")
        # ... (log other expected grads)

        self.assertIsInstance(y, gw.Node)
        self.assertTrue(np.array_equal(y_val, expected_y_val))
        self.assertTrue(np.array_equal(grad_x1_val, expected_grad_x1_val))
        self.assertTrue(np.array_equal(grad_x2_val, expected_grad_x2_val))
        self.assertTrue(np.array_equal(grad_x3_val, expected_grad_x3_val))
        self.assertTrue(np.array_equal(grad_x4_val, expected_grad_x4_val))
        print(f"--- {test_name} PASSED ---")

    def test_add_mul_mix_3(self):
        test_name = "test_add_mul_mix_3"
        print(f"\n--- Running {test_name} ---")
        x2 = gw.Variable(name="x2")
        log_node_details(test_name, "x2", x2)
        x3 = gw.Variable(name="x3")
        log_node_details(test_name, "x3", x3)

        # z = x2 * x2 + x2 + x3 + 3
        # y = z * z + x3
        print(f"[{test_name}] Constructing graph for z = x2*x2 + x2 + x3 + 3")
        x2_sq = x2 * x2
        log_node_details(test_name, "x2_sq (x2*x2)", x2_sq)
        z_term1 = x2_sq + x2
        log_node_details(test_name, "z_term1 (x2*x2+x2)", z_term1)
        z_term2 = z_term1 + x3
        log_node_details(test_name, "z_term2 (x2*x2+x2+x3)", z_term2)
        z = z_term2 + 3 # This uses __add__ which calls add_byconst_op
        log_node_details(test_name, "z (x2*x2+x2+x3+3)", z)

        print(f"[{test_name}] Constructing graph for y = z*z + x3")
        z_sq = z * z
        log_node_details(test_name, "z_sq (z*z)", z_sq)
        y = z_sq + x3
        log_node_details(test_name, "y (z*z+x3)", y)

        print(f"[{test_name}] Calculating gradients for y w.r.t [x2, x3]...")
        grad_x2_node, grad_x3_node = gw.gradients(y, [x2, x3])
        log_node_details(test_name, "grad_x2_node", grad_x2_node)
        log_node_details(test_name, "grad_x3_node", grad_x3_node)

        print(f"[{test_name}] Creating executor...")
        executor = gw.Executor([y, grad_x2_node, grad_x3_node])
        x2_val = 2 * np.ones(3)
        x3_val = 3 * np.ones(3)
        print(f"[{test_name}] x2_val: {x2_val}, x3_val: {x3_val}")

        print(f"[{test_name}] Running executor...")
        y_val, grad_x2_val, grad_x3_val = executor.run(feed_dict={x2: x2_val, x3: x3_val})
        print(f"[{test_name}] y_val (computed): {y_val}")
        print(f"[{test_name}] grad_x2_val (computed): {grad_x2_val}")
        print(f"[{test_name}] grad_x3_val (computed): {grad_x3_val}")

        z_val_calc = x2_val * x2_val + x2_val + x3_val + 3
        expected_yval = z_val_calc * z_val_calc + x3_val
        expected_grad_x2_val = 2 * z_val_calc * (2 * x2_val + 1)
        expected_grad_x3_val = 2 * z_val_calc * 1 + 1
        print(f"[{test_name}] z_val_calc (manual): {z_val_calc}")
        print(f"[{test_name}] y_val (expected): {expected_yval}")
        print(f"[{test_name}] grad_x2_val (expected): {expected_grad_x2_val}")
        print(f"[{test_name}] grad_x3_val (expected): {expected_grad_x3_val}")

        self.assertIsInstance(y, gw.Node)
        self.assertTrue(np.array_equal(y_val, expected_yval))
        self.assertTrue(np.array_equal(grad_x2_val, expected_grad_x2_val))
        self.assertTrue(np.array_equal(grad_x3_val, expected_grad_x3_val))
        print(f"--- {test_name} PASSED ---")

    def test_grad_of_grad(self):
        test_name = "test_grad_of_grad"
        print(f"\n--- Running {test_name} ---")
        x2 = gw.Variable(name="x2")
        log_node_details(test_name, "x2", x2)
        x3 = gw.Variable(name="x3")
        log_node_details(test_name, "x3", x3)

        # y = x2*x2 + x2*x3
        x2_sq = x2 * x2
        log_node_details(test_name, "x2_sq", x2_sq)
        x2_x3 = x2 * x3
        log_node_details(test_name, "x2_x3", x2_x3)
        y = x2_sq + x2_x3
        log_node_details(test_name, "y", y)

        print(f"[{test_name}] Calculating 1st order gradients grad_y_x2, grad_y_x3...")
        grad_y_x2_node, grad_y_x3_node = gw.gradients(y, [x2, x3])
        log_node_details(test_name, "grad_y_x2_node", grad_y_x2_node)
        log_node_details(test_name, "grad_y_x3_node", grad_y_x3_node)

        print(f"[{test_name}] Calculating 2nd order gradients grad_y_x2_x2, grad_y_x2_x3...")
        grad_y_x2_x2_node, grad_y_x2_x3_node = gw.gradients(grad_y_x2_node, [x2, x3])
        log_node_details(test_name, "grad_y_x2_x2_node", grad_y_x2_x2_node)
        log_node_details(test_name, "grad_y_x2_x3_node", grad_y_x2_x3_node)

        print(f"[{test_name}] Creating executor...")
        nodes_to_eval = [y, grad_y_x2_node, grad_y_x3_node, grad_y_x2_x2_node, grad_y_x2_x3_node]
        executor = gw.Executor(nodes_to_eval)
        x2_val = 2 * np.ones(3)
        x3_val = 3 * np.ones(3)
        print(f"[{test_name}] x2_val: {x2_val}, x3_val: {x3_val}")

        print(f"[{test_name}] Running executor...")
        results = executor.run(feed_dict={x2: x2_val, x3: x3_val})
        y_val, grad_y_x2_val, grad_y_x3_val, grad_y_x2_x2_val, grad_y_x2_x3_val = results
        print(f"[{test_name}] y_val (computed): {y_val}")
        print(f"[{test_name}] grad_y_x2_val (computed): {grad_y_x2_val}")
        print(f"[{test_name}] grad_y_x3_val (computed): {grad_y_x3_val}")
        print(f"[{test_name}] grad_y_x2_x2_val (computed): {grad_y_x2_x2_val}")
        print(f"[{test_name}] grad_y_x2_x3_val (computed): {grad_y_x2_x3_val}")

        expected_yval = x2_val * x2_val + x2_val * x3_val
        expected_grad_y_x2_val = 2 * x2_val + x3_val
        expected_grad_y_x3_val = x2_val
        expected_grad_y_x2_x2_val = 2 * np.ones_like(x2_val)
        expected_grad_y_x2_x3_val = 1 * np.ones_like(x3_val)
        print(f"[{test_name}] yval (expected): {expected_yval}")
        # ... (log other expected grads)

        self.assertIsInstance(y, gw.Node)
        self.assertTrue(np.array_equal(y_val, expected_yval))
        self.assertTrue(np.array_equal(grad_y_x2_val, expected_grad_y_x2_val))
        self.assertTrue(np.array_equal(grad_y_x3_val, expected_grad_y_x3_val))
        self.assertTrue(np.array_equal(grad_y_x2_x2_val, expected_grad_y_x2_x2_val))
        self.assertTrue(np.array_equal(grad_y_x2_x3_val, expected_grad_y_x2_x3_val))
        print(f"--- {test_name} PASSED ---")

    def test_matmul_two_vars(self):
        test_name = "test_matmul_two_vars"
        print(f"\n--- Running {test_name} ---")
        x2_node_A = gw.Variable(name="x2_A_matrix") # Matrix A
        log_node_details(test_name, "x2_node_A", x2_node_A)
        x3_node_B = gw.Variable(name="x3_B_matrix") # Matrix B
        log_node_details(test_name, "x3_node_B", x3_node_B)

        y = gw.matmul_op(x2_node_A, x3_node_B)
        log_node_details(test_name, "y (MatMul(A,B))", y)

        print(f"[{test_name}] Calculating gradients for y w.r.t [A, B]...")
        grad_A_node, grad_B_node = gw.gradients(y, [x2_node_A, x3_node_B])
        log_node_details(test_name, "grad_A_node (dL/dA)", grad_A_node)
        log_node_details(test_name, "grad_B_node (dL/dB)", grad_B_node)

        print(f"[{test_name}] Creating executor...")
        executor = gw.Executor([y, grad_A_node, grad_B_node])
        x2_val_A = np.array([[1., 2.], [3., 4.], [5., 6.]]) # Shape (3, 2)
        x3_val_B = np.array([[7., 8., 9.], [10., 11., 12.]]) # Shape (2, 3)
        print(f"[{test_name}] x2_val_A:\n{x2_val_A}")
        print(f"[{test_name}] x3_val_B:\n{x3_val_B}")

        print(f"[{test_name}] Running executor...")
        y_val, grad_A_val, grad_B_val = executor.run(feed_dict={x2_node_A: x2_val_A, x3_node_B: x3_val_B})
        print(f"[{test_name}] y_val (computed):\n{y_val}")
        print(f"[{test_name}] grad_A_val (dL/dA computed):\n{grad_A_val}")
        print(f"[{test_name}] grad_B_val (dL/dB computed):\n{grad_B_val}")

        expected_yval = np.dot(x2_val_A, x3_val_B)
        dL_dY_val = np.ones_like(expected_yval) # Implicit gradient of sum(Y) w.r.t Y
        expected_grad_A_val = np.dot(dL_dY_val, np.transpose(x3_val_B))
        expected_grad_B_val = np.dot(np.transpose(x2_val_A), dL_dY_val)
        print(f"[{test_name}] y_val (expected):\n{expected_yval}")
        print(f"[{test_name}] dL_dY_val (implicit for sum(Y)):\n{dL_dY_val}")
        print(f"[{test_name}] grad_A_val (dL/dA expected):\n{expected_grad_A_val}")
        print(f"[{test_name}] grad_B_val (dL/dB expected):\n{expected_grad_B_val}")

        self.assertIsInstance(y, gw.Node)
        self.assertTrue(np.array_equal(y_val, expected_yval))
        self.assertTrue(np.array_equal(grad_A_val, expected_grad_A_val))
        self.assertTrue(np.array_equal(grad_B_val, expected_grad_B_val))
        print(f"--- {test_name} PASSED ---")

if __name__ == '__main__':
    unittest.main(verbosity=2) # Run with verbosity 2 if script is run directly
