"""实验 8: Transpose 与 Conv 融合边界 — 微模型测试。"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np
import onnx
from onnx import helper, TensorProto
import onnxruntime as ort
from common.benchmark import save_results

HERE = os.path.dirname(os.path.abspath(__file__))
WARMUP, ITERS = 10, 100

def make_simple_model(name, nodes_fn, input_shape=(1, 64, 32, 32)):
    C_in, C_out = input_shape[1], 64
    inputs = [helper.make_tensor_value_info('input', TensorProto.FLOAT, input_shape)]
    outputs = [helper.make_tensor_value_info('output', TensorProto.FLOAT, None)]
    weight = np.random.randn(C_out, C_in, 3, 3).astype(np.float32)
    bias = np.random.randn(C_out).astype(np.float32)
    init = [
        helper.make_tensor('w', TensorProto.FLOAT, weight.shape, weight.tobytes(), raw=True),
        helper.make_tensor('b', TensorProto.FLOAT, bias.shape, bias.tobytes(), raw=True),
    ]
    nodes = nodes_fn()
    graph = helper.make_graph(nodes, name, inputs, outputs, init)
    return helper.make_model(graph, opset_imports=[helper.make_opsetid('', 12)])

def run_benchmark(model, model_name, opt_level, opt_name):
    path = f'/tmp/exp08_{model_name}_{opt_name}.onnx'
    onnx.save(model, path)
    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = opt_level
    sess = ort.InferenceSession(path, sess_opts, providers=['CPUExecutionProvider'])
    inp = np.random.randn(1, 64, 32, 32).astype(np.float32)
    for _ in range(WARMUP): sess.run(None, {'input': inp})
    times = []
    for _ in range(ITERS):
        t0 = time.time(); sess.run(None, {'input': inp}); times.append((time.time()-t0)*1000)
    times = np.array(times)
    loaded = onnx.load(path)
    t_count = sum(1 for n in loaded.graph.node if n.op_type == 'Transpose')
    c_count = sum(1 for n in loaded.graph.node if n.op_type == 'Conv')
    return {
        'model': model_name, 'optimization': opt_name,
        'nodes_total': len(loaded.graph.node),
        'transpose_nodes': t_count, 'conv_nodes': c_count,
        'mean_ms': float(times.mean()), 'fps': float(1000.0/times.mean()),
    }

def build_conv_only():
    return [helper.make_node('Conv', ['input', 'w', 'b'], ['output'], name='conv1', kernel_shape=[3, 3])]

def build_t_conv_t():
    return [
        helper.make_node('Transpose', ['input'], ['t1_out'], name='t1', perm=[0, 2, 3, 1]),
        helper.make_node('Transpose', ['t1_out'], ['t2_out'], name='t2', perm=[0, 3, 1, 2]),
        helper.make_node('Conv', ['t2_out', 'w', 'b'], ['c_out'], name='conv1', kernel_shape=[3, 3]),
        helper.make_node('Transpose', ['c_out'], ['t3_out'], name='t3', perm=[0, 2, 3, 1]),
        helper.make_node('Transpose', ['t3_out'], ['output'], name='t4', perm=[0, 3, 1, 2]),
    ]

def run():
    print("=" * 60)
    print("实验 8: Transpose 与 Conv 融合边界")
    print("=" * 60)

    models = [
        ('Conv_Only', build_conv_only),
        ('T_Conv_T', build_t_conv_t),
    ]
    opt_levels = [
        ('DISABLE_ALL', ort.GraphOptimizationLevel.ORT_DISABLE_ALL),
        ('ALL', ort.GraphOptimizationLevel.ORT_ENABLE_ALL),
    ]
    results = []

    for model_name, nodes_fn in models:
        model = make_simple_model(model_name, nodes_fn)
        for opt_name, opt_val in opt_levels:
            r = run_benchmark(model, model_name, opt_val, opt_name)
            results.append(r)
            print(f"  {r['model']:<15} {r['optimization']:<14} nodes={r['nodes_total']} T={r['transpose_nodes']} Conv={r['conv_nodes']} mean={r['mean_ms']:.4f}ms")

    all_results = {'experiment': 'exp08_transpose_conv_fusion', 'results': results}
    save_results(all_results, 'exp08_transpose_conv_fusion.json')

    print("\n--- 融合分析 ---")
    for model_name, _ in models:
        base = [r for r in results if r['model'] == model_name and r['optimization'] == 'DISABLE_ALL'][0]
        opt = [r for r in results if r['model'] == model_name and r['optimization'] == 'ALL'][0]
        nodes_delta = base['nodes_total'] - opt['nodes_total']
        print(f"  {model_name}: DISABLE_ALL->ALL 节点减 {nodes_delta} (T: {base['transpose_nodes']}->{opt['transpose_nodes']})")
    print("\n实验 8 完成")

if __name__ == '__main__':
    run()
