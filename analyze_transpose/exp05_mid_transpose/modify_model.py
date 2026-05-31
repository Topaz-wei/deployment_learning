"""实验 5: 中间层 Transpose — 在第 20 个 Conv 输出后插入 round-trip Transpose pair。"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from common.model_utils import (
    load_model, save_model, make_transpose_pair,
    insert_nodes_after, print_graph_summary
)

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'yolov8m.onnx')
DST = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'modified_models', 'yolov8m_mid_transpose.onnx')
TARGET_CONV_IDX = 20

def modify():
    model = load_model(SRC)
    graph = model.graph
    print_graph_summary(graph, 'before')
    conv_count = 0
    target = None
    for n in graph.node:
        if n.op_type == 'Conv':
            if conv_count == TARGET_CONV_IDX:
                target = n
                break
            conv_count += 1
    if target is None:
        raise RuntimeError(f'Could not find Conv #{TARGET_CONV_IDX}')
    print(f"  Target Conv: [{conv_count}] {target.name}")
    orig_out = target.output[0]
    t1, t2, _ = make_transpose_pair('mid', orig_out, f'{orig_out}_back',
                                     fwd_perm=[0, 2, 3, 1], rev_perm=[0, 3, 1, 2])
    rewired = insert_nodes_after(graph, orig_out, [t1, t2])
    print(f"  Rewired {rewired} downstream consumers")
    print_graph_summary(graph, 'after')
    save_model(model, DST)

if __name__ == '__main__':
    modify()
