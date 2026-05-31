"""实验 6: 多层 Transpose 累积 — 在 Backbone 4 个 Stage 输出后各插入 round-trip Transpose pair。"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from common.model_utils import (
    load_model, save_model, make_transpose_pair,
    insert_nodes_after, print_graph_summary
)

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'yolov8m.onnx')
DST = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'modified_models', 'yolov8m_multi_transpose.onnx')

STAGE_CONV_NAMES = [
    '/model.0/conv/Conv',
    '/model.2/cv2/conv/Conv',
    '/model.4/cv2/conv/Conv',
    '/model.6/cv2/conv/Conv',
]

def modify():
    model = load_model(SRC)
    graph = model.graph
    print_graph_summary(graph, 'before')
    name_to_node = {n.name: n for n in graph.node}
    total_inserted = 0
    for i, name in enumerate(STAGE_CONV_NAMES):
        target = name_to_node.get(name)
        if target is None:
            print(f"  WARNING: node {name} not found, skipping")
            continue
        orig_out = target.output[0]
        t1, t2, _ = make_transpose_pair(f'stage{i}', orig_out, f'{orig_out}_back',
                                         fwd_perm=[0, 2, 3, 1], rev_perm=[0, 3, 1, 2])
        rewired = insert_nodes_after(graph, orig_out, [t1, t2])
        print(f"  Stage {i} [{name}]: rewired {rewired} consumers")
        total_inserted += 2
    print(f"  Total Transpose nodes inserted: {total_inserted}")
    print_graph_summary(graph, 'after')
    save_model(model, DST)

if __name__ == '__main__':
    modify()
