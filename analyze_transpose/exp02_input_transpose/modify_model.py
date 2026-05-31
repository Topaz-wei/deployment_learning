"""实验 2: 输入层 Transpose — 在 images 后插入 NCHW→NHWC→NCHW round-trip Transpose pair。"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from common.model_utils import (
    load_model, save_model, make_transpose_pair,
    insert_nodes_at_graph_input, print_graph_summary
)

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'yolov8m.onnx')
DST = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'modified_models', 'yolov8m_input_transpose.onnx')

def modify():
    model = load_model(SRC)
    graph = model.graph
    print_graph_summary(graph, 'before')

    t1, t2, _ = make_transpose_pair(
        'input', 'images', 'images_transposed_back',
        fwd_perm=[0, 2, 3, 1],   # NCHW → NHWC
        rev_perm=[0, 3, 1, 2],   # NHWC → NCHW
    )
    insert_nodes_at_graph_input(graph, 'images', [t1, t2])

    print_graph_summary(graph, 'after')
    save_model(model, DST)
    print(f"  Note: round-trip pair — 单次 Transpose 开销 = (delta from baseline) / 2")


if __name__ == '__main__':
    modify()
