"""实验 4: 输入+输出双 Transpose。"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from common.model_utils import (
    load_model, save_model, make_transpose_pair,
    insert_nodes_at_graph_input, insert_transpose_at_graph_output,
    print_graph_summary
)

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'yolov8m.onnx')
DST = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'modified_models', 'yolov8m_dual_transpose.onnx')

def modify():
    model = load_model(SRC)
    graph = model.graph
    print_graph_summary(graph, 'before')
    t1, t2, _ = make_transpose_pair('input', 'images', 'images_transposed_back',
                                     fwd_perm=[0, 2, 3, 1], rev_perm=[0, 3, 1, 2])
    insert_nodes_at_graph_input(graph, 'images', [t1, t2])
    insert_transpose_at_graph_output(graph, 'output0', perm=[0, 2, 1], transpose_name='output_dual_t')
    print_graph_summary(graph, 'after')
    save_model(model, DST)

if __name__ == '__main__':
    modify()
