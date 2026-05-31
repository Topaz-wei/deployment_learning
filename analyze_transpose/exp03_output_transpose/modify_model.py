"""实验 3: 输出层 Transpose -- 在 output0 之后插入单次 Transpose(perm=[0,2,1])。"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from common.model_utils import (
    load_model, save_model, insert_transpose_at_graph_output, print_graph_summary
)

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'yolov8m.onnx')
DST = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'modified_models', 'yolov8m_output_transpose.onnx')

def modify():
    model = load_model(SRC)
    graph = model.graph
    print_graph_summary(graph, 'before')
    new_out = insert_transpose_at_graph_output(graph, 'output0', perm=[0, 2, 1])
    print(f"  Graph output redirected: output0 -> {new_out}")
    print_graph_summary(graph, 'after')
    save_model(model, DST)


if __name__ == '__main__':
    modify()
