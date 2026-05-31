"""ONNX 图操作工具: 加载、修改、保存模型。"""
import onnx
from onnx import helper, TensorProto
import os

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODIFIED_DIR = os.path.join(HERE, 'modified_models')
os.makedirs(MODIFIED_DIR, exist_ok=True)


def load_model(path):
    """加载 ONNX 模型"""
    return onnx.load(path)


def save_model(model, path, check=False):
    """保存 ONNX 模型，可选 onnx.checker 验证"""
    if check:
        onnx.checker.check_model(model)
    onnx.save(model, path)
    print(f"  [save] {path}")
    return path


def make_transpose_node(name, input_name, output_name, perm):
    """创建 Transpose 节点"""
    return helper.make_node('Transpose', [input_name], [output_name], name=name, perm=perm)


def make_transpose_pair(base_name, input_name, output_name, fwd_perm, rev_perm):
    """创建 round-trip Transpose 节点对。返回 (t1, t2, intermediate_name)"""
    mid_name = f'{input_name}_{base_name}_mid'
    t1 = make_transpose_node(f'{base_name}_fwd', input_name, mid_name, fwd_perm)
    t2 = make_transpose_node(f'{base_name}_rev', mid_name, output_name, rev_perm)
    return t1, t2, mid_name


def insert_nodes_after(graph, target_node_output_name, new_nodes):
    """在产生 target_node_output_name 的节点之后插入 new_nodes，
    并将所有下游消费者的对应输入重定向到 new_nodes 最后一个的输出。"""
    new_output = new_nodes[-1].output[0]
    new_names = {n.name for n in new_nodes}
    rewired = 0
    for n in graph.node:
        for i, inp in enumerate(n.input):
            if inp == target_node_output_name and n.name not in new_names:
                n.input[i] = new_output
                rewired += 1
    # 找到产生 target_node_output_name 的节点
    insert_pos = 0
    for idx, n in enumerate(graph.node):
        if target_node_output_name in n.output:
            insert_pos = idx
            break
    for n in reversed(new_nodes):
        graph.node.insert(insert_pos + 1, n)
    return rewired


def insert_nodes_at_graph_input(graph, input_name, new_nodes):
    """在 graph 输入 input_name 之后插入 new_nodes，
    将原 consumer 的输入重定向到 new_nodes 最后一个的输出。"""
    new_output = new_nodes[-1].output[0]
    new_names = {n.name for n in new_nodes}
    rewired = 0
    for n in graph.node:
        for i, inp in enumerate(n.input):
            if inp == input_name and n.name not in new_names:
                n.input[i] = new_output
                rewired += 1
    for n in reversed(new_nodes):
        graph.node.insert(0, n)
    return rewired


def insert_transpose_at_graph_output(graph, output_name, perm, transpose_name='output_transpose'):
    """在 graph 输出 output_name 之后插入单次 Transpose 并重定向 graph.output。
    仅适用于输出为 3D tensor 的场景。"""
    new_output = f'{output_name}_transposed'
    t_node = make_transpose_node(transpose_name, output_name, new_output, perm)
    graph.node.append(t_node)
    for out in graph.output:
        if out.name == output_name:
            graph.output.remove(out)
            break
    graph.output.append(helper.make_tensor_value_info(new_output, TensorProto.FLOAT, None))
    return new_output


def find_node_by_name(graph, name):
    """按名称查找节点，返回 (index, node) 或 (None, None)"""
    for i, n in enumerate(graph.node):
        if n.name == name:
            return i, n
    return None, None


def find_nodes_by_op_type(graph, op_type):
    """返回所有匹配 op_type 的 (index, node) 列表"""
    return [(i, n) for i, n in enumerate(graph.node) if n.op_type == op_type]


def count_nodes(graph):
    """统计节点数（按类型）"""
    from collections import Counter
    cnt = Counter(n.op_type for n in graph.node)
    return {'total': len(graph.node), 'by_type': dict(cnt)}


def print_graph_summary(graph, label=''):
    """打印图摘要"""
    counts = count_nodes(graph)
    print(f"  [{label}] 总节点: {counts['total']}")
    transpose_count = counts['by_type'].get('Transpose', 0)
    conv_count = counts['by_type'].get('Conv', 0)
    print(f"  [{label}] Conv: {conv_count}, Transpose: {transpose_count}")
