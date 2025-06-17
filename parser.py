import re
import numpy as np
import networkx as nx
import json
from scipy.sparse import lil_matrix, save_npz
from collections import deque
import os

# ===== Step 0: BFS Level 計算 =====
# Loretta： BFS
def compute_bfs_levels(adj, start_nodes):
    num_nodes = adj.shape[0]
    levels = [-1] * num_nodes  # -1 表示沒被訪問過
    queue = deque()

    for s in start_nodes:
        levels[s] = 0
        queue.append(s)

    while queue:
        node = queue.popleft()
        for neighbor in adj.rows[node]:  # lil_matrix 的 row 可直接抓 list of neighbors
            if levels[neighbor] == -1:
                levels[neighbor] = levels[node] + 1
                queue.append(neighbor)
    return levels

def compute_reverse_bfs_levels(adj, start_nodes):
    adj = adj.T.tolil()  # ✅ 強制轉置後仍是 LIL，才能用 .rows[node]
    num_nodes = adj.shape[0]
    levels = [-1] * num_nodes
    queue = deque()

    for s in start_nodes:
        levels[s] = 0
        queue.append(s)

    while queue:
        node = queue.popleft()
        for pred in adj.rows[node]:  # ✅ 用轉好的 adj
            if levels[pred] == -1:
                levels[pred] = levels[node] + 1
                queue.append(pred)
    return levels


# ===== Step 1: 解析 Verilog 檔案 =====
def parse_verilog(verilog_code):
    gates = []
    primary_inputs = ["1'b1","1'b0"]
    primary_outputs = []
    for line in verilog_code.splitlines():
        line = line.strip()
        #print(f"Processing line: {line}")

        if line.startswith('input'):
            line = line.strip().rstrip(';')
            line = line[len('input'):].strip()

            vector_range = None

            # Check for vector declaration like [11:0]
            if line.startswith('['):
                closing_bracket_index = line.find(']')
                vector_range_str = line[1:closing_bracket_index]
                msb, lsb = map(int, vector_range_str.split(':'))
                vector_range = range(lsb, msb + 1) if lsb <= msb else range(lsb, msb - 1, -1)
                line = line[closing_bracket_index + 1:].strip()

            # Split remaining part into signal names
            signals = [name.strip() for name in line.split(',')]

            for sig in signals:
                if vector_range:
                    primary_inputs.extend([f"{sig}[{i}]" for i in vector_range])
                else:
                    primary_inputs.append(sig)

        # check if line starts with output, such as: output n6, n7;
        # or such as: output [11:0] n8;
        # then n6, n7, n8 are primary outputs
        if line.startswith('output'):
            line = line.strip().rstrip(';')
            line = line[len('output'):].strip()

            vector_range = None
            # Check for vector declaration like [11:0]
            if line.startswith('['):
                closing_bracket_index = line.find(']')
                vector_range_str = line[1:closing_bracket_index]
                msb, lsb = map(int, vector_range_str.split(':'))
                vector_range = range(lsb, msb + 1) if lsb <= msb else range(lsb, msb - 1, -1)
                line = line[closing_bracket_index + 1:].strip()
            # Split remaining part into signal names
            signals = [name.strip() for name in line.split(',')]
            for sig in signals:
                if vector_range:
                    primary_outputs.extend([f"{sig}[{i}]" for i in vector_range])
                else:
                    primary_outputs.append(sig)

        # 解析 BUF gate
        buf_match = re.match(r'^\s*buf\s+(\S+)\((\S+),\s*(\S+)\);', line)
        if buf_match:
            gates.append(('BUF', buf_match.group(1), buf_match.group(2), buf_match.group(3)))  # buf, output, input
            continue

        # 解析 DFF gate（具名端口的 DFF gate，處理 .RN, .SN 等）
        dff_match = re.match(r'^\s*dff\s+(\S+)\s*\(\.RN\(([^)]+)\),\s*\.SN\(([^)]+)\),\s*\.CK\(([^)]+)\),\s*\.D\(([^)]+)\),\s*\.Q\(([^)]+)\)\);', line)
        if dff_match:
            gates.append(('DFF', dff_match.group(1), dff_match.group(6), dff_match.group(3),
                          dff_match.group(4), dff_match.group(5), dff_match.group(2)))  # dff, RN, SN, CK, D, Q
            continue

        # 解析 OR gate (雙輸入 gate)
        or_match = re.match(r'^\s*or\s+(\S+)\((\S+)\s*,\s*(\S+)\s*,\s*(\S+)\);', line)
        if or_match:
            gates.append(('OR', or_match.group(1), or_match.group(2), or_match.group(3), or_match.group(4)))  # or, output, input1, input2
            continue

        # 解析 NOR gate (雙輸入 gate)
        nor_match = re.match(r'^\s*nor\s+(\S+)\((\S+)\s*,\s*(\S+)\s*,\s*(\S+)\);', line)
        if nor_match:
            gates.append(('NOR', nor_match.group(1), nor_match.group(2), nor_match.group(3), nor_match.group(4)))  # nor, output, input1, input2
            continue

        # 解析 NOT gate (單輸入 gate)
        not_match = re.match(r'^\s*not\s+(\S+)\((\S+)\s*,\s*(\S+)\);', line)
        if not_match:
            gates.append(('NOT', not_match.group(1), not_match.group(2), not_match.group(3)))  # not, output, input
            continue

        # 解析 XOR gate (雙輸入 gate)
        xor_match = re.match(r'^\s*xor\s+(\S+)\((\S+)\s*,\s*(\S+)\s*,\s*(\S+)\);', line)
        if xor_match:
            gates.append(('XOR', xor_match.group(1), xor_match.group(2), xor_match.group(3), xor_match.group(4)))  # xor, output, input1, input2
            continue

        # 解析 AND gate (雙輸入 gate)
        and_match = re.match(r'^\s*and\s+(\S+)\((\S+)\s*,\s*(\S+)\s*,\s*(\S+)\);', line)
        if and_match:
            gates.append(('AND', and_match.group(1), and_match.group(2), and_match.group(3), and_match.group(4)))  # and, output, input1, input2
            continue

        # 解析 NAND gate (雙輸入 gate)
        nand_match = re.match(r'^\s*nand\s+(\S+)\((\S+)\s*,\s*(\S+)\s*,\s*(\S+)\);', line)
        if nand_match:
            gates.append(('NAND', nand_match.group(1), nand_match.group(2), nand_match.group(3), nand_match.group(4)))  # nand, output, input1, input2
            continue

        # 解析 XNOR gate (雙輸入 gate)
        xnor_match = re.match(r'^\s*xnor\s+(\S+)\((\S+)\s*,\s*(\S+)\s*,\s*(\S+)\);', line)
        if xnor_match:
            gates.append(('XNOR', xnor_match.group(1), xnor_match.group(2), xnor_match.group(3), xnor_match.group(4)))  # xor, output, input1, input2
            continue
    #print(gates)
    return gates, primary_inputs, primary_outputs
# ===== Step 2: 轉換成 infolist 格式 =====
def gates_to_infolist(gates, trojan_gates=[]):
    infolist = []
    for g in gates:
        gtype = g[0]
        instname = g[1]
        output = g[2]
        inputs = list(g[3:])

        portnames = ['Y'] + [f'A{i+1}' for i in range(len(inputs))]
        connnames = [output] + inputs

        is_trojan = (instname in trojan_gates or output in trojan_gates or any(inp in trojan_gates for inp in inputs))
        infolist.append((
            gtype, gtype, instname, instname, portnames, connnames, is_trojan
        ))
        #print("connnames = ",connnames)
    return infolist

# ===== Step 3: 建立 adjacency matrix & features =====


def build_lookup(infolist, output_nodes=None):
    lookup = {}
    for i, info in enumerate(infolist):
        conns = info[5]  # [output, input1, input2, ...]

        # 處理 inputs → gate 的映射
        for conn in conns[1:]:
            if conn not in lookup:
                lookup[conn] = []
            lookup[conn].append(i)

        # ✅ 額外處理：如果 output wire 是 primary output，就也放進 lookup
        if output_nodes is not None and conns[0] in output_nodes:
            if conns[0] not in lookup:
                lookup[conns[0]] = []
            lookup[conns[0]].append(i)

    return lookup


def build_graph_features(infolist, primary_inputs=None, primary_outputs=None):
    numnodes = len(infolist) + len(primary_inputs) + len(primary_outputs)
    adj = lil_matrix((numnodes, numnodes), dtype=bool)
    class_map = {}
    train_indices = list(range(numnodes))  # 全部都當 train

    gatelist = sorted(list(set([x[0] for x in infolist])))
    gatelookup = {g: i for i, g in enumerate(gatelist)}

    # feature: one-hot + in degree + out degree + bfs detection?(Loretta :所以我把他變成+3喔)
    feats = np.zeros((numnodes, len(gatelist) + 4))
    gate_map={}
    lookup = build_lookup(infolist, primary_outputs)

    for i, info in enumerate(infolist):
        gatetype = info[0]
        conns = info[5]
        feats[i][gatelookup[gatetype]] = 1

        # Loretta
        output_wire = conns[0]  # output wire

        if output_wire in lookup:
            for j in lookup[output_wire]:
                if i != j:
                    adj[i, j] = True
                    feats[i][-1] += 1  # out degree
                    feats[j][-2] += 1  # in degree

        class_map[i] = 1 if info[6] else 0
        gate_map[i] = info[2]

    for i, signal in enumerate(primary_inputs):
        node_index = i + len(infolist)  # primary input 的 node index
        if signal not in lookup:
            class_map[node_index] = 0
            gate_map[node_index] = signal
            continue
        for j in lookup[signal]:  # 所有吃這個 signal 的 gate
            adj[node_index, j] = True  # PI ➜ gate
            feats[node_index][-1] += 1  # PI 的 out degree
            feats[j][-2] += 1  # gate 的 in degree
        class_map[node_index] = 0
        gate_map[node_index] = signal

    for i, signal in enumerate(primary_outputs):
        node_index = i + len(infolist) + len(primary_inputs)  # output node 的 index
        for j, info in enumerate(infolist):
            output_wire = info[5][0]
            if output_wire == signal:
                adj[j, node_index] = True  # ✅ gate ➜ primary output
                feats[j][-1] += 1  # gate 的 out degree
                feats[node_index][-2] += 1  # PO node 的 in degree
                class_map[node_index] = 0
                gate_map[node_index] = signal

    # Loretta(BFS Level 計算)
    input_nodes = list(range(len(infolist), len(infolist) + len(primary_inputs)))
    output_nodes = list(range(len(infolist) + len(primary_inputs), numnodes))
    bfs_levels = compute_bfs_levels(adj, input_nodes)
    reverse_levels = compute_reverse_bfs_levels(adj, output_nodes)
    # print(bfs_levels)
    # print(reverse_levels)
    for i in range(len(feats)):
        feats[i][-3] = bfs_levels[i]
    for i in range(len(feats)):
        feats[i][-4] = reverse_levels[i]

    return adj, feats, train_indices, class_map, gate_map

# ===== Step 4: 儲存 GraphSAGE 所需格式 =====
def save_graphsage_format(adj, feats, class_map, train_indices, gate_map, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    save_npz(f"{output_dir}/adj_full.npz", adj.tocsr())  # 儲存完整的 adjacency matrix
    save_npz(f"{output_dir}/adj_train.npz", adj.tocsr())  # 訓練用的 adjacency matrix, 此處與完整的相同

    np.savez(f"{output_dir}/feat_full.npz", feats=feats, allow_pickle=False)  # 儲存完整的 features

    with open(f"{output_dir}/class_map.json", "w") as f:
        json.dump(class_map, f)

    with open(f"{output_dir}/role.json", "w") as f:
        json.dump({'tr': train_indices, 'va': [], 'te': []}, f)

    with open(f"{output_dir}/gate_map.json", "w") as f:  # <== 新增這段
        json.dump(gate_map, f)

# ===== 主流程 =====
def process_single_verilog(filepath, gt_trojan_filepath=None):
    with open(filepath, 'r') as f:
        code = f.read()

    gates, primary_inputs, primary_outputs = parse_verilog(code)
    print(f"Parsed {len(gates)} gates, {len(primary_inputs)} primary inputs, {len(primary_outputs)} primary outputs.")
    # 若無trojan gates, txt只有一行: NO_TROJAN
    # 若有trojan gates, txt第一行是 "TROJANED", 第二行是 "TROJAN_GATES", 最後一行是 "END_TROJAN_GATES"
    trojan_gates = []
    if gt_trojan_filepath and os.path.exists(gt_trojan_filepath):
        with open(gt_trojan_filepath, 'r') as f:
            lines = [l.strip() for l in f]
            if lines and lines[0] == "TROJANED":
                for line in lines[2:]:
                    if line == "END_TROJAN_GATES":
                        break
                    trojan_gates.append(line)
            else:
                trojan_gates = []
    infolist = gates_to_infolist(gates, trojan_gates)
    adj, feats, train_indices, class_map, gate_map = build_graph_features(infolist, primary_inputs, primary_outputs)
    output_dir = "./output/" + os.path.splitext(os.path.basename(filepath))[0]
    save_graphsage_format(adj, feats, class_map, train_indices, gate_map, output_dir)

    print("✅ Graph feature files saved.")

if __name__ == "__main__":
    # design_folder = ./process_data/
    # gt_trojan_folder = ./gt_data/
    # process through all design files in the design_folder
    for i in range(30):
        design_file = f"./process_data/design{i}.v"
        gt_trojan_file = f"./gt_data/result{i}.txt"
        if os.path.exists(design_file):
            print(f"Processing {design_file} with ground truth {gt_trojan_file}...")
            process_single_verilog(design_file, gt_trojan_file)
        else:
            print(f"Design file {design_file} does not exist, skipping.")
