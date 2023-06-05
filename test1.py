import os
import onnx
from maptools import OnnxConverter, TileMapper, NocMapper
from maptools.core import NNModelArch, CTG
from maptools import MapPlotter
from acg import ACG
from layout_designer import LayoutDesigner
from conflict_analysis import get_conflit_metrics

# 读取onnx模型，注意你自己的路径
nvcim_root = os.environ.get('NVCIM_HOME')
model_path = os.path.join(nvcim_root, "onnx_models/simp-yolo.onnx")
model = onnx.load(model_path)

# 创建onnx转换器
oc = OnnxConverter(model, arch=NNModelArch.YOLO_V3)

# 执行转换
oc.run_conversion()

# oc.plot_host_graph() # 查看主机算子图
# oc.plot_device_graph() # 查看设备算子图


# 获得转换得到的设备算子图
og = oc.device_graph

# 创建xbar映射器
tm = TileMapper(og, 64, 64*5)

# 执行映射
tm.run_map()

# 打印映射信息
tm.print_config()

# 获得映射得到的CTG
ctg = tm.ctg

cnt = 0
total_list = []
best_maxc = 10000
while True:
    # 创建NoC映射器
    nm = NocMapper(ctg,8,9)

    # 执行映射
    nm.run_map()
    totalc, maxc = get_conflit_metrics([nm.cast_paths])
    if maxc < best_maxc:
        best_maxc = maxc
    print(f"cnt: {cnt}\tbest_maxc: {best_maxc}")
    cnt += 1

# for totalc, maxc in total_list:
#     print(f"totalc: {totalc}\tmaxc: {maxc}")

