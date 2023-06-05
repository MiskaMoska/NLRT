import os
import onnx
from maptools import OnnxConverter, TileMapper, NocMapper
from maptools.core import NNModelArch, CTG
from maptools import MapPlotter
from acg import ACG
from layout_designer import LayoutDesigner
from routing_designer import RoutingDesigner
from encoding import RoutingPatternCode
from maptype import DLEMethod

# 读取onnx模型，注意你自己的路径
nvcim_root = os.environ.get('NVCIM_HOME')
model_path = os.path.join(nvcim_root, "onnx_models/simp-resnet18.onnx")
model = onnx.load(model_path)

# 创建onnx转换器
oc = OnnxConverter(model, arch=NNModelArch.RESNET)

# 执行转换
oc.run_conversion()

# oc.plot_host_graph() # 查看主机算子图
# oc.plot_device_graph() # 查看设备算子图


# 获得转换得到的设备算子图
og = oc.device_graph

# 创建xbar映射器
tm = TileMapper(og, 256, 256*5)

# 执行映射
tm.run_map()

# 打印映射信息
tm.print_config()

# 获得映射得到的CTG
ctg = tm.ctg

# ctg.plot_ctg()

acg = ACG(6, 8)
ld = LayoutDesigner(ctg, acg, dle=None)

ld.run_layout()
layout = ld.layout_result

rd = RoutingDesigner(ctg, acg, layout)
rd.run_routing()

routing = rd.routing_result
print(routing.max_conflicts)

layout.draw()
routing.draw()

