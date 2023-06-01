import os
import onnx
from maptools import OnnxConverter, TileMapper, NocMapper
from maptools.core import NNModelArch, CTG
from maptools import MapPlotter
from acg import ACG
from layout_designer import LayoutDesigner

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

for i, cluster in ctg.clusters:
    print(i, cluster)

acg = ACG(7, 7)
ld = LayoutDesigner(ctg, acg)

while True:
    ld.init_layout()
    ld.run_layout()
    if ld.is_patches(ld.hotmap):
        ld.plot_result()
        pass
    else:
        ld.plot_result()
        # break
