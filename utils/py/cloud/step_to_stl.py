import cadquery as cq
import os

filename = input("请输入 STEP 文件路径: ")
output_filename = os.path.splitext(os.path.basename(filename))[0] + ".stl"

# 导入 STEP
assy = cq.importers.importStep(filename)

# 导出为 STL
cq.exporters.export(assy, output_filename, tolerance=1.0, angularTolerance=1.0)