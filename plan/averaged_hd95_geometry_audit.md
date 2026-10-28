# averaged目录的物理HD95几何核验

扫描238份NIfTI：83训练图像、83训练标签、72测试图像。三份下载manifest数量均一致，全部文件MD5匹配=True，空间单位均为mm=True。有spacing不等于可用于当前Synapse病例。

83份标签全部仅含0/1；体积最短轴8至17层。当前12例H5为85至198层、每层512×512，采用0..8多器官标签（case0035没有标签2）。所有12例在83份原标签中，任何轴排列/翻转下的shape候选数均为0，不能进入逐体素匹配。

例如DET0000101_avg_seg.nii.gz：XYZ=192×156×12，spacing XYZ=(1.5625,1.5625,7.0)mm，只有标签0/1；case0001.npy.h5：ZYX=147×512×512，标签0..8。没有可信的DET-to-case对应表，不能按相似数字配对，或将某例/平均spacing套到所有病例。

结论：averaged不能提供本实验可验证的物理几何。未生成priors/synapse_spacing.verified.json，未更改指标代码或原体素HD95，未启动毫米重评。未确定DET文件具体模态/器官，不作猜测。下载完整性已通过，问题是数据对应关系和几何不匹配。

需要当前腹部CT病例对应的原始NIfTI，优先含label0001.nii.gz等标签，覆盖case0001,0002,0003,0004,0008,0022,0025,0029,0032,0035,0036,0038。核验13类到8类映射及轴序/翻转；若裁剪/重采样还需几何记录。通过后可重评已有checkpoint，无需重训，原Dice规则不变。

官方入口已核对：Abdomen https://www.synapse.org/Synapse:syn3376386 下列出RawData.zip https://www.synapse.org/Synapse:syn3379050 。本批averaged-training-images的manifest parentId是syn10285054。不能以同在Synapse平台推断是同一数据。

ASEE已结束，exit_code=0，220轮，171至220共50轮/600病例测试齐全，最佳196轮mean Dice=85.447484%。原成绩与checkpoint保留；本轮未转换HD95。

证据目录：outputs/audit/averaged_geometry_20261023T015002Z；geometry_audit.json保存逐文件MD5、来源、尺寸、spacing、标签及12例比较；completed_asee_run.json保存训练退出状态与测试完整性。
