# CycleVAE 实验代码
## 文件夹 test
- corpus: VCC2018 코퍼스
- preprocess: 每个语料库的preprocess代码集(目前只有VCC2018万)
- data: 保存VCC2018 corpus preprocess的结果
- model: 使用各种方法存储VAE模型
- result: 存储转换成各个VAE模型的语音
- calculate: 计算mcd, msd, gv的代码集
- stats: 按型号存储mcd、msd结果
- log: 代码执行时，保存到控制台输出的部分文本
## 文件
- model.py: 基于PyTorch的VAE模型
- train_base.py: VAE1 VAE2 VAE3 MD 学习
- train_vae3_next.py: 以vae3模型为基础，应用各种方法进行学习
- train_md_next.py: 以md模型为基础，应用各种方法学习
- convert.py: 将语音转换成已学习的VAE
- print_stat.py: 计算过的mcd, msd的统计值输出
- loss.py:  loss计算和log输出相关的模块
- speech_tools.py: 其他必要的函数
- run.sh: 实验进行脚本

## 设置方法(在下载的项目文件夹内运行)
0. 创建虚拟环境 <code><pre> virtualenv </code></pre> <code><pre> source venv/bin/activate </code></pre>
1. 安装包 <code><pre> pip3 install torch pyworld pysptk h5py numpy kaldi-io librosa scipy scikit-learn fastdtw </code></pre>
2. 下载VCC2018数据 https://datashare.is.ed.ac.uk/handle/10283/3061 连接
3. training data for building parallel and non-parallel VC systems released to participants (117.0Mb) 下载后解压
   <pre><code>	wget https://datashare.ed.ac.uk/bitstream/handle/10283/3061/vcc2018_database_training.zip    </code></pre>
4. evaluation data (source speaker's data) released to participants (31.79Mb) 下载后解除压缩\n
   <pre><code> wget https://datashare.ed.ac.uk/bitstream/handle/10283/3061/vcc2018_database_evaluation.zip </code></pre>
5. 将vcc2018_training文件夹以train的名义存储在corpus文件夹中
6. 将vcc2018_evaluation文件夹以test的名称保存在corpus文件夹中
7. 数据预处理 <code><pre> python3 preprocess/preprocess-vcc2018.py </code></pre>
8. kaldi 安装 - ppg 文件提取需要 (8~11不需要编号)
9. 再回到 ppg文件夹，生成 model/timit_sp_ppg_mono 创建文件夹后，下面的google drive链接 .zip 下载文件后解除压缩
10. https://drive.google.com/file/d/1cBmWiQ3GYW9uvrm_AeModCwBT5b-fDVQ/view?usp=sharing 
11. 再次进入 ppg 文件夹 里 <code><pre> python3 ppg_vcc2018.py </code></pre> 
——语音转换（使用了PyTorch框架和Kaldi工具包）\n
      处理训练集和开发集的数据，并生成所需的PPG特征文件。\n
主要目的：加载 一个预训练的BiGRU-HMM（双向门控循环单元-隐马尔可夫模型）网络，\n
          并 将该网络应用于提取训练集和开发集数据的特征，\n
          然后 生成Pseudo-Periodic-Gains（伪周期增益，简称PPG），这是一种语音特征表示，常用于语音转换任务。
13. VAE 学习 -> Voice conversion -> MCD -> CycleVAE 学习 -> Voice conversion -> MCD <code><pre> bash run_all.sh </code></pre>
14. 确认每个型号的MCD和MSD值 <code><pre> python3 print_stat.py </code></pre> 
15. Loss graph 输出 <code><pre> bash run_graph.sh </code></pre>

