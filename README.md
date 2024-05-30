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

## 세팅방법 (다운받은 프로젝트 폴더 내에서 실행)
0. 가상환경 생성 <code><pre> virtualenv </code></pre> <code><pre> source venv/bin/activate </code></pre>
1. 패키지 설치 <code><pre> pip3 install torch pyworld pysptk h5py numpy kaldi-io librosa scipy scikit-learn fastdtw </code></pre>
2. VCC2018 데이터 다운 https://datashare.is.ed.ac.uk/handle/10283/3061 접속
3. training data for building parallel and non-parallel VC systems released to participants (117.0Mb) 다운로드 후 압축 해제
   <pre><code>	wget https://datashare.ed.ac.uk/bitstream/handle/10283/3061/vcc2018_database_training.zip    </code></pre>
4. evaluation data (source speaker's data) released to participants (31.79Mb) 다운로드 후 압축 해제\n
   <pre><code> wget https://datashare.ed.ac.uk/bitstream/handle/10283/3061/vcc2018_database_evaluation.zip </code></pre>
5. vcc2018_training 폴더를 corpus 폴더에 train이라는 이름으로 저장
6. vcc2018_evaluation 폴더를 corpus 폴더에 test라는 이름으로 저장
7. 데이터 전처리 <code><pre> python3 preprocess/preprocess-vcc2018.py </code></pre>
8. kaldi 설치 - ppg 파일 추출시 필요 (8~11번은 필요 없음)
9. 다시 ppg폴더로 와서 model/timit_sp_ppg_mono 폴더 생성후 아래 구글드라이브 링크의 .zip 파일 다운후 압축해제
10. https://drive.google.com/file/d/1cBmWiQ3GYW9uvrm_AeModCwBT5b-fDVQ/view?usp=sharing 
11. 다시 ppg 폴더로 와서 <code><pre> python3 ppg_vcc2018.py </code></pre> 
12. VAE 학습 -> Voice conversion -> MCD -> CycleVAE 학습 -> Voice conversion -> MCD <code><pre> bash run_all.sh </code></pre>
13. 각 모델별 MCD 및 MSD 값 확인 <code><pre> python3 print_stat.py </code></pre> 
14. Loss graph 출력 <code><pre> bash run_graph.sh </code></pre>

