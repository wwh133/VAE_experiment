# CycleVAE 실험 코드
## 폴더 test
- corpus: VCC2018 코퍼스
- preprocess: 각 코퍼스 별 preprocess 코드 모음 (현재는 VCC2018만 존재)
- data: VCC2018 코퍼스 preprocess된 결과 저장
- model: 각 방법을 적용한 VAE 모델 저장
- result: 각 VAE 모델로 변환한 음성 저장
- calculate: mcd, msd, gv 계산하는 코드 모음
- stats: 모델별 mcd, msd 결과 저장
- log: 코드 실행시 콘솔에 출력되는 부분 텍스트로 저장
## 파일
- model.py: PyTorch 기반 VAE 모델
- train_base.py: VAE1 VAE2 VAE3 MD 학습
- train_vae3_next.py: VAE3 모델 기반으로 각 방법 적용하여 학습
- train_md_next.py: MD 모델 기반으로 각 방법 적용하여 학습
- convert.py: 학습된 VAE로 음성 변환
- print_stat.py: 계산된 mcd, msd의 통계값 출력
- loss.py: loss 계산 및 log 출력과 관련된 모듈
- speech_tools.py: 기타 필요한 함수
- run.sh: 실험 진행 스크립트

## 세팅방법
1. pip3 install torch pyworld pysptk h5py numpy kaldi-io librosa scipy scikit-learn fastdtw 설치
2. https://datashare.is.ed.ac.uk/handle/10283/3061 접속
3. training data for building parallel and non-parallel VC systems released to participants (117.0Mb) 다운로드 후 압축 해제
4. evaluation data (source speaker's data) released to participants (31.79Mb) 다운로드 후 압축 해제
5. vcc2018_training 폴더를 corpus 폴더에 train이라는 이름으로 저장
6. vcc2018_evaluation 폴더를 corpus 폴더에 test라는 이름으로 저장
7. python3 preprocess/preprocess-vcc2018.py 실행
8. kaldi 설치 - ppg 파일 추출시 필요 (연구실 서버에는 설치 되어있음, 개인 pc 에서 작업시 설치) 
9. 다시 ppg폴더로 와서 model/timit_sp_ppg_mono 폴더 생성후 아래 구글드라이브 링크의 .zip 파일 다운후 압축해제
10. https://drive.google.com/file/d/1cBmWiQ3GYW9uvrm_AeModCwBT5b-fDVQ/view?usp=sharing 
11. 다시 ppg 폴더로 와서 python3 ppg_vcc2018.py 실행 
12. bash run_all.sh 실행 (nohup 실행 권장)
13. python3 print_stat.py 실행하여 각 모델별 MCD 및 MSD 값 확인

## 해결해야 될 문제
1. VAE3 MCD 수치가 너무 높게 나옴 (VAE1, VAE2에 비해)
    - 교수님 조언에 따라 화자 벡터 파라미터 줄이는 시도 중
2. 논문의 제안방법인 여러 loss 병합 모델을 적용하였을 때 MCD 수치가 크게 달라지지 않음
    - 코드 잘못된 부분 확인 + coef 값 조정(한가지 loss 만 선택하여 체계적으로) e.g. 한번에 SI, I의 coef 수정해서 실험X
    
## coefficient 실험 lambda 범위  
- 1.0 / 3.0 / 5.0 / 7.0 / 10.0 

- 0.5 / 0.25 / 0.125 / 0.05
