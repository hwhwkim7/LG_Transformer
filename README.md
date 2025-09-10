# LG_Transformer

## Project Folder Structure

- 📂 **code**
  - 📄 data_processor.py
  - 📄 dataloader.py
  - 📄 main.py
  - 📄 model.py
  - 📄 train.py
  - 📄 utils.py
  - 📂 checkpoints : 코드 실행 후 생성될 예정
    - 📄 best_model.pth : 가장 성능이 좋은 모델을 저장
    - 📄 metrics.csv : train/valid/test 결과가 저장됨
      - .._window : 각 윈도우에 대한 결과
      - .._sequence : 시퀀스에 해당하는 윈도우를 모아서 얻은 결과
- 📂 **dataset**
  - 📂 final_raw : 분할 & 증강한 dataset
  - 📂 final_npy : 코드 실행 후 전처리된 데이터가 저장됨
- 📄 README.md

```
python main.py
```
위 명령어를 통해 전처리부터 모델 학습/저장/test까지 수행
