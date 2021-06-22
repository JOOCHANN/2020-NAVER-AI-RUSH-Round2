# 스팸 메일 분류 (Spam Mail Classification)

## Task
+ 비식별화된 메일의 제목과 본문으로부터 메일의 스팸여부 판정<br/>(classify spam / ham mail from an indexed title / content)

## Requirements
```
numpy>=1.18.1
torch==1.3.1
sklearn
NSML
```
+ NSML binary 설치는 [여기](https://airush.nsml.navercorp.com/download)를 참조 (refer [here](https://airush.nsml.navercorp.com/download) to install NSML)
+ requirements를 pip로 설치 (run pip install for requirements)
  ```
  pip install -r requirements.txt
  ```
+ local Python에서 NSML 라이브러리를 사용하기 위해 다음 pip 명령어를 별도로 실행<br/>(run the below pip install to use NSML library on local Python)
  ```
  pip install git+https://github.com/n-CLAIR/nsml-local
  ```

## Dataset Description
### Names
+ rush7-1
+ rush7-2
+ rush7-3

### Structure
```
\_ train
    \_ train_data (text file)
    \_ train_label (text file)
\_ test
    \_ test_data (text file)
    \_ test_label (text file - hided)
\_ test_submit
    \_ test_data (text file - 5 fake samples)
    \_ test_label (text file - 5 fake samples) 
```

### Label (Caution!)
+ `0`: spam
+ `1`: ham (= not spam)

## Run Train Session
```
nsml run -d {CHALLENGE_NAME} -c 4 -g 1 --memory 8G -m "baseline" -a "--epochs 1"
```

## List of Checkpoints in the Session
```
nsml model ls {USER_ID}/{CHALLENGE_NAME}/{SESSION_NUMBER}
```

## Submit Session
```
nsml submit {USER_ID}/{CHALLENGE_NAME}/{SESSION_NUMBER} {CHECKPOINT_NAME}
```
1. baseline code는 각 epoch마다 checkpoint 저장하며, validation score가 갱신될 때마다 `best_model` checkpoint를 overwrite 합니다<br/>(baseline code saves each epoch and overwrite `best_model` when validation score is updated)
2. baseline 모델의 각 dataset 별로 submit 소요시간은 아래와 같습니다
    1. rush7-1: `Infer test set takes 3.4091691970825195 seconds`
    2. rush7-2: `Infer test set takes 6.5713207721710205 seconds`
    3. rush7-3: `Infer test set takes 9.261323690414429 seconds`
