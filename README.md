
```sh
    ├── readme.md          <- 공간에 대한 목적과 개요를 나타내요.
    │
    ├── notebook           <- 모델 파이프라인 내용 설명이 있어요. 
    │
    ├── main.py            <- 모델 파이프라인 내용입니다.
    │
    ├── requirement.txt    <- 라이브러리 설치 목록이에요.
```

```sh
# 분류 모델을 실행할 수 있어요.
$ python baseline.py --is_classifier=1 

# 회귀 예측 모델을 실행해요.
$ python baseline.py --is_classifier=0
```