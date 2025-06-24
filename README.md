# Small Korean VLM Project!

📢 2025년 1학기 [AIKU](https://github.com/AIKU-Official) 활동으로 진행한 프로젝트입니다.   
🎉 2025년 1학기 AIKU Conference xx 수상!

---
## 소개

기존의 한국어 VLM들은 영어로 된 데이터셋을 한글로 번역한 데이터셋에 의존하여 성능이 좋지 않았습니다.    
이에 다양한 한국어 데이터셋과 함께 고성능을 유지하면서도 경량화된 구조를 설계하여,   
한국어 성능 및 한국 문화 이해 특화된 small VLM을 구축하는 것을 목표로 하였습니다.   
또한 이는 real time inference가 필요한 온디바이스 모델에 적합하므로 그 실용성까지 보인 프로젝트입니다.

---
## 방법론

- 한국어 및 한국 문화에 특화된 VLM이 부족한 상황 개선
- 디바이스에 탑재 가능한 경량 모델 구조 설계 (1B 이하)
- 현존하는 한국어 small VLM들 뿐만아니라 14배 더 큰 모델인 VLM KoLLLaVA-7B 대비 더 나은 성능 달성

### 모델링
<img src="https://github.com/user-attachments/assets/5e5417b4-328d-451f-8756-b26fe2754513"  width="600" />   



- **Image Encoder**: *clip-vit-large-patch14-336*
- **Connection Module**:
    - MLP Projection : 이미지 feature 변환
    - Average pooling : token 압축
    - ResNet Block : 시각 정보 향상
- **Language Model**: *HyperCLOVAX-SEED-Text-Instruct-0.5B*


<img src="https://github.com/user-attachments/assets/1bb88acc-9db0-4ec7-afba-767242cf6305"  width="600" />   


### **Stage 1 : Pretrain**

- Connection module 만 학습하며, 나머지는 얼려 image 를 text space로 align 시킵니다.
- AdamW Optimizer
- epoch : 1
- lr : 1e-3
- cosine learning rate decay

| **데이터셋** | **설명** |
| --- | --- |
| [KoLLaVA-CC3M-Pretrain-595K](https://huggingface.co/datasets/tabtoyou/KoLLaVA-CC3M-Pretrain-595K) | LLaVA Pretrain 데이터셋 한국어 번역 |
| [LLaVA-CC3M-Pretrain-595K](https://huggingface.co/datasets/liuhaotian/LLaVA-CC3M-Pretrain-595K) | LLaVA Pretrain 데이터셋 (CC3M) |
| [LAION/CC/SBU BLIP-Caption Concept-balanced 558K](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain/blob/main/blip_laion_cc_sbu_558k.json) | LLaVA Pretrain 데이터셋 (LAION) |



### **Stage 2 : Instruction Tuning**
- Connection module 과 LLM LoRA를 활성화하여 학습하고 나머지는 얼립니다.
- model이 image와 text 기반의 instruction을 따르도록 finetuning합니다
- AdamW Optimizer
- 1 epoch
- lr : 2e-5
- cosine learning rate decay
- LoRA rank 64
- alpha 16



### **Stage 별 데이터셋**
stage 2 에서는 기존 LLaVA instruction tuning 데이터셋을 한국어 번역한 것을 사용할 뿐 아니라   
직접 다양한 추가 fine-tuning 데이터셋을 구축하였습니다. 

| **데이터셋** | **설명** |
| --- | --- |
| [KoLLaVA-v1.5-Instruct-581k](https://huggingface.co/datasets/tabtoyou/KoLLaVA-v1.5-Instruct-581k) | LLaVA-Instruction 데이터셋 한국어 번역 |
| [LLaVA-Instruct-150K](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/blob/main/llava_v1_5_mix665k.json) | LLaVA-Instruction 데이터셋 |
| [AIHub 한국이미지(음식)](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=79) | 한국 음식 이미지를 포함한 시각-언어 데이터 |
| [AIHub 한국형사물 이미지](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=144) | 한국 사물 기반 이미지를 포함한 시각-언어 데이터 |
| [AIHub 다중언어 OCR 데이터](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=71730) | 다양한 언어 기반 OCR 학습용 데이터 |
| [AIHub 한국어 글자체 이미지](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=81) | 다양한 한글 글꼴 데이터 |
| [AIHub GQA 데이터](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=71711) | 한국어 기반 VQA(Visual Question Answering) |
| [AIHub 외부지식 기반 멀티모달 QA](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=71357) | 문서 기반 시각 질의응답 데이터 |



stage 2의 데이터셋 크기 비율은 다음과 같습니다:
![image](https://github.com/user-attachments/assets/995ddc73-d384-4426-b507-4220fcef9fd7)



### **Eval Metric**
task 별로 팀원들이 벤치마크를 구축하여 모델 평가에 활용했습니다.

| 카테고리 | 세부 벤치마크 / 태스크 | 관련 데이터셋 / 설명 | 평가 지표 (Eval Metric) |
| :------------------- | :-------------------------- | :------------------------------------------- | :---------------------------------------------------- |
| **1. 한국어 VLM 벤치마크** | 폐쇄형 벤치마크 | K-SEED, K-DTCBench, K-MMStar, K-MMBench | Accuracy |
| | 개방형 벤치마크 | K-LLaVA-WL | LLM 평가 (LLM Evaluation) |
| **2. 추론 지연 시간 (Inference Latency)** |-| - | 초당 생성 토큰 개수 (token/s) |
| **3. 이미지 캡셔닝 (Image Captioning)** | 한국어 이미지 캡셔닝 | MS COCO 한국어 검증 세트 | **N-gram 기반:** METEOR, CIDEr <br> **임베딩 기반:** KR-SBERT를 이용한 Cosine Similarity |
| **4. 한국어 OCR** | 단순 한국어 OCR | - | IoU (Intersection over Union) |
| | 복합 한국어 OCR (차트, 다이어그램, 그래프) | - | LLM 평가 (LLM Evaluation) |
| **5. 한국 문화 이해** | 전반적인 문화 이해 | - | Accuracy |


### **Benchmark Results**
<details>
  <summary> 벤치마크 성능을 자세히 보려면 여기를 누르세요! </summary>
    <h3> 1. Korean VLM benchmarks from NCSOFT </h3>
    <img src="https://github.com/user-attachments/assets/6b66a7db-a926-4176-ba88-0074b2924beb" />   
    <h3> 2. Inference Latency </h3>
    <img src="https://github.com/user-attachments/assets/3468036b-5649-44d6-88a5-b3875149d72b" />   
    <h3> 3. Image Captioning </h3>
    <img src="https://github.com/user-attachments/assets/8b54c6e5-d59b-4222-a428-517e8feb5a56" width="700"/>   
    <h3> 4.1. Korean OCR - simple Korean OCR </h3>
    <img src="https://github.com/user-attachments/assets/42217c72-491e-44a8-98dd-e8a9c1936552" width="400"/>   
    <h3> 4.2. Korean OCR - complex Korean OCR (chart, diagram, graph) </h3>
    <img src="https://github.com/user-attachments/assets/b31c1b73-f279-40ab-b4a8-121d65cbef43" width="400"/>   
    <h3> 5. Korean Culture </h3>
    <img src="https://github.com/user-attachments/assets/f0c2cd3a-cb29-4ebb-8e8a-820d75646d83" width="400" />   
    
</details>

---

## 환경 설정

(Requirements, Anaconda, Docker 등 프로젝트를 사용하는데에 필요한 요구 사항을 나열해주세요)
1. local demo - 용량상 모델의 pth 가 github에 업로드되어있지 않습니다.
```bash
git clone https://github.com/AIKU-Official/aiku-25-1-Small-Korean-VLM.git

cd aiku-25-1-Small-Korean-VLM
conda env create --file environment.yaml

cd /LLaVA/llava/demo
python demo_local.py  # our model 현재 체크포인트 및 경로 상 실행 불능
python demo_local.py # huggingface 모델로 대체
```
2. 잘 안 될 경우
별도 환경설정 없이 gradio 데모가 실행 가능한 상태입니다.(25.06.26)

[Small Korean VLM 데모 링크!](https://c2832db5d94df7fdf0.gradio.live/)

---
## 데모 사용 방법

1. 질문하고 싶은 이미지를 업로드합니다
2. 지시문을 명확하고 구체적으로 제시하면 원하는 답을 얻을 확률이 높아집니다.
---
## 예시 결과 데모 동영상

https://github.com/user-attachments/assets/a62062ca-39bb-4861-abe5-ecd238c5c9ec

---
## 팀원 및 역할

팀원분들 모두 고생하셨습니다!
  | 팀원 중 수상대상자                            | 역할                                       |
| ----------------------------- | ---------------------------------------- |
| [정다현](https://github.com/) |    모델 train 및 inference, OCR 및 VQA 데이터 관련    |
| [권보영](https://github.com/)     |    기컴비 발표, 한국문화 데이터, gradio 데모 제작     |
| [최현욱](https://github.com/)        |    모델링, 데이터 벤치마크 제작 및 평가    |
| [이제윤](https://github.com/)        |    한국문화 데이터 수집 가공, 관련 벤치마크 제작   |

위 표는 AIKU 프로젝트 심사 대상 팀원입니다. 이외에도 학회 외 인원과 심사위원 등 이름을 언급하지 못한 분들에게도 감사를 표합니다. :)

