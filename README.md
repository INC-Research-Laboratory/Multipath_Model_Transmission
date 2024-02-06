# Multipath_Model_Transmission

Outline
---
개별 기기에서 모델 학습을 진행한 후 중앙 서버로 가중치들을 전송하는 분산식 학습 방식인 연합학습에서 가용 유휴 장치와 다중 경로를 사용하지 않는 한계를 극복하기위해 모델 분할 전송을 통한 연합학습 개선연구를 수행하였다. 

본 프로젝트는 이 실험의 일부인 **모델 분할 전송에 관한 프로젝트**이며, 다중 경로 환경 구성 이후 진행하였다.

자세한 내용은 논문을 참고하면 된다.

Requirements
---
Python 3.9 version

Preparation
---

이 실험은 클라이언트 역할을 하는 1개의 Jetson Xavier와 서버 역할을 하는 1개의 Laptop을 사용하였다. 클라이언트에 1개의 NIC, 서버에는 2개의 NIC를 연결하여 주었고, AP 또한 같은 모델을 사용하여서 성능의 차이가 나지 않게 하였다.

같은 실험 환경 구현을 위해 하나의 클라이언트에서 서버의 다중 경로(2개의 Path) 연결 이후에 Single Path, Multi Path 실험을 진행하였다. 

또한 이기종의 특성에 맞게 Multi Path에서의 AP의 버전(ac , n)을 맞추어 주었다.


`OS` - Ubuntu 22.04(Server - MSI Sword GF76 A11UE) / Ubuntu 20.04(Client - Jetson Xavier) 

`Server` - Laptop (MSI Sword GF76 A11UE) 1대

`Client` - Jetson Xavier 1대

`NIC` - ipTIME A3000UA 4대

`AP` - iptime ac1200 2대

File
---
`Final Code` folder

- `Client` folder
>**Client_1path.py** - 단일 경로 전송 실험시 사용

>**Client_2path.py** - 다중 경로 전송 실험시 사용

- `Server` folder
>**server_main.py** - 실험 초기에 실행 및 서버 작동

Run 
----
본 실험은 Server 와 Client의 디바이스에서 각 기능에 맞는 코드를 실행시켜야 한다. 

또한 script는 모두 Linux환경에서 실행한다. 

- `Server`

```
pyhton server_main.py
```

- `Client`
>**단일** 경로 전송 실험 시 
```
pyhton Client_1path.py
```
>**다중** 경로 전송 실험 시
```
pyhton Client_2path.py
```

Result
---

- Transmission time comparison between 2.4GHz(n mode, 200MB) and 5GHz(ac mode, 200MB)

![Fig_14](https://github.com/INC-Research-Laboratory/Multipath_Model_Transmission/assets/145684303/ac4f81a6-52cd-4439-9390-b580da6492bb)

- Transmission time comparison between 2.4GHz(n mode, 80MB) and 5GHz(ac mode, 320MB)

![Fig_15](https://github.com/INC-Research-Laboratory/Multipath_Model_Transmission/assets/145684303/c4271bcf-7e68-4ad1-b64d-09abfc317ea5)

- Performance comparison between single path and multi path

![Fig_17](https://github.com/INC-Research-Laboratory/Multipath_Model_Transmission/assets/145684303/bb20ce5d-a588-46a1-aeef-5b8daddca947)
