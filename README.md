# model-web-deploy

## 프로젝트 설명(Description)
웹 서버에 직접 TF를 올리는 방식과 TFServing을 이용하여 API를 호출하는 두 가지 방식으로 detection 모델의 결과를 웹페이지에 불러올 수 있습니다.
Deploy TF detection model by TFServing and web server.

## 실행 시 주의할 점(Attention to run)
원격 서버에 TFServing을 실행하고 이를 API로 받아와 웹페이지에서 보여주기 위해서는 먼저 TFServing 서버의 외부 접속 주소를 변경해야 합니다.
변경해야 할 곳은 serving_app/views.py 내 `send_api` 함수의 `request_url` 부분입니다.
First, you should change the API server address for receiving API from the remote TFServing server.
You can find the remote server address in the file serving_app/views.py.
Function `send_api` has the variable `request_url`. This is the url of TFServer.
Change this with whatever your server address is.

## 프로젝트 목적(Purpose)
이 프로젝트는 오직 디텍션 모델의 웹 배포 결과만을 보기 위한 것입니다.
The purpose of this project is only for observing the result of a detection model on the web.
