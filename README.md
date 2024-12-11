# nuclio

도커 이미지를 다른 PC로 옮길 때, 용량이 12GB인 경우 Git LFS를 사용하려면 다음 단계를 따르면 됩니다:

1. 도커 이미지 저장
먼저, 도커 이미지를 tar 파일로 저장합니다.

bash
코드 복사
docker save -o image.tar your-image-name:tag
your-image-name:tag는 저장하려는 도커 이미지의 이름과 태그입니다. 이 명령은 image.tar라는 tar 파일을 생성합니다.

2. 이미지 파일 분할
Git LFS는 대용량 파일을 지원하지만, Git의 기본 동작 및 효율성을 고려해 이미지 파일을 분할하는 것이 좋습니다.

Linux/macOS:

bash
코드 복사
split -b 2G image.tar image-part-
Windows (PowerShell):

powershell
코드 복사
Get-Content image.tar -ReadCount 0 | Out-File -FilePath image-part-{0}.tar -Width 2147483648
이 명령은 2GB 크기의 파일로 image-part- 접두사를 가진 분할 파일을 생성합니다.

3. Git LFS 설정 및 분할 파일 추가
Git LFS를 사용하도록 저장소를 설정합니다.

bash
코드 복사
git lfs install
git lfs track "*.tar*"
추가한 분할 파일을 Git 저장소에 커밋합니다.

bash
코드 복사
git add image-part-*
git commit -m "Add docker image parts"
git push origin main
4. 다른 PC로 가져오기
다른 PC에서 저장소를 클론한 후, 파일을 병합합니다.

bash
코드 복사
cat image-part-* > image.tar
5. 도커 이미지 복원
병합된 image.tar 파일로 도커 이미지를 복원합니다.

bash
코드 복사
docker load -i image.tar
주의 사항
저장소 크기 제한: Git LFS로 업로드 가능한 파일 크기와 저장소 크기를 확인하세요.
네트워크 속도: 12GB 업로드에는 시간이 걸릴 수 있으니, 안정적인 네트워크 환경에서 작업하세요.
추가 설정 필요: 필요시 LFS 파일 크기 제한 설정을 조정하세요.
bash
코드 복사
git config --global lfs.concurrenttransfers 10
git config --global lfs.storage /path/to/custom/storage





