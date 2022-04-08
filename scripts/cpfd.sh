#! /bin/bash

BASE_IP=192.168.100.
END_POINTS=(134 199 121 120 107 124 181 196)

for e in "${END_POINTS[@]}"
do
	ip=${BASE_IP}${e}
	echo "${ip}의 정보를 가져오고 있습니다..."

	sshpass -p cpfd scp pi@${ip}:/home/pi/PMS7003/particle${e}.txt ../datasets/indoor_particles/
	echo "다운로드 성공"

done

