import tempfile
import json
import os
import glob
import shutil
from ansys.mapdl.core import launch_mapdl

save_directory_txt = 'D:/ansys/others/-300/txt'  #####고치셈
if not os.path.exists(save_directory_txt):
    os.makedirs(save_directory_txt)

save_directory_img = 'D:/ansys/others/-300/img'  #####고치셈
if not os.path.exists(save_directory_img):
    os.makedirs(save_directory_img)

save_directory_json = 'D:/ansys/others/-300/json'  #####고치셈
if not os.path.exists(save_directory_json):
    os.makedirs(save_directory_json)

# 반복할 단계 수 설정
step = 300  # 시간

for i in range(1, 2):
    # MAPDL 실행
    mapdl = launch_mapdl()

    # 기존 데이터베이스를 초기화하고 새로 시작
    mapdl.clear()
    mapdl.prep7()

    # PCB 크기 설정

    board_width = 0.05
    board_height = 0.05
    board_thickness = 0.002

    # PCB 블록 생성
    mapdl.block(0, board_width, 0, board_height, 0, board_thickness)

    # FR-4 재료 속성 설정
    mapdl.mp('EX', 1, 2.04e10)  # Young's Modulus X direction
    mapdl.mp('EY', 1, 1.84e10)  # Young's Modulus Y direction
    mapdl.mp('EZ', 1, 1.5e10)  # Young's Modulus Z direction

    mapdl.mp('PRXY', 1, 0.11)  # Poisson's Ratio XY
    mapdl.mp('PRYZ', 1, 0.09)  # Poisson's Ratio YZ
    mapdl.mp('PRXZ', 1, 0.14)  # Poisson's Ratio XZ

    mapdl.mp('GXY', 1, 9.2e9)  # Shear Modulus XY
    mapdl.mp('GYZ', 1, 8.4e9)  # Shear Modulus YZ
    mapdl.mp('GXZ', 1, 6.6e9)  # Shear Modulus XZ

    mapdl.mp('DENS', 1, 1840)  # Density

    mapdl.mp('ALPX', 1, 1.25e-5)  # Thermal Expansion Coefficient X direction
    mapdl.mp('ALPY', 1, 1.14e-5)  # Thermal Expansion Coefficient Y direction
    mapdl.mp('ALPZ', 1, 8.25e-5)  # Thermal Expansion Coefficient Z direction

    mapdl.mp('KXX', 1, 0.38)  # Thermal Conductivity X direction
    mapdl.mp('KYY', 1, 0.38)  # Thermal Conductivity Y direction
    mapdl.mp('KZZ', 1, 0.36)  # Thermal Conductivity Z direction

    # 요소 유형 및 메시 설정
    mapdl.et(1, 'SOLID186')
    mapdl.vmesh('ALL')

    # 경계 조건 설정 (한 면을 고정)
    mapdl.nsel('S', 'LOC', 'Z', 0)

    mapdl.d('ALL', 'UX', 0)  # X축 변위를 고정
    mapdl.d('ALL', 'UY', 0)  # Y축 변위를 고정

    mapdl.allsel()

    # 초기 온도 설정
    initial_temp = 25  # 기준 온도
    ambient_temp = -300  # 주변 온도
    conv_coeff = 100  # 대류 열전달 계수 (W/m^2-K)

    mapdl.bf('ALL', 'TEMP', initial_temp)  # 초기 온도 설정
    # 전체 표면에 대류 열전달 조건 적용
    mapdl.sf('ALL', 'CONV', conv_coeff, ambient_temp)

    # SOLUTION 모드로 전환
    mapdl.slashsolu()

    # 과도 열 해석 설정
    mapdl.antype('TRANS')  # 과도 해석
    mapdl.trnopt('FULL')  # 전체 과도 해석
    mapdl.timint('ON')  # 시간 적분 활성화
    mapdl.kbc(0)  # 선형적 시간 증가

    # 시간 단계 설정
    total_time = step  # 전체 해석 시간
    time_steps = 1  # 시간 단계 수 (1초 간격)
    mapdl.deltim(1)  # 시간 증분 1초
    mapdl.nsubst(total_time / time_steps, 1, 1)  # 시간 단계 옵션

    # 각 시간 단계의 시간 설정

    for current_time in range(1, total_time + 1):
        mapdl.time(current_time)

        # 해석 실행
        mapdl.solve()

    mapdl.finish()

    # 작업 디렉토리와 결과 파일 확인
    print(f"MAPDL 작업 디렉토리: {mapdl.directory}")
    result_file = os.path.join(mapdl.directory, 'file.rst')

    # 후처리: 결과 시각화
    mapdl.post1()

    # 각 시간 단계별 결과를 반복하여 플롯
    for current_time in range(1, total_time + 1):
        mapdl.set(time=current_time)

        #######################################노드값 출력###############################################################
        # 첫 번째 노드 선택
        max_uz = -float('inf')
        min_uz = float('inf')
        max_node = -1
        min_node = -1

        # 모든 노드를 순회하면서 최대 및 최소 UZ 변위 찾기
        for i in range(1, 2073):
            uz_node = mapdl.get("uz", "node", i, "u", "z")
            if uz_node > max_uz:
                max_uz = uz_node
                max_node = i
            if uz_node < min_uz:
                min_uz = uz_node
                min_node = i
        # 최대 및 최소 UZ 변위 출력
        json_data = {
            'MINIMUM VALUES': {
                'NODE': min_node,
                'VALUE': min_uz,
            },
            'MAXIMUM VALUES': {
                'NODE': max_node,
                'VALUE': max_uz,
            }
        }

        print(f"최대 UZ 변위: 노드 {max_node}에서 {max_uz}")
        print(f"최소 UZ 변위: 노드 {min_node}에서 {min_uz}")

        json_file_path = os.path.join(save_directory_json, f'MINMAX_{current_time}.json')
        with open(json_file_path, 'w') as json_file:
            json.dump(json_data, json_file, indent=4)

        ##################################################출력끝#########################################################

        # Z축 변위 출력
        strain_data = mapdl.prnsol('U', 'Z')  # 변형률 데이터를 가져옴
        strain_file = os.path.join(save_directory_txt, f'strain_data_{current_time}.txt')
        with open(strain_file, 'w') as file:
            file.write(strain_data)

        # 이미지를 저장할 파일 경로 설정
        result_file = os.path.join(save_directory_img, f'displacement_{current_time}s.png')

        # Z축 변위를 이미지로 저장
        mapdl.plnsol('U', 'Z')
        mapdl.show('PNG')
        temp_files = glob.glob(os.path.join(mapdl.directory, '*.png'))

        # 가장 최근에 생성된 PNG 파일 찾기
        if temp_files:
            latest_file = max(temp_files, key=os.path.getctime)
            shutil.move(latest_file, result_file)
            print(f"{current_time}초의 Z 방향 변위 이미지가 {result_file}에 저장됨")

    # MAPDL 종료
    mapdl.exit()
# 임시 폴더 삭제
temp_dir = tempfile.gettempdir()
temp_folders = glob.glob(os.path.join(temp_dir, 'ansys_*'))

for folder in temp_folders:
    try:
        shutil.rmtree(folder)
        print(f"임시 폴더 {folder}가 삭제되었습니다.")
    except Exception as e:
        print(f"폴더 {folder} 삭제 중 오류 발생: {e}")
