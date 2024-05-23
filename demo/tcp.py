import socket

# 서버 설정
server_ip = '0.0.0.0'  # 모든 인터페이스에서 접근 가능
server_port = 12345  # 사용할 포트 번호
buffer_size = 1024  # 데이터 버퍼 크기

# TCP 서버 시작
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((server_ip, server_port))
server_socket.listen(1)
print(f"서버 {server_ip}:{server_port}에서 대기 중...")

# 클라이언트 연결 대기
conn, addr = server_socket.accept()
print(f"{addr}에서 연결됨")

# 데이터 전송
try:
    while True:
        # mmhuman3d로부터 얻은 데이터를 전송
        # 예제에서는 단순 문자열을 전송하나, 실제 데이터 구조에 맞게 변경 필요
        data = "mmhuman3d 결과 데이터"
        conn.send(data.encode())
        
        # 추가 데이터 전송이 필요한 경우 루프 내에서 처리
        break  # 예제에서는 한 번의 전송으로 종료
finally:
    # 연결 종료
    conn.close()
    server_socket.close()


import socket
import pickle
import time

server_ip = 'localhost'
server_port = 12345

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((server_ip, server_port))

try:
    while True:
        # 서버에 요청 메시지 전송
        print("서버에 요청 전송")
        client_socket.sendall("요청".encode())
        
        # 서버로부터 데이터 수신
        data = b""
        while True:
            packet = client_socket.recv(4096)
            if not packet:
                break
            data += packet
        
        # 수신된 데이터를 NumPy 배열로 복원
        np_data = pickle.loads(data)
        print(f"수신된 NumPy 배열: {np_data}")
        
        # 일정 시간 대기 후 다시 요청 (예: 5초)
        time.sleep(5)
finally:
    client_socket.close()

