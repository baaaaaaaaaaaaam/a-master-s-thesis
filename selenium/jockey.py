from datetime import datetime, timedelta
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
import time
import mysql.connector
from datetime import datetime


def connectMysql():
    mydb = mysql.connector.connect(
    host="10.0.2.29",
    user="root",
    password="roboarete!1",
    database="horse"
    )
    return mydb


def diffDays(debut_str,today_str):
    debut_date = datetime.strptime(debut_str, '%Y/%m/%d')
    today_date = datetime.strptime(today_str, '%Y%m%d')
    delta = today_date - debut_date
    return delta.days
    
def insert_data(data,num,mycursor,wednesday):
    for line in data.split('\n'):
        if line.strip():  # 빈 줄은 무시
            if '기수명' in line:  # 헤더는 건너뜀
                continue
            try:
                fields = line.split()
                name = fields[0]
                birth_date = fields[2]
                age = fields[3]
                debut_date = fields[4]
                weight_1 = int(fields[5])
                weight_2 = int(fields[6])
                total_appearances = int(fields[7])
                total_first_place = int(fields[8])
                total_second_place = int(fields[9])
                total_third_place = int(fields[10])
                yearly_appearances = int(fields[11])
                yearly_first_place = int(fields[12])
                yearly_second_place = int(fields[13])
                yearly_third_place = int(fields[14])
                # 데이터 삽입 쿼리
                insert_query = """
                INSERT INTO jockey (
                    round, name, birth_date, age, debut_date, weight_1, weight_2,
                    total_appearances, total_first_place, total_second_place,total_third_place,
                    yearly_appearances, yearly_first_place, yearly_second_place,yearly_third_place,
                    carrer_day,current_day) 
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """
                # 데이터 삽입
                carrer_day = diffDays(debut_date,wednesday)
                mycursor.execute(insert_query, (
                    num, name, birth_date, age, debut_date, weight_1, weight_2,
                    total_appearances,total_first_place, total_second_place, total_third_place,
                    yearly_appearances, yearly_first_place, yearly_second_place, yearly_third_place,
                    carrer_day,wednesday))
            except Exception as e:
                print(wednesday,e)
                
def wednesdays_between_years():
     # 2019년 1월 1일부터 12월 31일까지 확인합니다.
    start_date = datetime(2018, 8, 8)
    end_date = datetime(2023, 12, 31)

    # 결과를 저장할 리스트
    wednesdays = []

    # 하루씩 이동하면서 토요일과 일요일을 확인합니다.
    current_date = start_date
    while current_date <= end_date:
        # 현재 날짜의 요일을 가져옵니다. (0: 월요일, 1: 화요일, ..., 6: 일요일)
        weekday = current_date.weekday()
        
        # 만약 토요일(5)이거나 일요일(6)이라면 리스트에 추가합니다.
        if weekday == 3:
            wednesdays.append(current_date.strftime("%Y%m%d"))
        
        # 다음 날짜로 이동합니다.
        current_date += timedelta(days=1)

    return wednesdays


def runSelenium(day):
    service = Service()
    service.executable_path ='/home/riley/horse/selenium/chrome-linux64' 
    driver = webdriver.Chrome(service=service)

    time.sleep(2)
    url = f'https://race.kra.co.kr/dbdata/fileDownLoad.do?fn=internet/seoul/jockey/{day}sdb2.txt&meet=1'
    driver.get(url)
    preText = driver.find_element(By.XPATH,'//pre').text
    driver.quit()
    return preText


def main():
    # 특정 연도 범위 내에서 매주 수요일의 날짜 구하기
    wednesdays = wednesdays_between_years()
    num = 0
    for wednesday in wednesdays:
        preText= runSelenium(wednesday)
        mydb = connectMysql()
        mycursor = mydb.cursor()
        insert_data(preText,num,mycursor,wednesday)
        # 변경사항 저장
        mydb.commit()
        # 연결 종료
        mydb.close()
        num+=1

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(e)