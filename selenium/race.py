from datetime import datetime, timedelta
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
import time
import mysql.connector
from datetime import datetime
import math
import re


def connectMysql():
    mydb = mysql.connector.connect(
    host="10.0.2.29",
    user="root",
    password="roboarete!1",
    database="horse"
    )
    return mydb

def race_road(data):
    _pattern = r'주로:(\S{2})'
    _match = re.search(_pattern, data)
    if _match:
        road = _match.group(1)
    else:
        road = None
    return road

def getGame(data):
    _pattern = r'경주명 : (\S{2})'
    _match = re.search(_pattern, data)
    if _match:
        game = _match.group(1)
    else:
        game = None
    return game

def getGrade(data):
    _pattern = r'(\d{1})등급'
    _match = re.search(_pattern, data)
    if _match:
        grade = _match.group(1)
    else:
        grade = None
    return grade


def getDistance(data):
    distance_pattern = r'(\d{4})M'
    distance_match = re.search(distance_pattern, data)
    if distance_match:
        distance = distance_match.group(1)
    else:
        distance = None
    return distance

def getHumidity(data):
    humidity_pattern = r'(\d{1,2})%'
    humidity_match = re.search(humidity_pattern, data)
    if humidity_match:
        humidity = humidity_match.group(1)
    else:
        humidity = None
    return humidity
def getPart1(_data,raceResult):
 
    try:
        data = _data.split('\n')
        for line in data:
            if len(line) == 0:
                continue
            fields = line.split()
            순위,마번,마명,산지,성별,연령,부담증량,기수명,조교사,레이팅 = [None] * 10
            
            if len(fields)==10:
                순위 = fields[0]
                마번 = fields[1]
                마명 = fields[2]
                산지 = fields[3]
                성별 = fields[4]
                연령 = fields[5]
                부담증량 = fields[6]
                기수명 = fields[7]
                조교사 = fields[8]
                레이팅 = 0
            else:
                순위 = fields[0]
                마번 = fields[1]
                마명 = fields[2]
                산지 = fields[3]
                성별 = fields[4]
                연령 = fields[5]
                부담증량 = fields[6]
                기수명 = fields[7]
                조교사 = fields[8]
                레이팅 = fields[10]
            # print(순위,마번,마명,산지,성별,연령,부담증량,기수명,조교사,레이팅)
            raceResult[마번] = [순위,마번,마명,산지,성별,연령,부담증량,기수명,조교사,레이팅]
    except Exception as e:
        print(e)
    return raceResult


def getPart2(data,raceResult):
    pattern2 = r'\s+(\d+)\s+(\d+)\s+((\S+|\S+\s\S+))\s+(-?\d+)\(\s*([+-]?\d+)\)\s+(\d+:\d+\.\d+)\s+'
    for line in data.split('\n'):
        matches = re.findall(pattern2, line)
        for match in matches:
            마번 = match[1]
            중량 = match[4]
            증감량 = match[5]

            경기기록 = match[6]
            raceResult[마번].extend([중량,증감량,경기기록])
    return raceResult

def getPart3(data,raceResult):
    try:
        data = data.split('\n')
        for line in data:
            if len(line) == 0:
                continue
            fields = line.split()
            마번,승리배당,입상배당 = [None] * 3
            마번 = fields[1]
            승리배당 = fields[-2]
            입상배당 = fields[-1]
            # print(순위,마번,마명,산지,성별,연령,부담증량,기수명,조교사,레이팅)
            raceResult[마번].extend( [승리배당,입상배당])
    except Exception as e:
        print(e)
    return raceResult

    # if int(distance) < 1600:
    # # 인덱스 0번과 7번 값을 찾는 정규식
    #     pattern3 = r"^\s*\S+\s+(\S+)\s+\S+\s+\S+\s+\S+\s+\S+\s+\S+\s+(\S+)\s+(\S+)"
    #     # 각 줄에 대해 정규식을 적용하여 값을 가져옴
    #     for line in data3.splitlines():
    #         match = re.match(pattern3, line)
    #         if match:
    #             마번 = match.group(1)
    #             배당 = match.group(2)
    #             배당1 = match.group(3)

    #             raceResult[마번].extend([배당,배당1])
    # elif int(distance) == 1600:
    # # 인덱스 0번과 7번 값을 찾는 정규식
    #     pattern3 = r"^\s*\S+\s+(\S+)\s+\S+\s+\S+\s+\S+\s+\S+\s+\S+\s+\S+\s+(\S+)\s+(\S+)"
    #     # 각 줄에 대해 정규식을 적용하여 값을 가져옴
    #     for line in data3.splitlines():
    #         match = re.match(pattern3, line)
    #         if match:
    #             마번 = match.group(1)
    #             배당 = match.group(2)
    #             배당1 = match.group(3)

    #             raceResult[마번].extend([배당,배당1])
    # else:
    #     pattern3 = r"^\s*\S+\s+(\S+)\s+\S+\s+\S+\s+\S+\s+\S+\s+\S+\s+\S+\s+\S+\s+(\S+)\s+(\S+)"
    #     # 각 줄에 대해 정규식을 적용하여 값을 가져옴
    #     for line in data3.splitlines():
    #         match = re.match(pattern3, line)
    #         if match:
    #             마번 = match.group(1)
    #             배당 = match.group(2)
    #             배당1 = match.group(3)
    #             raceResult[마번].extend([배당,배당1])
    return raceResult


num_detail = 1
def insert_data(preText,num,mycursor,_days):
    global num_detail
    days = str(_days)
    arr = None
    if datetime(int(days[0:4]),int(days[4:6]),int(days[6:])) < datetime(2019,7,1):
        arr= preText.split('----------------------------------------------------------------------------------------------')
    else:
        arr= preText.split('-------------------------------------------------------------------------------------------------')

    for i in range(0, len(arr)-1, 14):
        try:
            raceResult={}
            distance = getDistance(arr[i])
            humidity = getHumidity(arr[i])
            game = getGame(arr[i])
            grade = getGrade(arr[i])
            road= race_road(arr[i])
            if road == '불량':
                humidity = 20
            if game == '일반' :
                raceResult = getPart1(arr[i+2],raceResult)
                raceResult = getPart2(arr[i+4],raceResult)
                raceResult = getPart3(arr[i+6],raceResult)
                # print(raceResult)
                for i in raceResult:
                    race = raceResult[i]
                    # print(race)
                    # break
                    insert_query = """
                            INSERT INTO race (
                                round, round_detail, distance, humidity, grade, goal_number, start_number,
                                horse_name, horse_birth, horse_sex, horse_age, burden_weight, 
                                jockey_name, teacher_name, rating, horse_weight, horse_weight_diff, record, win_odd,place_odd,current_day 
                                )
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,%s,%s, %s)
                            """
                    # ['1', '8', '메가드래곤', '한', '수', '3', '52.0', '빅투아르', '리카디', '37', '511', '-8', '1:41.7'],
                    mycursor.execute(insert_query, (
                                num, num_detail, distance, humidity, grade, race[0], race[1],
                                race[2], race[3], race[4], race[5], race[6],
                                race[7], race[8], race[9], race[10], race[11], race[12],race[13], race[14],days
                            ))
        except Exception as e:
            print(e)
        num_detail+=1

        
def get_sat_sun():
    # 2019년 1월 1일부터 12월 31일까지 확인합니다.
    start_date = datetime(2018, 8, 8)
    end_date = datetime(2023, 12, 31)

    # 결과를 저장할 리스트
    sat_sun_dates = []

    # 하루씩 이동하면서 토요일과 일요일을 확인합니다.
    current_date = start_date
    while current_date <= end_date:
        # 현재 날짜의 요일을 가져옵니다. (0: 월요일, 1: 화요일, ..., 6: 일요일)
        weekday = current_date.weekday()
        
        # 만약 토요일(5)이거나 일요일(6)이라면 리스트에 추가합니다.
        if weekday == 5 or weekday == 6:
            sat_sun_dates.append(current_date.strftime("%Y%m%d"))
        
        # 다음 날짜로 이동합니다.
        current_date += timedelta(days=1)

    return sat_sun_dates


def runSelenium(day):
    service = Service()
    service.executable_path ='/home/riley/horse/selenium/chrome-linux64' 
    driver = webdriver.Chrome(service=service)

    time.sleep(2)
    url = f'https://race.kra.co.kr/dbdata/fileDownLoad.do?fn=chollian/seoul/jungbo/rcresult/{day}dacom11.rpt&meet=1'
    driver.get(url)
    preText = driver.find_element(By.XPATH,'//pre').text
    driver.quit()
    return preText

def main():
    # 특정 연도 범위 내에서 매주 수요일의 날짜 구하기
    days = get_sat_sun()
    num = 0.0
    for day in days:
        print(day)
        preText= runSelenium(day)
        mydb = connectMysql()
        mycursor = mydb.cursor()
        insert_data(preText,math.floor(num),mycursor,day)
        # 변경사항 저장
        mydb.commit()
        # 연결 종료
        mydb.close()
        num+=0.5

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(e)
