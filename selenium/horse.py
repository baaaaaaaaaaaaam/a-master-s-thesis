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
    if delta.days < 0:
        return 0
    return delta.days
    

def insert_data(data,num,mycursor,wednesday):
   for line in data.split('\n'):
        if line.strip():  # 빈 줄은 무시
            try:
                fields = line.split()
                name,birth_place,gender,birth_date,grade,total_appearances,total_first_place,total_second_place,total_third_place,yearly_appearances,yearly_first_place,yearly_second_place,yearly_third_place,rating=[None] * 14
                if len(fields)==21:
                    name=fields[0]
                    birth_place = fields[1]
                    gender = fields[2]
                    birth_date = fields[3]
                    grade = fields[5]
                    total_appearances = fields[-10]
                    total_first_place = fields[-9]
                    total_second_place = fields[-8]
                    total_third_place = fields[-7]
                    yearly_appearances = fields[-6]
                    yearly_first_place = fields[-5]
                    yearly_second_place = fields[-4]
                    yearly_third_place = fields[-3]
                    rating = 0
                else:
                    isfieldsInt = False
                    try:
                        int(fields[-11])
                        isfieldsInt = True
                    except:
                        pass
                    if isfieldsInt:
                        name=fields[0]
                        birth_place = fields[1]
                        gender = fields[2]
                        birth_date = fields[3]
                        grade = fields[5]
                        total_appearances = fields[-11]
                        total_first_place = fields[-10]
                        total_second_place = fields[-9]
                        total_third_place = fields[-8]
                        yearly_appearances = fields[-7]
                        yearly_first_place = fields[-6]
                        yearly_second_place = fields[-5]
                        yearly_third_place = fields[-4]
                        rating = fields[-2]
                    else:
                        name=fields[0]
                        birth_place = fields[1]
                        gender = fields[2]
                        birth_date = fields[3]
                        grade = fields[5]
                        total_appearances = fields[-10]
                        total_first_place = fields[-9]
                        total_second_place = fields[-8]
                        total_third_place = fields[-7]
                        yearly_appearances = fields[-6]
                        yearly_first_place = fields[-5]
                        yearly_second_place = fields[-4]
                        yearly_third_place = fields[-3]
                        rating = 0
                
                # 데이터 삽입 쿼리
                insert_query = """
                INSERT INTO horse (
                    round, name, birth_date,  birth_place, gender, grade,rating,
                    total_appearances, total_first_place, total_second_place,total_third_place,
                    yearly_appearances, yearly_first_place, yearly_second_place,yearly_third_place,
                    age_day, current_day) 
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,%s, %s,%s, %s)
                """
                age_day = diffDays(birth_date,wednesday)

                # 데이터 삽입
                mycursor.execute(insert_query, (
                    num, name, birth_date,birth_place, gender, grade, rating,
                    total_appearances, total_first_place, total_second_place,total_third_place,
                    yearly_appearances, yearly_first_place, yearly_second_place,yearly_third_place,
                    age_day, wednesday)
                                 )
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
    url = f'https://race.kra.co.kr/dbdata/fileDownLoad.do?fn=internet/seoul/horse/{day}sdb1.txt&meet=1'
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