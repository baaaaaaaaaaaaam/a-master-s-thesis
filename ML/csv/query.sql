CREATE TABLE `race` (
  `id` int NOT NULL AUTO_INCREMENT,
  `round` int DEFAULT NULL,
  `round_detail` int DEFAULT NULL,
  `distance` int DEFAULT NULL,
  `humidity` int DEFAULT NULL,
  `grade` int DEFAULT NULL,
  `goal_number` int DEFAULT NULL,
  `start_number` int DEFAULT NULL,
  `horse_name` varchar(255) DEFAULT NULL,
  `horse_birth` varchar(255) DEFAULT NULL,
  `horse_sex` varchar(255) DEFAULT NULL,
  `horse_age` int DEFAULT NULL,
  `burden_weight` int DEFAULT NULL,
  `jockey_name` varchar(255) DEFAULT NULL,
  `teacher_name` varchar(255) DEFAULT NULL,
  `rating` int DEFAULT NULL,
  `horse_weight` int DEFAULT NULL,
  `horse_weight_diff` int DEFAULT NULL,
  `record` varchar(255) DEFAULT NULL,
  `current_day` date DEFAULT NULL,
  `win_odd` float DEFAULT NULL,
  `place_odd` float DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb3;


CREATE TABLE jockey (
        id INT AUTO_INCREMENT PRIMARY KEY,
        round INT,
        name VARCHAR(255),
        birth_date DATE,
        age Int,
        debut_date DATE,
        weight_1 INT,
        weight_2 INT,
        total_appearances INT,    ## 총 출전 횟수
        total_first_place INT,    ## 총 1위 횟수
        total_second_place INT,    ## 총 2위 횟수
        total_third_place INT,    ## 총 3위 횟수
        yearly_appearances INT,   ## 1년간 출전 횟수 
        yearly_first_place INT,    ## 1년간 1위 횟수 
        yearly_second_place INT,    ## 1년간 2위 횟수 
        yearly_third_place INT,    ## 1년간 3위 횟수 
        carrer_day INT,
        current_day Date
    )



    CREATE TABLE horse (
    id INT AUTO_INCREMENT PRIMARY KEY,  ## 고유 번호
		round INT,               ## 회차 2020년 1월 첫경기 1회차
    name VARCHAR(255),       ## 경주마 이름
    birth_date DATE,          ## 경주마 생일
    birth_place VARCHAR(255), ## 경주마 산지
    gender VARCHAR(10),      ## 경주마 성별. 
    grade VARCHAR(20),       ## 경주마 등급
    total_appearances INT,    ## 총 출전 횟수
    total_first_place INT,    ## 총 1위 횟수
    total_second_place INT,    ## 총 2위 횟수
    total_third_place INT,    ## 총 3위 횟수
    yearly_appearances INT,   ## 1년간 출전 횟수 
    yearly_first_place INT,    ## 1년간 1위 횟수 
    yearly_second_place INT,    ## 1년간 2위 횟수 
    yearly_third_place INT,    ## 1년간 3위 횟수 
    age_day INT,
    current_day Date,
		rating INT
);


CREATE TABLE IF NOT EXISTS trainer (
        id INT AUTO_INCREMENT PRIMARY KEY,  ## 고유 정보 
				round INT,                ## 회차 2020년 1월 첫경기 1회차
        name VARCHAR(255),        ## 조교사 이름 
        birth_date DATE,          ## 조교사 생일
        age INT,
        debut_date DATE,          ## 조교사 데뷔일
        total_appearances INT,    ## 총 출전 횟수
        total_first_place INT,    ## 총 1위 횟수
        total_second_place INT,    ## 총 2위 횟수
        total_third_place INT,    ## 총 3위 횟수
        yearly_appearances INT,   ## 1년간 출전 횟수 
        yearly_first_place INT,    ## 1년간 1위 횟수 
        yearly_second_place INT,    ## 1년간 2위 횟수 
        yearly_third_place INT,    ## 1년간 3위 횟수 
        carrer_day INT,
        current_day Date
    )



select r.id,
r.distance,

/* if(r.distance = 1000,1,0) as 1000M,
if(r.distance = 1200,1,0) as 1200M,
if(r.distance = 1300,1,0) as 1300M,
if(r.distance = 1400,1,0) as 1400M,
if(r.distance = 1600,1,0) as 1600M,
if(r.distance = 1700,1,0) as 1700M,
if(r.distance = 1800,1,0) as 1800M,
if(r.distance = 1900,1,0) as 1900M,
if(r.distance = 2000,1,0) as 2000M,
if(r.distance = 2300,1,0) as 2300M, */

r.round_detail,r.win,r.win_rate_1,r.win_rate_2,r.start_number,r.horse_weight,
    (SUBSTRING_INDEX(r.record, ':', 1) * 60) + SUBSTRING_INDEX(SUBSTRING_INDEX(r.record, ':', -1), '.', 1) + (SUBSTRING_INDEX(r.record, '.', -1) / 10) AS total_seconds,
if(h.birth_place = "한",true,false) as birth_korea,
if(h.birth_place = "미",true,false) as birth_america,
if(h.birth_place = "한(포)" ,true,false) as birth_other,
if(h.gender = "수" ,true,false) as male,
if(h.gender = "암" ,true,false) as female,
if(h.gender = "거" ,true,false) as castration,
RIGHT(h.grade, 1) as grade,
h.total_appearances,
h.total_first_place,
h.total_second_place,
h.total_third_place,
h.yearly_appearances,
h.yearly_first_place,
h.yearly_second_place,
h.yearly_third_place,
h.age_day,
h.rating,
j.weight as jockey_weight,
j.total_appearances as jockey_total_appearances,
j.total_first_place as jockey_total_first_place,
j.total_second_place as jockey_total_second_place,
j.total_third_place as jockey_total_third_place,
j.yearly_appearances as jockey_yearly_appearances,
j.yearly_first_place as jockey_yearly_first_place,
j.yearly_second_place as jockey_yearly_second_place,
j.yearly_third_place as jockey_yearly_third_place,
j.carrer_day as jockey_carrer,

t.total_appearances as trainer_total_appearances,
t.total_first_place as trainer_total_first_place,
t.total_second_place as trainer_total_second_place,
t.total_third_place as trainer_total_third_place,
t.yearly_appearances as trainer_yearly_appearances,
t.yearly_first_place as trainer_yearly_first_place,
t.yearly_second_place as trainer_yearly_second_place,
t.yearly_third_place as trainer_yearly_third_place,
t.carrer_day as trainer_carrer

from race r 
left join horse h on r.horse_name = h.name and r.round = h.round 
left join jockey j on r.jockey_name = j.name and r.round = j.round
left join trainer t on r.teacher_name = t.name and r.round = t.round

where  h.id is not null and SUBSTRING(grade, 2) REGEXP '^[0-9]+$' and j.weight is not null and j.weight != 0 
order by r.id ;