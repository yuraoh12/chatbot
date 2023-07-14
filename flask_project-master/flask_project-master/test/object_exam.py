# 1. 클래스 객체

## 객체 => 데이터 묶음 + 함수

### 사람 정보 표현

def introduce(age, name) :
    print(f"{age}살 {name}입니다.")


age = 20
name = "홍길동"
## 한사람의 정보가 여러개면 묶자. -> 클래스
age2 = 22
name2 = "이순신"


## 사람을 모델링함.

## Person class
class Person:
    def __init__(self, age, name, address, phone) :
        self.age = age
        self.name = name
        self.address = address

    def introduce(self) :
        print(f"{self.age}살 {self.name}입니다.")

p1 = Person(age = 20, name = "홍길동", address = "대전") # Person class를 이용해서 복사본 하나 만들기
p2 = Person(22, "이순신", "서울") # Person class를 이용해서 복사본 하나 만들기
p3 = Person(32, "황진이", "광주") # Person class를 이용해서 복사본 하나 만들기

#introduce(p1.age, p1.name)
#introduce(p2.age, p2.name)

p1.introduce()
p2.introduce()
p3.introduce()

p1.age


