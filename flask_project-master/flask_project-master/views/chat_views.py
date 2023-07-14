from flask import Blueprint, render_template, request
import requests
import json

bp = Blueprint('chat', __name__, url_prefix='/chat')



@bp.route('/iris-data')
def iris_data() :
    sepal_length = int(request.args.get("sepal_length"))
    sepal_width = int(request.args.get("sepal_width"))
    petal_length = int(request.args.get("petal_length"))
    petal_width = int(request.args.get("petal_width"))

    # AI 서버로 전송 (Json)

    ## dictionary로 데이터 구조화
    iris = {}
    iris["sepal_length"] = sepal_length
    iris["sepal_width"] = sepal_width
    iris["petal_length"] = petal_length
    iris["petal_width"] = petal_width

    ## dictionary를 json 문자열로 변환
    json_str = json.dumps(iris, ensure_ascii=False)

    ## json 문자열을 AI서버에 요청할 때 넣어서 보낸다.
    url = "http://127.0.0.1:8099/iris/predict"
    headers = {'Content-type' : 'application/json'}
    res = requests.post(url=url, data=json_str, headers=headers)

    data = res.json()
    print(data["result"])

    return render_template("chat/iris_result.html", data=data)

@bp.route('/iris-form')
def iris_form():

    return render_template("chat/iris_test.html")

@bp.route('/test')
def test() :
    res = requests.get("http://127.0.0.1:8099/iris/get-data")
    data = res.json()

    return render_template('chat/data_test.html', data=data)
