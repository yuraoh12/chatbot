from flask import Blueprint, request, jsonify
import json

bp = Blueprint('iris', __name__, url_prefix='/iris')


@bp.route('/get-data')
def test() :

    hong = {}
    hong["이름"] = "이순신"
    hong["거주지"] = "서울"
    hong["나이"] = 33

    json_str = json.dumps(hong, ensure_ascii=False)

    return json_str

@bp.route('/predict', methods=['POST'])
def predict() :
    data = request.get_json()
    sepal_length = data["sepal_length"]
    sepal_width = data["sepal_width"]
    petal_length = data["petal_length"]
    petal_width = data["petal_width"]

    from sklearn.datasets import load_iris

    iris = load_iris()  # 붓꽃 데이터

    data = iris.data  # 학습용 데이터
    target = iris.target
    feature_names = iris.feature_names  # 특성명
    target_names = iris.target_names  # 타겟명

    from sklearn.model_selection import train_test_split

    trd, tsd, trt, tst = train_test_split(data, target, test_size=0.2, random_state=42)

    from sklearn.ensemble import RandomForestClassifier

    rfc = RandomForestClassifier(max_depth=5, n_estimators=30)
    rfc.fit(trd, trt)

    pred_idx = rfc.predict([[sepal_length, sepal_width, petal_length, petal_width]])
    result = target_names[pred_idx[0]]

    result_dic = {
        "result" : result
    }

    return json.dumps(result_dic)
