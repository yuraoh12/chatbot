from sklearn.datasets import load_iris

iris = load_iris() # 붓꽃 데이터

data = iris.data # 학습용 데이터
target = iris.target
feature_names = iris.feature_names # 특성명
target_names = iris.target_names # 타겟명

from sklearn.model_selection import train_test_split

trd, tsd, trt, tst = train_test_split(data, target, test_size=0.2, random_state=42)

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(max_depth=5, n_estimators=30)
rfc.fit(trd, trt)

rst = rfc.predict([[5,5,2,2]])
