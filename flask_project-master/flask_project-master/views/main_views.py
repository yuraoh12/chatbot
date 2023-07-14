from flask import Blueprint, render_template, url_for
from pybo.models import Question
from werkzeug.utils import redirect

bp = Blueprint('main', __name__, url_prefix='')


@bp.route('/hello')
def hello_pybo():
    return 'Hello, Pybo!'


@bp.route('/gugu/<int:num>')
def gugu8(num):
    #question_list = Question.query.order_by(Question.create_date.desc())
    return render_template('question/question_list.html', dan=num)

@bp.route('/fruit')
def fruit_list():

    fruit_list = ['사과', '바나나', '포도', '배', '오렌지']
    return render_template('question/fruit_list.html', fruit_list=fruit_list)


@bp.route('/test')
def atest():
    return render_template('question/a_test.html')

@bp.route('/')
def index():
    return redirect(url_for('question._list'))