from flask import *
import matplotlib.pyplot as plt
import random as r
from sklearn.datasets import load_digits
import io
import base64
import joblib

model=joblib.load('dig_model')

app = Flask(__name__)
dig=load_digits()
@app.route('/')
def build_plot():

    img = io.BytesIO()
    rand=r.randint(0,1700) 
    
    i1=dig.images[rand]
    plt.gray()
    plt.matshow(i1)
    plt.xticks([])
    plt.yticks([])
    plt.savefig(img, format='png')
    img.seek(0)

    plot_url = base64.b64encode(img.getvalue()).decode()

    return render_template('index.html',plot_url=plot_url,rand=rand)


@app.route('/',methods=['POST'])
def select():
       return build_plot()


@app.route('/result',methods=['POST'])
def predict():
    img = io.BytesIO()
    pre=int(request.form['image'])
    plt.gray()
    i1=dig.images[pre]
    plt.matshow(i1)
    plt.xticks([])
    plt.yticks([])
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    predicted= model.predict(dig.data[[pre]])
    return render_template('result.html',plot_url=plot_url,predicted=predicted)


if __name__ == '__main__':
    app.run(debug = True)