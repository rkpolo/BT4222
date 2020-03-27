from flask import Flask,Response,render_template,url_for,request,jsonify
import pandas as pd 
import gpt_2_simple as gpt2
from wtforms import TextField, Form
import json


app = Flask(__name__)

#Tesing the autocomplete
cities = ["Bratislava",
          "Banská Bystrica",
          "Prešov",
          "Považská Bystrica",
          "Žilina",
          "Košice",
          "Ružomberok",
          "Zvolen",
          "Poprad"]


class SearchForm(Form):
    autocomp = TextField('Insert City', id='city_autocomplete')


#Original Basic code
@app.route('/predict',methods=['POST'])
def predict():
	sess = gpt2.start_tf_sess()
	gpt2.load_gpt2(sess, run_name='review_star_1_large')
	if request.method == 'POST':
		message = request.form['message']
		my_prediction = gpt2.generate(sess, run_name= 'review_star_1_large', length=50, prefix= message, sample_delim = '<|endoftext|>', include_prefix=False, nsamples=1, return_as_list=True)

	return render_template('result.html',prediction = my_prediction[0])

#Reactive code
@app.route('/')
def interactive_input():
	return render_template('interactive.html')

#Reactive function that will enable the code to run 
@app.route('/background_process')
def background_process():
	try:
		lang = request.args.get('message', 0, type=str)
		sess = gpt2.start_tf_sess()
		gpt2.load_gpt2(sess, run_name='review_star_1_30000')
		my_prediction = gpt2.generate(sess, run_name= 'review_star_1_30000', length=15, prefix= lang, sample_delim = '<|endoftext|>', include_prefix=False, nsamples=3, return_as_list=True)
		print("First result: " + str(my_prediction[0]))
		print("Second result: " + str(my_prediction[1]))
		print("Third result: " + str(my_prediction[2]))
		return jsonify(result1 = my_prediction[0], result2 = my_prediction[1], result3 = my_prediction[2])
	except Exception as e:
		return str(e)

@app.route('/_autocomplete', methods=['GET'])
def autocomplete():
    return Response(json.dumps(cities), mimetype='application/json')


@app.route('/search', methods=['GET', 'POST'])
def index():
    form = SearchForm(request.form)
    return render_template("search.html", form=form)

if __name__ == '__main__':
	app.run(debug=True)