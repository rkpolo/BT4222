from flask import Flask,Response,render_template,url_for,request,jsonify
from flask_bootstrap import Bootstrap
import pandas as pd 
import gpt_2_simple as gpt2
import json


app = Flask(__name__)
Bootstrap(app)

#Reactive code
@app.route('/')
def interactive_input():
	return render_template('interactive.html')

#Reactive function that will enable the code to run 
@app.route('/background_process')
def background_process():
	try:
		lang = request.args.get('message', 0, type=str)
		complexity = request.args.get('complexity', 0, type=str)
		complexity = float(complexity)
		sess = gpt2.start_tf_sess()
		gpt2.load_gpt2(sess, run_name='review_star_1_30000')
		my_prediction = gpt2.generate(sess, run_name= 'review_star_1_30000',temperature=complexity, length=15, prefix= lang, sample_delim = '<|endoftext|>', include_prefix=False, nsamples=3, return_as_list=True)
		res1 = str(my_prediction[0]).replace('<|endoftext|>', '')
		res2 = str(my_prediction[1]).replace('<|endoftext|>', '')
		res3 = str(my_prediction[2]).replace('<|endoftext|>', '')
		return jsonify(result1 = res1, result2 = res2, result3 = res3)
	except Exception as e:
		return str(e)


if __name__ == '__main__':
	app.run(debug=True)