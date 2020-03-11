from flask import Flask,render_template,url_for,request
import pandas as pd 
import gpt_2_simple as gpt2


app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
	sess = gpt2.start_tf_sess()
	gpt2.load_gpt2(sess, run_name='review_star_1_large')
	if request.method == 'POST':
		message = request.form['message']
		my_prediction = gpt2.generate(sess, run_name= 'review_star_1_large', length=50, prefix= message, sample_delim = '<|endoftext|>', include_prefix=False, nsamples=1, return_as_list=True)

	return render_template('result.html',prediction = my_prediction[0])



if __name__ == '__main__':
	app.run(debug=True)