from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
#import final

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])

def home():
	if(request.method == 'GET'):
		return render_template("home.html")

	elif(request.method == 'POST'):
		f=request.files['img-path']
		f.save('./uploads/'+secure_filename(f.filename))
		detection=1#final.result()
		if(detection==0):
			return render_template("fire.html")
		else:
			return render_template("nonfire.html")


if __name__ == '__main__':
	app.run(debug=True, host='0.0.0.0')
