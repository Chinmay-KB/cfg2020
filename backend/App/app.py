from flask import Flask
from flask import jsonify,after_this_request,make_response
import json
import sqlquery as query
from flask_cors import CORS, cross_origin
app = Flask(__name__)
CORS(app)
app.config['JSON_SORT_KEYS'] = False


@app.route('/getRows',methods=['GET'])
def getRows():
	all_rows, column_names=query.getRows()
	res=[{column_names[i]:all_rows[i] for i in range(len(all_rows))} for row in all_rows]

	return jsonify(res)


@app.route('/')
def hello_name():
   return 'Hello'

if __name__ == '__main__':
   app.run(host='localhost', port=5000)