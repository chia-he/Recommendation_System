
##### Import libraries

import warnings
from flask.helpers import url_for

from werkzeug.utils import redirect
warnings.filterwarnings('ignore') # stop from gensim warning (set before 'gensim')

from gensim.models import Word2Vec 
from flask import Flask, render_template, request
import pymongo
import numpy as np
import json
from  pandas import DataFrame as df
import pandas as pd


##### Functions

# Find Most N similar products
def similar_products(model, products_dict, v, n = 6):
    ms = model.wv.similar_by_vector(v, topn= n+1)[1:]
    new_ms = []
    for i, j in enumerate(ms):
        pair = (products_dict[j[0]][0], j[1])
        new_ms.append(pair)
        print(i, j[0], pair)
    
    print()
    return new_ms

# Find the StockCode of a product by Description
def findDescription(name, products_dict):
    print(name)
    for key, value in products_dict.items():
        if name.replace(" ", "") == value[0].replace(" ", ""): # Compare ignore whitespaces ( Some products have whitespaces at the end )
            print("Key: ", key)
            return key
    print("Key not found")
    return ""

# Retrieve Data from MongoDB ( Database: mydatabase, Collections: aiot_final )
def retrieveData(recommendation, products_dict):
    print("Retrieve Data!!")

    # Connect MongoDB
    myclient = pymongo.MongoClient("mongodb://localhost:27017/")
    mydb = myclient["mydatabase"]
    mycol = mydb["aiot_final"]

    result = []
    for i, item in enumerate(recommendation):
        stockcode = findDescription(item[0], products_dict)
        myquery = { "StockCode": stockcode }
        mydoc = mycol.find(myquery)
        tmp = [x for x in mydoc]
        if mydoc != None:
            result.append(tmp)
            print(i, ": ", tmp)
            print("Description: ", tmp[0]['Description'])
            print()
        else:
            print("Empty!!")

    return result


##### Flask

app = Flask(__name__)

@app.route("/", methods = ['GET', 'POST'])
def index():

    # load models and data
    model = Word2Vec.load("./model/v2.model")
    model.init_sims(replace=True) # make the model much more memory-efficient
    products_dict = json.loads(open("./data/input/products_dict.json", "r").read())
    plot_eval = json.loads(open("./data/output/plot_eval.json", "r").read())

    if request.method == 'POST':
        print("------- POST -------")

        if  request.form['search']:

            StockCode = ""
            
            # Make the Search Bar capable of processing "Description" queries
            if request.form['search'][0].isdigit(): # StockCode
                StockCode = str(request.form['search'])
                print("StockCode")
            else: # Description
                Description = str(request.form['search'])
                StockCode = findDescription(Description, products_dict)
                print("Description")

            if StockCode == "":
                return render_template('index.html')
            
            result = []
            for item in enumerate(similar_products(model, products_dict, model.wv[StockCode])):
                result.append([ item[1][0], int(round(item[1][1], 2)*100) ])

            # Retrieve Data from MongoDB
            if StockCode == '21246' or StockCode == '21479' or StockCode == '21238': 
                # For those data that exist in database
                data = retrieveData(result, products_dict)
            else: 
                # For those data that didnt exist in the database
                data = []
                for item in result:
                    data.append([ {"Description": item[0], "price": "XX.xx","src": "https://i.pinimg.com/originals/0e/ca/10/0eca10f07594819a0937a0e312a25846.png"} ])

            # print(data)
            return render_template('index.html', products = data, similarity = result, target = products_dict[StockCode][0])
            
        else:
            return render_template('index.html')
    else:
        print("------- GET -------")
       
        lookup = {'bowl': '21246', 'bottle': '21479', 'cup': '21238'}

        StockCode = lookup[plot_eval]
        result = []
        for item in enumerate(similar_products(model, products_dict, model.wv[StockCode])):
            result.append([ item[1][0], int(round(item[1][1], 2)*100) ])

        data = retrieveData(result, products_dict)

        return render_template('index.html', products = data, similarity = result, target = products_dict[StockCode][0])

##### Main

def startWeb():

    # run Flask Web page
    app.run(debug=True, use_reloader=True)

if __name__ == '__main__':
    startWeb()
