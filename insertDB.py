''' 
MongoDB:
Insert data to mydatabase.aiot_final
'''

from pymongo import MongoClient
import numpy as np

# Connect to the server with the hostName and portNumber.
connection = MongoClient("localhost", 27017)
mydb = connection['mydatabase'] # Database
mycol = mydb["aiot_final"] # Collections

mylist = [
  
  # Water Bottle
  { "_id": 1, "StockCode": "22111", "Description": "SCOTTIE DOG HOT WATER BOTTLE", "price": "23.95", "src": "https://m.media-amazon.com/images/I/71k-PKmSWzS._AC_UL320_.jpg"},
  { "_id": 2, "StockCode": "21484", "Description": "CHICK GREY HOT WATER BOTTLE", "price": "11.04", "src": "https://m.media-amazon.com/images/I/71ukZp7nwVL._AC_UL320_.jpg"},
  { "_id": 3, "StockCode": "21485", "Description": "RETROSPOT HEART HOT WATER BOTTLE", "price": "17.99", "src": "https://m.media-amazon.com/images/I/81MmqB0FZYL._AC_UL320_.jpg"},
  { "_id": 4, "StockCode": "23356", "Description": "LOVE HOT WATER BOTTLE", "price": "8.99", "src": "https://m.media-amazon.com/images/I/81aVmCGW8kL._AC_UL320_.jpg"},
  { "_id": 5, "StockCode": "84029G", "Description": "KNITTED UNION FLAG HOT WATER BOTTLE", "price": "9.59", "src": "https://m.media-amazon.com/images/I/71GZ8wnTPaL._AC_UL320_.jpg"},
  { "_id": 6, "StockCode": "84029E", "Description": "RED WOOLLY HOTTIE WHITE HEART.", "price": "30.03", "src": "https://m.media-amazon.com/images/I/61OCIAbncnS._AC_UL320_.jpg"},
  
  # Bowl
  { "_id": 7, "StockCode": "21238", "Description": "RED RETROSPOT CUP", "price": "30.03", "src": "https://m.media-amazon.com/images/I/71BQ3o-SlzL._AC_UL320_.jpg"},
  { "_id": 8, "StockCode": "21242", "Description": "RED RETROSPOT PLATE", "price": "8.99", "src": "https://m.media-amazon.com/images/I/613qdCHs3hL._AC_UL320_.jpg"},
  { "_id": 9, "StockCode": "21537", "Description": "RED RETROSPOT PUDDING BOWL", "price": "9.69", "src": "https://m.media-amazon.com/images/I/61REOoBI4-L._AC_UL320_.jpg"},
  { "_id": 10, "StockCode": "22202", "Description": "MILK PAN PINK POLKADOT", "price": "19.99", "src": "https://m.media-amazon.com/images/I/51TdcGRt4QL._AC_UL320_.jpg"},
  { "_id": 11, "StockCode": "20676", "Description": "RED RETROSPOT BOWL", "price": "23.95", "src": "https://m.media-amazon.com/images/I/61q0NOPo9uS._AC_UL320_.jpg"},
  { "_id": 12, "StockCode": "20675", "Description": "BLUE POLKADOT BOWL", "price": "11.04", "src": "https://m.media-amazon.com/images/I/61b0aMwdQ8L._AC_UL320_.jpg"},

  # Cup
  { "_id": 13, "StockCode": "21240", "Description": "BLUE POLKADOT CUP", "price": "12.55", "src": "https://m.media-amazon.com/images/I/81Ph3SEhtlL._AC_UL320_.jpg"},
  { "_id": 14, "StockCode": "21239", "Description": "PINK POLKADOT CUP", "price": "7.69", "src": "https://m.media-amazon.com/images/I/71pHmptpR2L._AC_UL320_.jpg"},
  { "_id": 15, "StockCode": "21244", "Description": "BLUE POLKADOT PLATE", "price": "17.25", "src": "https://m.media-amazon.com/images/I/613BpRrH1KL._AC_UL320_.jpg"},
  { "_id": 16, "StockCode": "21243", "Description": "PINK POLKADOT PLATE", "price": "9.99", "src": "https://m.media-amazon.com/images/I/61R8+9-gYkL._AC_UL320_.jpg"},
  { "_id": 17, "StockCode": "21245", "Description": "GREEN POLKADOT PLATE", "price": "10.03", "src": "https://m.media-amazon.com/images/I/71N9mYZ3UML._AC_UL320_.jpg"},

]

x = mycol.insert_many(mylist)

# Print list of the _id values of the inserted documents:
print(x.inserted_ids)