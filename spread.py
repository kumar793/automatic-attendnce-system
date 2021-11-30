import datetime,gspread,random
import oauth2client
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd

scope = ["https://spreadsheets.google.com/feeds",'https://www.googleapis.com/auth/spreadsheets',"https://www.googleapis.com/auth/drive.file","https://www.googleapis.com/auth/drive"]

creds = ServiceAccountCredentials.from_json_keyfile_name("credentials.json", scope)

client = gspread.authorize(creds)
sheet = client.open("face recognization",).sheet1
def enroll(name,Roll_Number,d,t):
    nrows = len(sheet.col_values(1))
    pin=random.randint(999,9999)
    sheet.update_cell(nrows+1,1,name)
    sheet.update_cell(nrows+1,2,Roll_Number)
    sheet.update_cell(nrows+1,3,d)
    sheet.update_cell(nrows+1,4,t)
   
    

   
   
