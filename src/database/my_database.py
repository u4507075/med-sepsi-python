#!/usr/bin/python
import MySQLdb
import pandas

def connect():
    db = MySQLdb.connect(host="127.0.0.1",    # your host, usually localhost
                         user="root",         # your username
                         passwd="root",  # your password
                         port=3307,
                         db="ds_main_ap2_pt")        # name of the data base
    return db

def get_data(sql):
    db = connect()
    df = pandas.read_sql(sql, con=db)
    db.close()
    return df
