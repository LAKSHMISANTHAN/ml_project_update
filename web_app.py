import os
from flask_sqlalchemy import SQLAlchemy
from flask import Flask, render_template, request
from sqlalchemy.sql import func
from track import predict
import threading

def scheduled_task(filename):
    row = Example.query.filter_by(file_name = filename).first()
    if(row!=None):
        print("Hello")
        id = row.id
        updated_row = Example.query.filter_by(id = id).first()
        updated_row.processing = 1
        db.session.commit()
        updated_row = Example.query.filter_by(id = id).first()
        l = predict(filename)
        print(l)
        updated_row.processed = 1
        updated_row.processing = 0
        updated_row.total_persons = l[0]+l[1]
        updated_row.male = l[0]
        updated_row.female = l[1]
        updated_row.age = " ".join(map(str, l[2]))
        db.session.commit()

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:root@localhost/pro'
db = SQLAlchemy(app)

class Example(db.Model):
    __tablename__ = 'sample_test5'
    id = db.Column('id',db.Integer, primary_key=True)
    file_name = db.Column('filename',db.Unicode)
    measured_on = db.Column('measured_on',db.TIMESTAMP)
    processed = db.Column('processed',db.Integer,default=0)
    processing = db.Column('processing',db.Integer,default=0)
    total_persons = db.Column('total_persons',db.Integer)
    male = db.Column('male',db.Integer)
    female = db.Column('female',db.Integer)
    age = db.Column('age',db.Unicode)

    def __init__(self, file_name):
        self.file_name = file_name
folder_path = os.path.dirname(os.path.abspath(__file__))

@app.route("/")
def home():
    return render_template("upload.html")

@app.route("/upload" ,methods=['POST'])
def upload():
    target = os.path.join(folder_path)
    for file in request.files.getlist("input_file"):
        filename = file.filename
        destination = "/".join([target, filename])
        file.save(destination)
        new_ex = Example(filename)
        db.session.add(new_ex)
        db.session.commit()
        t1 = threading.Thread(target=scheduled_task, args=(filename,))
        t1.start()
    return 'Video Uploaded Successfully.'

@app.route("/processed_videos")
def processed_videos():
    processed_rows = Example.query.filter_by(processed = 1)
    return render_template("processed_videos.html",data = processed_rows)


if __name__=="__main__":
    app.run(port=4555,debug=True)
