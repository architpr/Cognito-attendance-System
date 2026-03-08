from flask import Flask, render_template, Response, redirect, url_for, request
import cv2
import os
import pickle
import face_recognition
import numpy as np
import cvzone
from datetime import datetime
import json
from dotenv import load_dotenv

load_dotenv()

import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import storage


app = Flask(__name__)  # initializing


# database credentials
cred_json = os.environ.get("FIREBASE_CREDENTIALS")
if cred_json:
    # We are on Render, load from environment string
    cred_dict = json.loads(cred_json)
    cred = credentials.Certificate(cred_dict)
else:
    # We are testing locally
    cred = credentials.Certificate("serviceAccountKey.json")
    
firebase_admin.initialize_app(
    cred,
    {
        "databaseURL": "https://cognito-2312c.firebaseio.com/",
        "storageBucket": "cognito-2312c.firebasestorage.app",
    },
)

bucket = storage.bucket()


def dataset(id):
    studentInfo = db.reference(f"Students/{id}").get()
    if studentInfo is not None:
        blob = bucket.get_blob(f"static/Files/Images/{id}.jpg")
        if blob is not None:
            array = np.frombuffer(blob.download_as_string(), np.uint8)
            imgStudent = cv2.imdecode(array, cv2.COLOR_BGRA2BGR)
            if studentInfo["last_attendance_time"] is not None:
                datetimeObject = datetime.strptime(studentInfo["last_attendance_time"], "%Y-%m-%d %H:%M:%S")
                secondElapsed = (datetime.now() - datetimeObject).total_seconds()
            else:
                datetimeObject = None
                secondElapsed = None
            return studentInfo, imgStudent, secondElapsed
    return None


import base64

already_marked_id_student = []
already_marked_id_admin = []

# Global state for video processing
modeType = 0
current_student_id = -1
imgStudent_global = []
counter = 0

@app.route("/process_frame", methods=["POST"])
def process_frame():
    global modeType, current_student_id, imgStudent_global, counter
    
    data = request.json
    if not data or 'image' not in data:
        return {"error": "No image provided"}, 400
        
    # Decode base64 image from frontend
    img_data = base64.b64decode(data['image'].split(',')[1])
    np_arr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    
    # Load background and modes
    imgBackground = cv2.imread("static/Files/Resources/background.png")
    if imgBackground is None:
        return {"error": "Background not found"}, 500
        
    # Resize camera feed to match UI cutout
    img = cv2.resize(img, (640, 480))
    imgBackground[162 : 162 + 480, 55 : 55 + 640] = img
    
    folderModePath = "static/Files/Resources/Modes/"
    modePathList = os.listdir(folderModePath)
    imgModeList = [cv2.imread(os.path.join(folderModePath, path)) for path in modePathList]
    
    imgBackground[44 : 44 + 633, 808 : 808 + 414] = imgModeList[modeType]
    
    # Process face
    imgSmall = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgSmall = cv2.cvtColor(imgSmall, cv2.COLOR_BGR2RGB)
    
    try:
        with open("EncodeFile.p", "rb") as file:
            encodeListKnownWithIds = pickle.load(file)
        encodedFaceKnown, studentIDs = encodeListKnownWithIds
    except:
        encodedFaceKnown, studentIDs = [], []

    faceCurrentFrame = face_recognition.face_locations(imgSmall)
    encodeCurrentFrame = face_recognition.face_encodings(imgSmall, faceCurrentFrame)
    
    if faceCurrentFrame and encodedFaceKnown:
        for encodeFace, faceLocation in zip(encodeCurrentFrame, faceCurrentFrame):
            matches = face_recognition.compare_faces(encodedFaceKnown, encodeFace)
            faceDistance = face_recognition.face_distance(encodedFaceKnown, encodeFace)
            
            if len(faceDistance) > 0:
                matchIndex = np.argmin(faceDistance)
                
                y1, x2, y2, x1 = faceLocation
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                bbox = 55 + x1, 162 + y1, x2 - x1, y2 - y1
                imgBackground = cvzone.cornerRect(imgBackground, bbox, rt=0)
                
                if matches[matchIndex]:
                    current_student_id = studentIDs[matchIndex]
                    if counter == 0:
                        cvzone.putTextRect(imgBackground, "Face Detected", (65, 200), thickness=2)
                        counter = 1
                        modeType = 1
                else:
                    cvzone.putTextRect(imgBackground, "Face Not Found", (65, 200), thickness=2)
                    modeType = 4
                    counter = 0
                    imgBackground[44 : 44 + 633, 808 : 808 + 414] = imgModeList[modeType]

        if counter != 0:
            if counter == 1:
                res = dataset(current_student_id)
                if res:
                    studentInfo, imgStudent_global, secondElapsed = res
                    if secondElapsed is None or secondElapsed > 60:
                        ref = db.reference(f"Students/{current_student_id}")
                        studentInfo["total_attendance"] += 1
                        ref.child("total_attendance").set(studentInfo["total_attendance"])
                        ref.child("last_attendance_time").set(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                    else:
                        modeType = 3
                        counter = 0
                        imgBackground[44 : 44 + 633, 808 : 808 + 414] = imgModeList[modeType]
                        already_marked_id_student.append(current_student_id)
                        already_marked_id_admin.append(current_student_id)
            
            if modeType != 3:
                if 5 < counter <= 10:
                    modeType = 2
                imgBackground[44 : 44 + 633, 808 : 808 + 414] = imgModeList[modeType]
                
                if counter <= 5:
                    res = dataset(current_student_id)
                    if res:
                        studentInfo, _, _ = res
                        cv2.putText(imgBackground, str(studentInfo["total_attendance"]), (861, 125), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
                        cv2.putText(imgBackground, str(studentInfo["major"]), (1006, 550), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
                        cv2.putText(imgBackground, str(current_student_id), (1006, 493), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
                        standing = studentInfo.get("standing", "N/A")
                        cv2.putText(imgBackground, str(standing), (910, 625), cv2.FONT_HERSHEY_COMPLEX, 0.6, (100, 100, 100), 1)
                        
                        (w, h), _ = cv2.getTextSize(str(studentInfo["name"]), cv2.FONT_HERSHEY_COMPLEX, 1, 1)
                        offset = (414 - w) // 2
                        cv2.putText(imgBackground, str(studentInfo["name"]), (808 + offset, 445), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 50), 1)
                        
                        if imgStudent_global is not None and len(imgStudent_global) > 0:
                            imgStudentResize = cv2.resize(imgStudent_global, (216, 216))
                            imgBackground[175 : 175 + 216, 909 : 909 + 216] = imgStudentResize
                counter += 1
                if counter >= 10:
                    counter = 0
                    modeType = 0
                    current_student_id = -1
                    imgStudent_global = []
                    imgBackground[44 : 44 + 633, 808 : 808 + 414] = imgModeList[modeType]
    else:
        modeType = 0
        counter = 0
        
    _, buffer = cv2.imencode(".jpeg", imgBackground)
    encoded_img = base64.b64encode(buffer).decode('utf-8')
    return {"image": f"data:image/jpeg;base64,{encoded_img}"}

#########################################################################################################################


@app.route("/")
def index():
    return render_template("index.html")


@app.route('/loginspage.html')
def login():
    firebase_config = {
        "apiKey": os.getenv("FIREBASE_API_KEY"),
        "authDomain": os.getenv("FIREBASE_AUTH_DOMAIN"),
        "projectId": os.getenv("FIREBASE_PROJECT_ID"),
        "storageBucket": os.getenv("FIREBASE_STORAGE_BUCKET"),
        "messagingSenderId": os.getenv("FIREBASE_MESSAGING_SENDER_ID"),
        "appId": os.getenv("FIREBASE_APP_ID"),
        "measurementId": os.getenv("FIREBASE_MEASUREMENT_ID")
    }
    return render_template('loginspage.html', firebase_config=firebase_config)

@app.route('/signup.html')
def signup():
    firebase_config = {
        "apiKey": os.getenv("FIREBASE_API_KEY"),
        "authDomain": os.getenv("FIREBASE_AUTH_DOMAIN"),
        "projectId": os.getenv("FIREBASE_PROJECT_ID"),
        "storageBucket": os.getenv("FIREBASE_STORAGE_BUCKET"),
        "messagingSenderId": os.getenv("FIREBASE_MESSAGING_SENDER_ID"),
        "appId": os.getenv("FIREBASE_APP_ID"),
        "measurementId": os.getenv("FIREBASE_MEASUREMENT_ID")
    }
    return render_template('signup.html', firebase_config=firebase_config)

@app.route('/aboutus.html')
def aboutus():
    return render_template('aboutus.html')

@app.route('/contact.html')
def contact():
    return render_template('contact.html')

@app.route('/home.html')
def home():
    return render_template('home.html')


#########################################################################################################################










#########################################################################################################################


@app.route("/admin")
def admin():
    all_student_info = []
    studentIDs, _ = add_image_database()
    for i in studentIDs:
        student_info = dataset(i)
        if student_info is not None:
            all_student_info.append(student_info)
    return render_template("admin.html", data=all_student_info)


@app.route("/admin/admin_attendance_list", methods=["GET", "POST"])
def admin_attendance_list():
    if request.method == "POST":
        if request.form.get("button_student") == "VALUE1":
            already_marked_id_student.clear()
            return redirect(url_for("admin_attendance_list"))
        else:
            request.form.get("button_admin") == "VALUE2"
            already_marked_id_admin.clear()
            return redirect(url_for("admin_attendance_list"))
    else:
        unique_id_admin = list(set(already_marked_id_admin))
        student_info = []
        for i in unique_id_admin:
            student_info.append(dataset(i))
        return render_template("admin_attendance_list.html", data=student_info)



#########################################################################################################################

def add_image_database():
    folderPath = "static/Files/Images"
    imgPathList = os.listdir(folderPath)
    imgList = []
    studentIDs = []

    for path in imgPathList:
        imgList.append(cv2.imread(os.path.join(folderPath, path)))
        studentIDs.append(os.path.splitext(path)[0])

        fileName = f"{folderPath}/{path}"
        bucket = storage.bucket("cognito-2312c.firebasestorage.app")
        blob = bucket.blob(fileName)
        blob.upload_from_filename(fileName)

    return studentIDs, imgList


def findEncodings(images):
    encodeList = []

    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)

    return encodeList


@app.route("/admin/add_user", methods=["GET", "POST"])
def add_user():
    id = request.form.get("id", False)
    name = request.form.get("name", False)
    password = request.form.get("password", False)
    major = request.form.get("major", False)
    total_attendance = request.form.get("total_attendance", False)
    
    last_attendance_date = request.form.get("last_attendance_date", False)
    last_attendance_time = request.form.get("last_attendance_time", False)
    

    
    last_attendance_datetime = f"{last_attendance_date} {last_attendance_time}"
    
    total_attendance = int(total_attendance)
     

    if request.method == "POST":
        image = request.files["image"]
        filename = f"{'static/Files/Images'}/{id}.jpg"
        image.save(os.path.join(filename))

    studentIDs, imgList = add_image_database()

    encodeListKnown = findEncodings(imgList)

    encodeListKnownWithIds = [encodeListKnown, studentIDs]

    file = open("EncodeFile.p", "wb")
    pickle.dump(encodeListKnownWithIds, file)
    file.close()

    if id:
        add_student = db.reference(f"Students")

        add_student.child(id).set(
            {
                "id": id,
                "name": name,
                "password": password,
                "major": major,
                "total_attendance": total_attendance,
                "last_attendance_time": last_attendance_datetime,
            }
        )

    return render_template("add_user.html")


#########################################################################################################################


@app.route("/admin/edit_user", methods=["POST", "GET"])
def edit_user():
    value = request.form.get("edit_student")

    studentInfo, imgStudent, secondElapsed = dataset(value)
    hoursElapsed = round((secondElapsed / 3600), 2)

    info = {
        "studentInfo": studentInfo,
        "lastlogin": hoursElapsed,
        "image": imgStudent,
    }

    return render_template("edit_user.html", data=info)


#########################################################################################################################


@app.route("/admin/save_changes", methods=["POST", "GET"])
def save_changes():
    content = request.get_data()

    dic_data = json.loads(content.decode("utf-8"))

    dic_data = {k: v.strip() for k, v in dic_data.items()}

    dic_data["year"] = int(dic_data["year"])
    dic_data["total_attendance"] = int(dic_data["total_attendance"])
    dic_data["starting_year"] = int(dic_data["starting_year"])

    update_student = db.reference(f"Students")

    update_student.child(dic_data["id"]).update(
        {
            "id": dic_data["id"],
            "name": dic_data["name"],
            "major": dic_data["major"],
            "total_attendance": dic_data["total_attendance"],
            "last_attendance_time": dic_data["last_attendance_time"],
        }
    )

    return "Data received successfully!"


#########################################################################################################################


def delete_image(student_id):
    filepath = f"static/Files/Images/{student_id}.jpg"

    os.remove(filepath)

    bucket = storage.bucket()
    blob = bucket.blob(filepath)
    blob.delete()

    return "Successful"


@app.route("/admin/delete_user", methods=["POST", "GET"])
def delete_user():
    content = request.get_data()

    student_id = json.loads(content.decode("utf-8"))

    delete_student = db.reference(f"Students")
    delete_student.child(student_id).delete()

    delete_image(student_id)

    studentIDs, imgList = add_image_database()

    encodeListKnown = findEncodings(imgList)

    encodeListKnownWithIds = [encodeListKnown, studentIDs]

    file = open("EncodeFile.p", "wb")
    pickle.dump(encodeListKnownWithIds, file)
    file.close()

    return "Successful"


import traceback

def init_system():
    print("Initializing system... Syncing images from Firebase Storage")
    os.makedirs("static/Files/Images/", exist_ok=True)
    bucket = storage.bucket("cognito-2312c.firebasestorage.app")
    blobs = bucket.list_blobs(prefix="static/Files/Images/")
    downloaded = False
    
    for blob in blobs:
        file_path = blob.name
        # blob.name will be like 'static/Files/Images/123.jpg'
        if not os.path.exists(file_path):
            print(f"Downloading {file_path}...")
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            blob.download_to_filename(file_path)
            downloaded = True
            
    if downloaded or not os.path.exists("EncodeFile.p"):
        print("Rebuilding EncodeFile.p...")
        try:
            folderPath = "static/Files/Images"
            if os.path.exists(folderPath):
                imgPathList = os.listdir(folderPath)
                imgList = []
                studentIDs = []
                for path in imgPathList:
                    if path.endswith(('.png', '.jpg', '.jpeg')):
                        img = cv2.imread(os.path.join(folderPath, path))
                        if img is not None:
                            imgList.append(img)
                            studentIDs.append(os.path.splitext(path)[0])
                if imgList:
                    encodeListKnown = findEncodings(imgList)
                    encodeListKnownWithIds = [encodeListKnown, studentIDs]
                    with open("EncodeFile.p", "wb") as file:
                        pickle.dump(encodeListKnownWithIds, file)
                    print("EncodeFile.p rebuilt successfully.")
                else:
                    print("No images found to encode.")
        except Exception as e:
            print(f"Error rebuilding encodings: {e}")
            traceback.print_exc()

# Run initialization procedure to handle ephemeral storage
init_system()

#########################################################################################################################
if __name__ == "__main__":
    
    app.run(host="0.0.0.0", port=7860)
