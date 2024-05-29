import face_recognition
import cv2
import numpy as np
import os
import xlwt
import datetime
import xlrd
from xlutils.copy import copy as xl_copy
import time
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders

T = datetime.datetime.now().strftime('%H-%M-%S %d-%m-%Y')

# Function to create or open an Excel workbook and worksheet
def create_or_open_sheet(file, sheet_name):
    if not os.path.exists(file):
        wb = xlwt.Workbook()
        sheet = wb.add_sheet(sheet_name)
        wb.save(file)
    else:
        rb = xlrd.open_workbook(file, formatting_info=True)
        wb = xl_copy(rb)
        sheet_names = rb.sheet_names()
        for name in sheet_names:
            if name == sheet_name:
                return wb, wb.get_sheet(sheet_names.index(name))
        sheet = wb.add_sheet(sheet_name)
    return wb, sheet

# Function to load known face encodings
def load_known_faces(image_folder):
    known_face_encodings = []
    known_face_names = []

    for file_name in os.listdir(image_folder):
        image_path = os.path.join(image_folder, file_name)
        img = face_recognition.load_image_file(image_path)
        encoding = face_recognition.face_encodings(img)[0]  # Assuming only one face per image
        known_face_encodings.append(encoding)
        known_face_names.append(os.path.splitext(file_name)[0])

    return known_face_encodings, known_face_names

# Function to load master data
def load_master_data(file):
    workbook = xlrd.open_workbook(file)
    sheet = workbook.sheet_by_index(0)
    master_data = {}
    for i in range(1, sheet.nrows):
        roll_number = str(sheet.cell_value(i, 0))
        name = sheet.cell_value(i, 1)
        phone_number = str(sheet.cell_value(i, 2))  # Index 2 for phone number
        email_address = str(sheet.cell_value(i, 3))  # Index 3 for email
        master_data[roll_number] = {'name': name, 'phone_number': phone_number, 'email_address': email_address}
    return master_data

# OpenCV video capture
video_capture = cv2.VideoCapture(0)

# Get current directory
CurrentFolder = os.getcwd()

# Create or open the Excel sheet for attendance
attendance_folder = os.path.join(CurrentFolder, 'attendance_collections')
if not os.path.exists(attendance_folder):
    os.makedirs(attendance_folder)

attendance_file = os.path.join(attendance_folder, f'attendance_{T}.xls')

# Write headers for attendance sheet
attendance_workbook, attendance_sheet = create_or_open_sheet(attendance_file, 'Attendance')
attendance_sheet.write(0, 0, 'Roll Number')
attendance_sheet.write(0, 1, 'Name')
attendance_sheet.write(0, 2, 'Attendance')
attendance_sheet.write(0, 3, 'Timestamp')
attendance_sheet.write(0, 4, 'Email Address')  # Add email address column header

row = 1  # Initialize row variable

# Load known faces
image_folder = "student_images"
known_face_encodings, known_face_names = load_known_faces(image_folder)

# Initialize variables
already_attendance_taken = set()

# Load master data
master_file = os.path.join(CurrentFolder, 'master_dataset.xls')
master_data = load_master_data(master_file)

# Track recognized students
recognized_students = set()

# Set the duration (in seconds) for face recognition
recognition_duration = 60  # Change this value as needed

# Start time for face recognition
start_time = time.time()

while True:
    ret, frame = video_capture.read()

    # Resize frame for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Find all faces and face encodings in the current frame
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        roll_number = "Unknown"
        if True in matches:
            match_index = matches.index(True)  # Get the index of the first match
            roll_number = known_face_names[match_index]
            recognized_students.add(roll_number)

            if roll_number not in already_attendance_taken:
                student_info = master_data.get(roll_number)
                if student_info:  # Check if student_info is not None
                    name = student_info.get('name', "Unknown")
                    email_address = student_info.get('email_address', "example@example.com")  # Handle missing email address
                    timestamp = datetime.datetime.now().strftime('%H:%M:%S %d-%m-%Y')
                    attendance_sheet.write(row, 0, roll_number)
                    attendance_sheet.write(row, 1, name)
                    attendance_sheet.write(row, 2, "Present")
                    attendance_sheet.write(row, 3, timestamp)
                    attendance_sheet.write(row, 4, email_address)  # Write email address to attendance sheet
                    row += 1
                    already_attendance_taken.add(roll_number)

            # Draw a rectangle around the face
            cv2.rectangle(frame, (left * 4, top * 4), (right * 4, bottom * 4), (0, 255, 0), 2)
            # Draw roll number text
            cv2.putText(frame, roll_number, (left * 4, bottom * 4 + 20), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow('Video', frame)

    # Check if the time limit for face recognition has been reached
    if time.time() - start_time >= recognition_duration:
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the face recognition loop, find all students who are absent
absent_students = set(master_data.keys()) - recognized_students

# Start writing absent students from the next row after present students
absent_row = row  # Start writing absent students from the next row
for roll_number in absent_students:
    student_info = master_data.get(roll_number)
    if student_info:
        name = student_info.get('name', "Unknown")
        email_address = student_info.get('email_address', "example@example.com")
        timestamp = datetime.datetime.now().strftime('%H:%M:%S %d-%m-%Y')
        attendance_sheet.write(absent_row, 0, roll_number)
        attendance_sheet.write(absent_row, 1, name)
        attendance_sheet.write(absent_row, 2, "Absent")
        attendance_sheet.write(absent_row, 3, timestamp)
        attendance_sheet.write(absent_row, 4, email_address)  # Write email address to attendance sheet
        absent_row += 1

# Save attendance workbook and release resources
attendance_workbook.save(attendance_file)

print("Attendance has been taken successfully")

# Release video capture
video_capture.release()
cv2.destroyAllWindows()

# Send emails to absentees' parents
for roll_number in absent_students:
    student_info = master_data.get(roll_number)
    if student_info:
        name = student_info.get('name', "Unknown")
        email_address = student_info.get('email_address', "example@example.com")
        timestamp = datetime.datetime.now().strftime('%H:%M:%S %d-%m-%Y')
        message_body = f"Dear Parent,\n\nThis is to inform you that your child {name} was absent for the class on {timestamp}.\n\nManagement,\nRCEE"
        
        # Email configurations
        from_email = "rceefrs@gmail.com"  # Your Gmail address
        to_email = email_address  # Absentee's parent's email address
        subject = f"Attendance Notification: Absence of {name}"

        # Setup the email message
        msg = MIMEMultipart()
        msg['From'] = from_email
        msg['To'] = to_email
        msg['Subject'] = subject
        msg.attach(MIMEText(message_body, 'plain'))

        # Connect to the SMTP server and send the email
        smtp_server = "smtp.gmail.com"
        smtp_port = 587
        smtp_username = 'rceefrs@gmail.com'  # Your Gmail address
        smtp_password = 'itsq wawx djto rsvf'  # Your Gmail password or app-specific password

        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(smtp_username, smtp_password)
        server.sendmail(from_email, to_email, msg.as_string())
        server.quit()

print("Emails sent to absentees' parents.")

#send the report to the faculty
subject = f"Attendance for {T}"
from_email = "rceefrs@gmail.com"  # Your Gmail address
to_email = "rceefrs@gmail.com"  # Faculty's email address

msg = MIMEMultipart()
msg['From'] = from_email
msg['To'] = to_email
msg['Subject'] = subject

with open(attendance_file, "rb") as attachment:
    part = MIMEBase("application", "octet-stream")
    part.set_payload(attachment.read())

encoders.encode_base64(part)

part.add_header(
    "Content-Disposition",
    f"attachment; filename= {attendance_file}",
)

msg.attach(part)

smtp_server = "smtp.gmail.com"
smtp_port = 587
smtp_username = 'rceefrs@gmail.com'  # Your Gmail address
smtp_password = 'itsq wawx djto rsvf'  # Your Gmail password or app-specific password

server = smtplib.SMTP(smtp_server, smtp_port)
server.starttls()
server.login(smtp_username, smtp_password)
text = msg.as_string()
server.sendmail(from_email, to_email, text)
server.quit()

print("Attendance sheet sent to the subject faculty.")