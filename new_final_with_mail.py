import cv2
import os
import csv
import numpy as np
import tkinter as tk
from tkinter import messagebox, LabelFrame, Listbox, Scrollbar
from PIL import Image, ImageTk
import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Global variables
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

if os.path.exists("recognizer/trainningData.yml"):
    recognizer.read("recognizer/trainningData.yml")

# Load names from file or initialize
if os.path.exists('names.csv'):
    with open('names.csv', 'r') as file:
        reader = csv.reader(file)
        names = [row[0] for row in reader]
else:
    names = ['Name']

 #Create necessary directories
if not os.path.exists('dataset'):
    os.makedirs('dataset')
if not os.path.exists('recognizer'):
    os.makedirs('recognizer')

# Function to save names persistently in a CSV
def save_names():
    with open('names.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        for name in names:
            writer.writerow([name])

# Function to register a new user
def register_user():
    def register_action():
        face_id = id_entry.get()
        user_name = name_entry.get()

        if not face_id or not user_name:
            messagebox.showerror("Error", "User ID and Name are required!")
            return

        # Add user name to the names list
        names.append(user_name)
        save_names()

        cam = cv2.VideoCapture(0)
        cam.set(3, 640)
        cam.set(4, 480)

        count = 0
        while True:
            ret, img = cam.read()
            img = cv2.flip(img, 1)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                count += 1
                cv2.imwrite(f"dataset/User.{face_id}.{count}.jpg", gray[y:y+h, x:x+w])
                cv2.imshow('image', img)

            k = cv2.waitKey(100) & 0xff
            if k == 27 or count >= 30:
                break

        cam.release()
        cv2.destroyAllWindows()
        messagebox.showinfo("Info", "User Registered successfully!")
        train_model()
        registration_window.destroy()

    registration_window = tk.Toplevel(window)
    registration_window.title("Register New User")
    registration_window.geometry("500x400")
    registration_window.configure(bg='#A7C7E7')

    id_label = tk.Label(registration_window, text="Enter User ID:", font=("Helvetica", 14), bg='#A7C7E7')
    id_label.pack(pady=10)
    id_entry = tk.Entry(registration_window, font=("Helvetica", 14), width=30)
    id_entry.pack(pady=10)

    name_label = tk.Label(registration_window, text="Enter Name:", font=("Helvetica", 14), bg='#A7C7E7')
    name_label.pack(pady=10)
    name_entry = tk.Entry(registration_window, font=("Helvetica", 14), width=30)
    name_entry.pack(pady=10)

    capture_button = tk.Button(
        registration_window, text="Take Attandance & Train", font=("Helvetica", 14),
        bg="#4682B4", fg="#FFFFFF", command=register_action)
    capture_button.pack(pady=20)

# Function to train the recognizer
def train_model():
    path = 'dataset'

    def get_images_and_labels(path):
        image_paths = [os.path.join(path, f) for f in os.listdir(path)]
        faces = []
        ids = []
        for image_path in image_paths:
            img = Image.open(image_path).convert('L')
            img_np = np.array(img, 'uint8')
            id = int(os.path.split(image_path)[-1].split(".")[1])
            faces.append(img_np)
            ids.append(id)
        return np.array(ids), faces

    ids, faces = get_images_and_labels(path)
    recognizer.train(faces, ids)
    recognizer.save('recognizer/trainningData.yml')
    messagebox.showinfo("Info", "Model trained successfully!")

# Function to recognize a face and mark attendance
def recognize_face_window():
    recognition_window = tk.Toplevel(window)
    recognition_window.title("Face Recognition and Attendance")
    recognition_window.geometry("1000x600")
    recognition_window.configure(bg='#A7C7E7')

    camera_frame = LabelFrame(recognition_window, text="Camera", font=("Helvetica", 16), bg='#B0E0E6', padx=10, pady=10)
    camera_frame.pack(side="left", padx=20, pady=20)

    attendance_frame = LabelFrame(recognition_window, text="Attendance List", font=("Helvetica", 16), bg='#B0E0E6', padx=10, pady=10)
    attendance_frame.pack(side="right", padx=20, pady=20)

    attendance_listbox = Listbox(attendance_frame, width=40, height=20, font=("Helvetica", 14))
    attendance_listbox.pack(side="left", padx=10, pady=10)
    scrollbar = Scrollbar(attendance_frame, orient="vertical")
    scrollbar.pack(side="right", fill="y")
    attendance_listbox.config(yscrollcommand=scrollbar.set)
    scrollbar.config(command=attendance_listbox.yview)

    camera_label = tk.Label(camera_frame)
    camera_label.pack()

    cam = cv2.VideoCapture(0)
    cam.set(3, 640)
    cam.set(4, 480)
    minW = 0.1 * cam.get(3)
    minH = 0.1 * cam.get(4)

    today = datetime.datetime.now().date()
    marked_today = set()
    recognizer.read("recognizer/trainningData.yml")

    def update_frame():
        ret, img = cam.read()
        img = cv2.flip(img, 1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(int(minW), int(minH)),
        )

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            user_id, confidence = recognizer.predict(gray[y:y + h, x:x + w])
            user_name = names[user_id] if confidence < 100 and user_id < len(names) else "Unknown"
            confidence_text = f"{round(100 - confidence)}%"

            if user_name != "Unknown" and user_name not in marked_today:
                mark_attendance(user_name, user_id)
                marked_today.add(user_name)
                attendance_listbox.insert(tk.END, f"{user_name} (ID: {user_id}) - Marked at {datetime.datetime.now()}")
                messagebox.showinfo("Success", f"Attendance marked for {user_name}")

            cv2.putText(img, str(user_name), (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(img, confidence_text, (x + 5, y + h - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 1)

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tk = ImageTk.PhotoImage(Image.fromarray(img_rgb))
        camera_label.imgtk = img_tk
        camera_label.configure(image=img_tk)
        camera_label.after(10, update_frame)

    update_frame()
    recognition_window.protocol("WM_DELETE_WINDOW", lambda: cam.release())

# Function to mark attendance and save to CSV 
def mark_attendance(user_name, user_id):
    today_str = str(datetime.datetime.now().date())
    filename = f"attendance_{today_str}.csv"
    if not os.path.exists(filename):
        with open(filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['User Name', 'User ID', 'Date', 'Time'])
    with open(filename, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([user_name, user_id, today_str, datetime.datetime.now().time()])
    
    # Send attendance via email
    send_email(user_name, user_id, today_str, datetime.datetime.now().time())

# Function to send email
def send_email(user_name, user_id, date, time):
    try:
        # Email configuration
        sender_email =  "vikas105106@gmail.com"  # Replace with your email
        sender_password = "qzhgjvvdlqgorzsx"      # Replace with your email password
        receiver_email = "goyalnavya2003@gmail.com"  # Replace with the recipient's email

        # Create the email content
        subject = "Attendance Confirmation"
        body = f"Attendance has been marked for:\n\nUser Name: {user_name}\nUser ID: {user_id}\nDate: {date}\nTime: {time}"

        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = receiver_email
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))

        # Connect to the SMTP server and send the email
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, receiver_email, msg.as_string())
        
        print(f"Email sent successfully to {receiver_email}")
    except Exception as e:
        print(f"Failed to send email: {e}")

# Main window
window = tk.Tk()
window.title("GL BAjAJ")
window.geometry("600x400")
window.configure(bg='#ADD8E6')

header_label = tk.Label(window, text="GL BAjAJ", font=("Helvetica", 24), fg="#FFFFFF", bg="#4682B4")
header_label.pack(fill="x")

register_button = tk.Button(window, text="Register New User", font=("Helvetica", 14), bg="#4682B4", fg="#FFFFFF", command=register_user)
register_button.pack(pady=20)

recognize_button = tk.Button(window, text="Recognize & Mark Attendance", font=("Helvetica", 14), bg="#4682B4", fg="#FFFFFF", command=recognize_face_window)
recognize_button.pack(pady=20)

window.mainloop()

