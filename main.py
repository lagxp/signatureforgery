import tkinter as tk   # For creating the graphical user interface (GUI)
from tkinter.filedialog import askopenfilename # For file selection dialog
from tkinter import messagebox   # For displaying message boxes
import os   # For operating system-related operations
import cv2  # For image processing using OpenCV
from PIL import Image, ImageTk   # For working with images
from skimage.metrics import structural_similarity as ssim  # For SSIM-based image comparison
import csv  # For handling CSV file operations
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input # For VGG16 model
from tensorflow.keras.models import Model  # For building deep learning models
import numpy as np  # For numerical operations and arrays
import matplotlib.pyplot as plt # Import the Matplotlib library for plotting and visualization
import subprocess # Import the subprocess module for executing external commands
from tkinter.filedialog import askopenfilename, askdirectory # Import the askopenfilename and askdirectory functions from the tkinter.filedialog module


# New function to write data to CSV file
def write_to_csv(path1, path2, similarity_percentage):
    file_exists = os.path.isfile('performance_data.csv')
    with open('performance_data.csv', 'a', newline='') as csvfile:
        headers = ['Signature 1', 'Signature 2', 'Percentage Similarity']
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        if not file_exists:
            writer.writeheader()
        writer.writerow({'Signature 1': path1, 'Signature 2': path2, 'Percentage Similarity': similarity_percentage})

# Load pre-trained VGG16 model (without the top classification layer)
base_model = VGG16(weights='vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5', include_top=False, input_shape=(224, 224, 3))

def preprocess_image(image_path):
    try:
        image = Image.open(image_path)
        image = image.resize((224, 224))
        image = np.array(image)
        if len(image.shape) == 2:  # If the image is grayscale, convert it to RGB
            image = np.stack((image,) * 3, axis=-1)
        image = preprocess_input(image)
        image = np.expand_dims(image, axis=0)
        return image
    except Exception as e:
        messagebox.showerror("Error", f"Failed to open image: {e}")
        return None

# Add this function to calculate the similarity between two feature vectors
def calculate_similarity(features1, features2):
    # Perform similarity calculation based on your requirements
    # For example, you can use cosine similarity:
    similarity = np.dot(features1, features2) / (np.linalg.norm(features1) * np.linalg.norm(features2))
    similarity_percentage = round(similarity * 100, 2)
    return similarity_percentage


# New function for deep learning-based signature similarity
def deep_learning_match(path1, path2):
    img1 = preprocess_image(path1)
    img2 = preprocess_image(path2)

    if img1 is None or img2 is None:
        messagebox.showerror("Error", "Failed to preprocess images. Please check the image paths.")
        return None

    img1_features = base_model.predict(img1).flatten()
    img2_features = base_model.predict(img2).flatten()

    similarity = calculate_similarity(img1_features, img2_features)

    return similarity

def detect_objects_on_image(image_path, image_entry):
    net = cv2.dnn.readNet("yolov3.cfg", "yolov3.weights")
    classes = []
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]

    layer_names = net.getLayerNames()
    outputlayers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    image = cv2.imread(image_path)
    height, width, _ = image.shape

    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(outputlayers)

    signature_detected = False

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.3:
                label = classes[class_id]
                print(label)
                if label.lower() != 'signature':
                    messagebox.showerror("Error", f"Detected object '{label}' is not a signature image.")
                    image_entry.delete(0, tk.END)  # Clear the entry widget
                    return
                else:
                    signature_detected = True


def capture_image_from_cam_into_temp(sign=1):
    cam = cv2.VideoCapture(0)

    cv2.namedWindow("test")

    while True:
        ret, frame = cam.read()
        if not ret:
            print("Failed to grab frame")
            break
        cv2.imshow("test", frame)

        k = cv2.waitKey(1)
        if k == 27:  # Press ESC to exit
            print("Escape hit, closing...")
            break
        elif k == 32:  # Press SPACE to capture
            if not os.path.isdir('temp'):
                os.mkdir('temp', mode=0o777)  # make sure the directory exists
            if sign == 1:
                img_name = "./temp/test_img1.png"
            else:
                img_name = "./temp/test_img2.png"
            print('imwrite=', cv2.imwrite(filename=img_name, img=frame))
            print("{} written!".format(img_name))
            break  # Exit after capturing a single image

    cam.release()
    cv2.destroyAllWindows()
    return True

def captureImage(ent, sign=1):
    if(sign == 1):
        filename = os.getcwd()+'\\temp\\test_img1.png'
    else:
        filename = os.getcwd()+'\\temp\\test_img2.png'
    # messagebox.showinfo(
    #     'SUCCESS!!!', 'Press Space Bar to click picture and ESC to exit')
    res = None
    res = messagebox.askquestion(
        'Click Picture', 'Press Space Bar to click picture and ESC to exit')
    if res == 'yes':
        capture_image_from_cam_into_temp(sign=sign)
        ent.delete(0, tk.END)
        ent.insert(tk.END, filename)
    return True

    
def browsefunc(ent):
    file_path = askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg"), ("All files", "*.*")])
    if file_path:
        ent.delete(0, tk.END)
        ent.insert(tk.END, file_path)
        # Call detect_objects_on_image with the entry widget as an argument
        detect_objects_on_image(file_path, ent)

def browse_folder(ent):
    folder_path = askdirectory()
    if folder_path:
        ent.delete(0, tk.END)
        ent.insert(tk.END, folder_path)

def show_both_signature_images(path1, path2):
    if os.path.isfile(path2):
        show_signature_comparison(path1, path2)
    elif os.path.isdir(path2):
        folder_images = [os.path.join(path2, f) for f in os.listdir(path2) if os.path.isfile(os.path.join(path2, f))]
        highest_similarity = 0
        highest_similarity_image = None
        for image_path in folder_images:
            similarity = deep_learning_match(path1=path1, path2=image_path)
            if similarity is not None and similarity > highest_similarity:
                highest_similarity = similarity
                highest_similarity_image = image_path
        
        if highest_similarity_image is not None:
            show_signature_comparison(path1, highest_similarity_image)
        else:
            messagebox.showerror("Error", "No images found in the folder.")
    else:
        messagebox.showerror("Error", "Invalid path provided.")

def show_signature_comparison(path1, path2):
    img1 = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(path2, cv2.IMREAD_GRAYSCALE)

    if img1 is None or img2 is None:
        messagebox.showerror("Error", "Failed to load one or both images.")
        return

    plt.figure(figsize=(10, 5))

    ax1 = plt.subplot(1, 2, 1)
    ax1.imshow(img1, cmap='gray')  # Use 'gray' colormap for grayscale
    ax1.set_title("Signature 1")
    ax1.set_xlabel("X-axis")
    ax1.set_ylabel("Y-axis")
    ax1.grid(True)  # Display gridlines

    ax2 = plt.subplot(1, 2, 2)
    ax2.imshow(img2, cmap='gray')  # Use 'gray' colormap for grayscale
    ax2.set_title("Signature 2")
    ax2.set_xlabel("X-axis")
    ax2.set_ylabel("Y-axis")
    ax2.grid(True)  # Display gridlines

    plt.tight_layout()
    plt.show()

# Mach Threshold
THRESHOLD = 70

def checkSimilarity(window, path1, path2):
    if os.path.isfile(path2):
        result = deep_learning_match(path1=path1, path2=path2)
        if result is not None:
            if result <= THRESHOLD:
                messagebox.showerror("Failure: Signatures Do Not Match",
                                     "Signatures are " + str(result) + f" % similar!!")
            else:
                messagebox.showinfo("Success: Signatures Match",
                                    "Signatures are " + str(result) + f" % similar!!")
            write_to_csv(path1, path2, result)
        return
    elif os.path.isdir(path2):
        folder_images = [os.path.join(path2, f) for f in os.listdir(path2) if os.path.isfile(os.path.join(path2, f))]

        image1_features = base_model.predict(preprocess_image(path1)).flatten()
        folder_images_features = [base_model.predict(preprocess_image(image_path)).flatten() for image_path in folder_images]

        similarity_scores = [calculate_similarity(image1_features, feature) for feature in folder_images_features]
        highest_similarity_index = np.argmax(similarity_scores)

        if highest_similarity_index is not None:
            highest_similarity = similarity_scores[highest_similarity_index]
            highest_similarity_image = folder_images[highest_similarity_index]
            
            if highest_similarity <= THRESHOLD:
                messagebox.showerror("Failure: Signatures Do Not Match",
                                     "Highest similarity found with " + os.path.basename(highest_similarity_image) + f", {highest_similarity} % similar!!")
            else:
                messagebox.showinfo("Success: Signatures Match",
                                    "Highest similarity found with " + os.path.basename(highest_similarity_image) + f", {highest_similarity} % similar!!")
            write_to_csv(path1, highest_similarity_image, highest_similarity)
        else:
            messagebox.showerror("Error", "No images found in the folder.")
        return
    else:
        messagebox.showerror("Error", "Invalid path provided.")
        return




def checkForgery(window, original_path, forgery_path):
    original_image = cv2.imread(original_path, cv2.IMREAD_GRAYSCALE)
    if original_image is None:
        messagebox.showerror("Error", "Failed to preprocess images. Please check the image paths.")
        return False

    if os.path.isfile(forgery_path):
        forgery_image = cv2.imread(forgery_path, cv2.IMREAD_GRAYSCALE)
        if forgery_image is None:
            messagebox.showerror("Error", "Failed to preprocess images. Please check the image paths.")
            return False

        # Resize images to the same dimensions for accurate comparison
        original_image = cv2.resize(original_image, (forgery_image.shape[1], forgery_image.shape[0]))

        # Calculate SSIM similarity between the two images
        similarity = ssim(original_image, forgery_image)

        forgery_threshold = 0.8  # You can adjust this threshold as needed
        if similarity < forgery_threshold:
            messagebox.showerror("Forgery Detected", "The signature appears to be a forgery!")
        else:
            messagebox.showinfo("Forgery Not Detected", "The signature seems genuine.")
        write_to_csv(original_path, forgery_path, similarity)
        return True
    elif os.path.isdir(forgery_path):
        folder_images = [os.path.join(forgery_path, f) for f in os.listdir(forgery_path) if os.path.isfile(os.path.join(forgery_path, f))]

        highest_similarity = 0
        highest_similarity_image = None
        for image_path in folder_images:
            forgery_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if forgery_image is None:
                messagebox.showerror("Error", f"Failed to preprocess image: {image_path}")
                continue

            # Resize images to the same dimensions for accurate comparison
            original_image_resized = cv2.resize(original_image, (forgery_image.shape[1], forgery_image.shape[0]))

            # Calculate SSIM similarity between the two images
            similarity = ssim(original_image_resized, forgery_image)

            if similarity > highest_similarity:
                highest_similarity = similarity
                highest_similarity_image = image_path

        if highest_similarity_image is not None:
            forgery_threshold = 0.8  # You can adjust this threshold as needed
            if highest_similarity < forgery_threshold:
                messagebox.showerror("Forgery Detected", f"The signature appears to be a forgery with {os.path.basename(highest_similarity_image)}!")
            else:
                messagebox.showinfo("Forgery Not Detected", f"The signature seems genuine with {os.path.basename(highest_similarity_image)}!")
            write_to_csv(original_path, highest_similarity_image, highest_similarity)
        else:
            messagebox.showerror("Error", "No images found in the folder.")
        return True
    else:
        messagebox.showerror("Error", "Invalid path provided.")
        return False



def detect_attendance(window, path):
    # Specify the path to your attendance.py script
    attendance_script_path = "C:/Users/Mukesh/OneDrive/Desktop/Signature-Matching-main/Signature-Matching-main/attendance signature/attendance.py"

    # Check if the attendance script exists
    if os.path.exists(attendance_script_path):
        # Use subprocess to execute the attendance.py script
        subprocess.Popen(["python", attendance_script_path])
    else:
        messagebox.showerror("Error", "attendance.py script not found.")
def login():
    # Replace 'username' and 'password' with the actual credentials for login
    username = "karan"
    password = "123456"

    entered_username = username_entry.get()
    entered_password = password_entry.get()

    if entered_username == username and entered_password == password:
        # Destroy the login window and show the main application window
        login_window.destroy()
        root.deiconify()  # Show the main application window
    else:
        messagebox.showerror("Login Failed", "Invalid username or password!")

def show_password():
    if password_entry["show"] == "":
        password_entry["show"] = "*"
    else:
        password_entry["show"] = ""

def show_login_page():
    global login_window, username_entry, password_entry
    login_window = tk.Toplevel()
    login_window.title("Login")

    # Set the window size
    width = 300
    height = 300

    # Center the window on the screen
    center_window(login_window, width, height)

    # Attempt to load the background image
    try:
        bg_image = Image.open("C:/Users/Mukesh/OneDrive/Desktop/Signature-Matching-main/Signature-Matching-main/logo/loginpage4.jpg")
        bg_photo = ImageTk.PhotoImage(bg_image)
        bg_label = tk.Label(login_window, image=bg_photo)
        bg_label.image = bg_photo  # Keep a reference to the image to avoid garbage collection
        bg_label.place(x=0, y=0, relwidth=1, relheight=1)
    except Exception as e:
        print("Error loading background image:", e)
        login_window.configure(bg="lightgray")  # Set a default background color if the image fails to load

    # Create a frame for the login elements with a transparent background
    frame = tk.Frame(login_window, bg="black", bd=5)
    frame.place(relx=0.5, rely=0.5, anchor="center")

    login_label = tk.Label(frame, text="Login", font=("Arial", 16))
    login_label.pack(pady=10)

    username_label = tk.Label(frame, text="Username:", font=10)
    username_label.pack(anchor="w", padx=5, pady=5)

    username_entry = tk.Entry(frame, font=10)
    username_entry.pack(anchor="w", padx=5, pady=5)

    password_label = tk.Label(frame, text="Password:", font=10)
    password_label.pack(anchor="w", padx=5, pady=5)

    password_entry = tk.Entry(frame, font=10, show="*")
    password_entry.pack(anchor="w", padx=5, pady=5)

    show_password_button = tk.Button(frame, text="Show Password", font=10, command=show_password, bg='blue', activebackground='darkblue', fg='white', padx=5, pady=2)
    show_password_button.pack(side=tk.LEFT, padx=5, pady=5)

    login_button = tk.Button(frame, text="Login", font=10, command=login, bg='blue', activebackground='darkblue', fg='white', padx=10, pady=5)
    login_button.pack(side=tk.RIGHT, padx=10, pady=5)

    # Hide the login window when the main application window is displayed
    login_window.protocol("WM_DELETE_WINDOW", root.destroy)

    # Start the login window's main loop
    login_window.mainloop()


def open_file():
    file_path = 'performance_data.csv'
    if os.path.exists(file_path):
        try:
            os.startfile(file_path)  # This will open the file with the default program
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open the file: {e}")
    else:
        messagebox.showerror("File Not Found", "The performance_data.csv file does not exist.")

def exit_application():
    if messagebox.askokcancel("Exit", "Are you sure you want to exit?"):
        root.destroy()
def center_window(window, width, height):
    # Get the screen width and height
    screen_width = window.winfo_screenwidth()
    screen_height = window.winfo_screenheight()

    # Calculate the x and y coordinates to center the window
    x = (screen_width - width) // 2
    y = (screen_height - height) // 2

    # Set the geometry of the window to the center of the screen
    window.geometry(f"{width}x{height}+{x}+{y}")

def main():
    global root
    root = tk.Tk()
    root.title("Signature Matching")

    # Set the window size
    width = 500
    height = 700

    # Center the window on the screen
    center_window(root, width, height)

    # Load the PNG logo
    logo_path = "logo2.png"
    logo_image = Image.open("C:/Users/Mukesh/OneDrive/Desktop/Signature-Matching-main/Signature-Matching-main/logo/logo4.png")
    logo_photo = ImageTk.PhotoImage(logo_image)
    logo_label = tk.Label(root, image=logo_photo)
    logo_label.pack()

    uname_label = tk.Label(root, text="Compare Two Signatures:", font=10)
    uname_label.place(x=90, y=50)

    image1_path_entry = tk.Entry(root, font=10)
    image1_path_entry.place(x=150, y=120)
    img1_message = tk.Label(root, text="Signature 1", font=10)
    img1_message.place(x=10, y=120)

    img1_capture_button = tk.Button(
        root, text="Capture", font=10, command=lambda: captureImage(ent=image1_path_entry, sign=1))
    img1_capture_button.place(x=400, y=90)
    img1_capture_button.configure(bg='green', activebackground='darkgreen')

    img1_browse_button = tk.Button(
        root, text="Browse", font=10, command=lambda: browsefunc(ent=image1_path_entry))
    img1_browse_button.place(x=400, y=140)
    img1_browse_button.configure(bg='orange', activebackground='darkorange')

    image2_path_entry = tk.Entry(root, font=10)
    image2_path_entry.place(x=150, y=240)
    img2_message = tk.Label(root, text="Signature 2", font=10)
    img2_message.place(x=10, y=250)

    img2_capture_button = tk.Button(
        root, text="Capture", font=10, command=lambda: captureImage(ent=image2_path_entry, sign=2))
    img2_capture_button.place(x=400, y=210)
    img2_capture_button.configure(bg='green', activebackground='darkgreen')

    img2_browse_button = tk.Button(
        root, text="Browse Image", font=10, command=lambda: browsefunc(ent=image2_path_entry))
    img2_browse_button.place(x=400, y=260)
    img2_browse_button.configure(bg='orange', activebackground='darkorange')

    folder_browse_button = tk.Button(
        root, text="Browse Folder", font=10, command=lambda: browse_folder(ent=image2_path_entry))
    folder_browse_button.place(x=400, y=300)
    folder_browse_button.configure(bg='orange', activebackground='darkorange')

    compare_button = tk.Button(
        root, text="Compare", font=10, command=lambda: checkSimilarity(window=root,
                                                                       path1=image1_path_entry.get(),
                                                                       path2=image2_path_entry.get(),))
    compare_button.place(x=200, y=320)
    compare_button.configure(bg='yellow', activebackground='lightyellow')

    forgery_check_button = tk.Button(
        root, text="Check Forgery", font=10,
        command=lambda: checkForgery(window=root,
                                     original_path=image1_path_entry.get(),
                                     forgery_path=image2_path_entry.get())
    )
    forgery_check_button.place(x=200, y=380)
    forgery_check_button.configure(bg='red', activebackground='red')

    show_both_img_button = tk.Button(
        root, text="Show Signature Images", font=10,
        command=lambda: show_both_signature_images(path1=image1_path_entry.get(), path2=image2_path_entry.get())
    )
    show_both_img_button.place(x=200, y=420)
    show_both_img_button.configure(bg='blue', activebackground='darkblue')
    attendance_button = tk.Button(
        root, text="Attendance Detection", font=10,
        command=lambda: detect_attendance(window=root, path=image1_path_entry.get())
    )
    attendance_button.place(x=200, y=480)
    attendance_button.configure(bg='green', activebackground='darkgreen')

    open_file_button = tk.Button(root, text="Open File", font=10, command=open_file, bg='yellow', activebackground='yellow', fg='black', padx=10, pady=5)
    open_file_button.place(x=50, y=550)

    exit_button = tk.Button(root, text="Exit", font=10, command=exit_application, bg='yellow', activebackground='yellow', fg='black', padx=10, pady=5)
    exit_button.place(x=420, y=20)

    # Show the login page initially and hide the main application window
    root.withdraw()  # Hide the main application window
    show_login_page()

    root.mainloop()

if __name__ == "__main__":
    main()
