from tkinter import *
from scipy import fft
from tkinter import filedialog, ttk
from tkinter.filedialog import askopenfile
from PIL import ImageTk, Image
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np

BACKGROUND_COLOR = "#152028"  # 121212
FONT_NAME = "Roboto"
FILE_NAMES = os.listdir("img")
IMG = None


def eq_hist(image):
    """
        Gray transformation from low light to enhanced light
    """
    chosen_method = radio_variable.get()
    if chosen_method == "log":
        c = 255 / np.log(1 + np.max(image))  # logarithmic transformation method
        enhanced_image = c * np.log1p(image)
        return enhanced_image.astype(np.uint8)
    elif chosen_method == "gamma":
        gamma = 0.3
        enhanced_image = np.power(image / 255.0, gamma)  # gamma gray transformation method
        enhanced_image = (enhanced_image * 255).astype(np.uint8)
        return enhanced_image
    elif chosen_method == "he":
        histogram, bins = np.histogram(image.flatten(), bins=256, range=[0, 256])  # histogram equalization
        cdf = histogram.cumsum()
        cdf_normalized = cdf * 255 / cdf[-1]
        equalized_image = np.interp(image.flatten(), bins[:-1], cdf_normalized)
        return equalized_image.reshape(image.shape).astype(np.uint8)
    else:
        print("Error on the chosen method")


def eq_hist_color(colored_image):
    """
        Equalize each RGB channel of the image for color enhancing
    :param colored_image:
    :return:
    """
    rgb_image = cv2.cvtColor(colored_image, cv2.COLOR_BGR2RGB)
    for i in range(3):
        rgb_image[:, :, i] = eq_hist(rgb_image[:, :, i])
    result = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    return result


def apply_Butterworth_filter(image, directory, img_name):
    """
        Below is the 2-D Butterworth filter
    """
    H, W = image.shape
    PrcKept = 0.10  # 10%
    DPI = 72
    FiltOrder = 1

    D = np.sqrt(W ** 2 + H ** 2)
    w_0 = PrcKept * D
    X = np.arange(0, W)
    Y = np.arange(0, H)
    XCoords, YCoords = np.meshgrid(X, Y)
    CoeffsMask = (1.0 / (
            1.0 + ((np.square(XCoords).astype(float) + np.square(YCoords).astype(float)) ** FiltOrder) / w_0 ** (
            2 * FiltOrder))).astype(float)

    InImgDCT = fft.dctn(image, norm="ortho")
    FiltImgDCT = np.multiply(InImgDCT, CoeffsMask)
    OutImgFilt = (fft.idctn(FiltImgDCT, norm="ortho")).astype(np.uint8)

    plt.figure(figsize=(2.3 * W / DPI + 1, 2.3 * H / DPI + 1))
    plt.subplot(2, 2, 1)
    plt.imshow((CoeffsMask * 255).astype(np.uint8), cmap='gray', vmin=0, vmax=255)
    plt.title('The filtering mask')
    plt.subplot(2, 2, 2)
    plt.imshow(np.clip(np.abs(FiltImgDCT), 0, 255), cmap='rainbow', vmin=0, vmax=255)
    plt.title('The filtered coefficients')
    plt.subplot(2, 2, 3)
    plt.imshow(image, cmap='gray', vmin=0, vmax=255)
    plt.title('The gray enhanced image')
    plt.subplot(2, 2, 4)
    plt.imshow(np.clip(OutImgFilt, 0, 255), cmap='gray', vmin=0, vmax=255)
    plt.title('The filtering result')
    plt.savefig(f'img/results/{directory}/plots_{img_name}')
    noiseless_gray_image = np.clip(OutImgFilt, 0, 255)
    cv2.imwrite(f'img/results/{directory}/noiseless_gray_{img_name}', noiseless_gray_image)


def exit_app():
    window.quit()


def upload_file():
    """
    This function is for choosing an image to be uploaded
    :return:
    """
    global IMG
    IMG = filedialog.askopenfile()
    chosen_path_image = IMG.name
    IMG = chosen_path_image
    # image = cv2.imread(chosen_path_image)
    begin_protocol()


def set_fullscreen():
    """
    Sets the application in fullscreen mode
    :return:
    """
    global window
    window.attributes('-fullscreen', fullscreen_var.get())
    if fullscreen_var.get():
        fullscreen_check.config(text="Exit fullscreen mode")
    else:
        fullscreen_check.config(text="Set fullscreen mode")


def begin_protocol():
    """
    The protocol after user chose the image
    :return:
    """
    # img_name = item
    if IMG is None:
        img_name = img_input.get()
    else:
        img_name = IMG
        nr_of_slashes = img_name.count('/')
        img_real_name = img_name.split('/')[nr_of_slashes]
        print(img_real_name)
    # img_name = f'ex2.jpg'

    if img_name not in FILE_NAMES and IMG is None:
        error_label_radio.config(text="")
        error_label_input.config(text="Please enter the file name located in /img directory!")
        print("File does not exist, try again")
    elif radio_variable.get() == "":
        error_label_input.config(text="")
        error_label_radio.config(text="Please choose a transformation method!")
        print("Please choose a transformation method")
    else:
        try:
            error_label_input.config(text="")
            error_label_radio.config(text="")
        except:
            pass
        if IMG is None:
            directory = f'{img_name}'
        else:
            directory = f'{img_real_name}'
        parent_dir = f"{os.path.dirname(os.path.abspath(__file__))}\\img\\results"
        path = os.path.join(parent_dir, directory)
        try:
            os.mkdir(path)
        except Exception as e:
            print(e)
            # os.remove(path)
            # os.mkdir(path)
        title.config(text="Result")
        nume_fisier.config(text="")

        if IMG is None:
            original_image = cv2.imread(f'img/{img_name}')
            gray_image = cv2.imread(f'img/{img_name}', 0)
        else:
            original_image = cv2.imread(f'{img_name}')
            gray_image = cv2.imread(f'{img_name}', 0)
            img_name = img_real_name
        print(img_name)
        cv2.imwrite(f'img/results/{directory}/original_{img_name}', original_image)
        cv2.imwrite(f'img/results/{directory}/gray_{img_name}', gray_image)
        cv2.imwrite(f'img/results/{directory}/enhanced_gray_{img_name}', eq_hist(gray_image))
        cv2.imwrite(f'img/results/{directory}/enhanced_original_{img_name}', eq_hist_color(original_image))
        # apply_Butterworth_filter(eq_hist(gray_image), directory, img_name)

        H, W = gray_image.shape
        DPI = 100

        plt.figure(figsize=(2.3 * W / DPI + 1, 2.3 * H / DPI + 1))
        plt.subplot(2, 2, 1)
        plt.hist(gray_image.ravel(), 256, [0, 255], color="red")
        plt.title(f'gray_{img_name} histogram')
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')

        plt.subplot(2, 2, 2)
        plt.hist(eq_hist(gray_image).ravel(), 256, [0, 255], color="red")
        plt.title(f'enhanced_gray_{img_name} histogram')
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')

        hist1 = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
        cumulative_hist = np.cumsum(hist1)
        plt.subplot(2, 2, 3)
        plt.plot(cumulative_hist, color="black")
        plt.title(f'gray_{img_name} cumulative histogram')
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')

        hist2 = cv2.calcHist([eq_hist(gray_image)], [0], None, [256], [0, 256])
        cumulative_hist = np.cumsum(hist2)
        plt.subplot(2, 2, 4)
        plt.plot(cumulative_hist, color="black")
        plt.title(f'enhanced_gray_{img_name} cumulative histogram')
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')

        plt.savefig(f'img/results/{directory}/histogram_enhanced_gray_{img_name}')

        select_enhance_method.pack_forget()
        log_button.pack_forget()
        gamma_button.pack_forget()
        histogram_button.pack_forget()

        # resulted_image_gray = ImageTk.PhotoImage(Image.open(f'img/results/{directory}/enhanced_gray_{img_name}'))
        # canvas = Canvas(width=300, height=200, bg=BACKGROUND_COLOR)
        # resulted_image_gray = PhotoImage(file=f'img/results/{directory}/enhanced_original_{img_name}')
        # canvas.create_image(100, 112, image=resulted_image_gray)
        # canvas.pack()

        image_path = f"img/results/{directory}/enhanced_original_{img_name}"
        print(image_path)
        pil_image = Image.open(image_path)
        # custom_width, custom_height = 400, 300
        # resized_image = pil_image.resize((custom_width, custom_height), Image.ANTIALIAS)
        # tk_image = ImageTk.PhotoImage(resized_image)
        tk_image = ImageTk.PhotoImage(pil_image)
        image_label = Label(image=tk_image)
        image_label.pack()
        image_label.image = tk_image


window = Tk()
window.title("Low Light Image Enhancement")
screen_width = window.winfo_screenwidth()
screen_height = window.winfo_screenheight()
print(screen_height, screen_width)
window.config(padx=7, pady=3, bg=BACKGROUND_COLOR)
ICON = PhotoImage(file="icon/icon.png")
window.iconphoto(False, ICON)

title = Label(text="Low Light Image Enhancement Methods", font=(FONT_NAME, 30, "bold"), bg=BACKGROUND_COLOR,
              fg="white")
# title.pack(side=TOP, fill=X, expand=True)
title.pack()

select_enhance_method = Label(text="Select a method for enhancement", font=(FONT_NAME, 15), bg=BACKGROUND_COLOR,
                              fg="white")
select_enhance_method.pack()

radio_frame = Frame(window)
radio_frame.pack()
radio_variable = StringVar()
log_button = Radiobutton(radio_frame, text="Logarithmic method", variable=radio_variable, value="log",
                         bg=BACKGROUND_COLOR,
                         fg="white")
gamma_button = Radiobutton(radio_frame, text="Gamma method", variable=radio_variable, value="gamma",
                           bg=BACKGROUND_COLOR, fg="white")
histogram_button = Radiobutton(radio_frame, text="Histogram equalization method", variable=radio_variable, value="he",
                               bg=BACKGROUND_COLOR, fg="white")
# log_button.pack(side=LEFT)
# gamma_button.pack(side=LEFT)
# histogram_button.pack(side=LEFT)
log_button.grid(row=0, column=0)
gamma_button.grid(row=0, column=1)
histogram_button.grid(row=0, column=2)

nume_fisier = Label(text="Type below the name of the image", font=(FONT_NAME, 15), bg=BACKGROUND_COLOR, fg="white")
# nume_fisier.grid(column=1, row=1)
nume_fisier.pack()

img_input = Entry()
# img_input.grid(column=1, row=2, sticky=S, ipadx=10, padx=10)
img_input.pack()

run_button = Button(text="Run", font=FONT_NAME, command=begin_protocol)
# run_button.grid(column=1, row=3)
run_button.pack()

or_label = Label(text="or", font=(FONT_NAME, 15), bg=BACKGROUND_COLOR, fg="white")
or_label.pack()

chose_file = Button(text='Click to select a file', command=upload_file)
chose_file.pack()

fullscreen_var = IntVar()
fullscreen_check = Checkbutton(text="Set fullscreen mode", font=FONT_NAME, bg=BACKGROUND_COLOR, fg="white",
                               highlightthickness=0, variable=fullscreen_var, command=set_fullscreen)
fullscreen_check.pack(side=RIGHT)

exit_button = Button(text="Quit", font=FONT_NAME, highlightthickness=0, command=exit_app)
exit_button.pack(side=BOTTOM)

error_label_input = Label(text="", font=(FONT_NAME, 10), bg=BACKGROUND_COLOR, fg="red")
error_label_input.pack()

error_label_radio = Label(text="", font=(FONT_NAME, 10), bg=BACKGROUND_COLOR, fg="red")
error_label_radio.pack()

window.mainloop()
