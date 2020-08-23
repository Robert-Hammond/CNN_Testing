import tkinter as tk
from tkinter import *
from PIL import Image, ImageTk
import os
import random

category = 'sides'

ink_color = (255, 255, 255)
example_img = None
total_symbols = 0
# data variables
grid = [0 for j in range(28*28)]
current_symbol = 1
num_categories = 3

def get_magnitude(dist, max_dist):
    fade = 0.6
    if dist > max_dist:
        return 0
    if dist < max_dist*fade:
        return 1
    return 1./(1.-fade)-(dist/(max_dist*(1.-fade)))


def rgb2hex(r):
    return "#{:02x}{:02x}{:02x}".format(r[0], r[1], r[2])


def get_shade(magnitude):
    shade = (round(ink_color[0]*magnitude), round(ink_color[1]
                                                  * magnitude), round(ink_color[2]*magnitude))
    return rgb2hex(shade)


def draw_round(event):
    size = 6
    x1, y1 = (event.x - size), (event.y - size)
    x2, y2 = (event.x + size), (event.y + size)
    c.create_oval(x1, y1, x2, y2, fill=rgb2hex(
        ink_color), outline=rgb2hex(ink_color))


def draw_square(event):
    size = 10
    max_dist = 1.2*size
    for x in range(28):
        for y in range(28):
            cx = size*x + size/2
            cy = size*y + size/2
            dist = pow(pow(event.x - cx, 2) + pow(event.y - cy, 2), 0.5)
            magnitude = get_magnitude(dist, max_dist)
            if magnitude > grid[28*y + x]:
                grid[28*y + x] = magnitude
                color = get_shade(magnitude)
                c.create_rectangle(cx-size//2, cy-size//2, cx+size//2, cy+size//2,
                                   fill=color, outline=color)


def clear_canvas():
    c.delete("all")
    for i in range(len(grid)):
        grid[i] = 0


def num_files_in_dir(dir):
    cwd = os.getcwd()
    os.chdir(dir)
    s = len(os.listdir())
    while os.getcwd() != cwd:
        os.chdir('..')
    return s


def get_bounding_box():
    x1 = y1 = 28
    x2 = y2 = -1
    for x in range(28):
        for y in range(28):
            if grid[28*y + x] > 0:
                x1 = min(x, x1)
                y1 = min(y, y1)
                x2 = max(x, x2)
                y2 = max(y, y2)
    return (x1, y1, x2, y2)


def center_grid():
    (x1, y1, x2, y2) = get_bounding_box()
    dx = round(13.5 - ((x1 + x2) / 2.))
    dy = round(13.5 - ((y1 + y2) / 2.))
    if dy > 1:
        # shift canvas down
        for y in range(27, -1, -1):
            for x in range(28):
                if y <= dy - 1:
                    grid[28*y + x] = 0
                else:
                    grid[28*y + x] = grid[28*(y - dy) + x]
    elif dy < -1:
        # shift canvas up
        for y in range(28):
            for x in range(28):
                if y >= 28 + dy:
                    grid[28*y + x] = 0
                else:
                    grid[28*y + x] = grid[28*(y - dy) + x]
    if dx > 1:
        # shift canvas right
        for x in range(27, -1, -1):
            for y in range(28):
                if x <= dx - 1:
                    grid[28*y + x] = 0
                else:
                    grid[28*y + x] = grid[x - dx + 28*y]
    elif dx < -1:
        # shift canvas right
        for x in range(28):
            for y in range(28):
                if x >= 28 + dx:
                    grid[28*y + x] = 0
                else:
                    grid[28*y + x] = grid[x - dx + 28*y]


def update_example():
    global current_symbol
    global example_img
    if os.path.isdir('data/' + category + '/examples'):
        os.chdir('data/' + category + '/examples')
        if not os.path.isfile(str(current_symbol)+'.png'):
            os.chdir('../../..')
            return
        
        # find which image to show
        choices = [str(current_symbol)+'.png']
        additional = 0
        while os.path.isfile(str(current_symbol)+'.'+str(additional)+'.png'):
            choices.append(str(current_symbol)+'.'+str(additional)+'.png')
            additional += 1
        image_name = random.choice(choices)

        # show the example image
        example_img = ImageTk.PhotoImage(Image.open(image_name))
        e.create_image(0,0, anchor=NW, image=example_img)
        message.config(text=('Draw symbol no. '+str(current_symbol))+'\t\t\t\t\texample:\t\t')
        os.chdir('../../..')
    
        

def submit():
    global current_symbol
    global num_categories
    global total_symbols

    center_grid()

    # grid values are between 0 and 1, so expand this to between 0 and 255
    for i in range(len(grid)):
        grid[i] = round(grid[i] * 255)
    if sum(grid) < 255:
        clear_canvas()
        return
    # create image
    img = Image.new('L', (28, 28))
    img.putdata(grid)
    # navigate to or create appropriate directories
    if not os.path.isdir('data/' + category + '/images'):
        os.makedirs('data/' + category + '/images')
    os.chdir('data/' + category + '/images')
    if not os.path.isdir(str(current_symbol)):
        os.makedirs(str(current_symbol))
    # save the file
    symbol_count = num_files_in_dir(str(current_symbol))
    img.save(str(current_symbol) + '/' + str(current_symbol) + '.' + str(symbol_count) + '.jpg')
    os.chdir('../../..')
    # update the current symbol
    current_symbol += 1
    if current_symbol > num_categories:
        current_symbol = 1
    total_symbols += 1

    clear_canvas()
    update_example()
    progress_message.config(text=(f'Total:   {total_symbols}'))


def keydown(e):
    pass

def keyup(e):
    if e.char == 'c':
        clear_canvas()
    elif e.char == 'q':
        quit()
    elif e.char == '\r' or e.char == ' ':
        submit()

def get_total_symbols():
    if not os.path.isdir('data/' + category + '/images'):
        return 0

    os.chdir('data/' + category + '/images')
    count = 0
    for i in range(1, num_categories+1):
        if os.path.isdir(str(current_symbol)):
            count += num_files_in_dir(str(current_symbol))
    os.chdir('../../..')
    return count

root = Tk()
root.geometry('700x350')
root.title(category)


c = Canvas(root, height=280, width=280, bg='black')
c.place(x=170, y=187, anchor=CENTER)
c.bind("<B1-Motion>", draw_square)

e = Canvas(root, height = 280, width = 280)
e.place(x=670, y=47, anchor=NE)

frame = tk.Frame(root)
frame.bind("<KeyPress>", keydown)
frame.bind("<KeyRelease>", keyup)
frame.pack()
frame.focus_set()

exit_button = tk.Button(frame, text="EXIT", fg="red", command=quit)
exit_button.pack(side=tk.LEFT)
clear_button = tk.Button(frame, text="clear", command=clear_canvas)
clear_button.pack(side=tk.RIGHT)
submit_button = tk.Button(frame, text="submit", command=submit)
submit_button.pack(side=tk.LEFT)

total_symbols = get_total_symbols()

message = tk.Label(root)
message.pack()
progress_message = tk.Label(root)
progress_message.place(relx = 0.1, y = 330)
update_example()

root.mainloop()
