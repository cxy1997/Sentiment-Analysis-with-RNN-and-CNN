#coding=UTF-8
from __future__ import print_function, division
try:
    import xml.etree.cElementTree as ET 
except ImportError:
    import xml.etree.ElementTree as ET
from Tkinter import *

groupID = 2
annotation_start = 4000
annotation_end = 5999
cn_dataset_path = 'dataset/cn/'

def annoteStart():
    groupID = int(input_0.get())
    annotation_start = int(input_1.get())
    annotation_end = int(input_2.get())
    
    if 0 <= groupID <= 4 and 2000 * groupID <= annotation_start < 2000 * (groupID + 1) and 2000 * groupID <= annotation_end < 2000 * (groupID + 1) and annotation_start < annotation_end:
        prompt_0.place_forget()
        entry_0.place_forget()
        prompt_1.place_forget()
        entry_1.place_forget()
        prompt_2.place_forget()
        entry_2.place_forget()
        startButton.place_forget()

        global xmltree, xmlroot, now, positive_comments, negative_comments, text
        xmltree = ET.parse(cn_dataset_path + "cn_comments.xml")
        xmlroot = xmltree.getroot()
        now = annotation_start
        positive_comments = []
        negative_comments = []

        def terminate():
            with open('cn_positive_%d.xml' % groupID, 'w') as f_p:
                f_p.write('<reviews>\n')
                for c in positive_comments:
                    f_p.write(('<review>\n%s\n</review>\n' % xmlroot[c].text[1:-1]).encode("utf-8"))
                f_p.write('</reviews>\n')
            with open('cn_negative_%d.xml' % groupID, 'w') as f_n:
                f_n.write('<reviews>\n')
                for c in negative_comments:
                    f_n.write(('<review>\n%s\n</review>\n' % xmlroot[c].text[1:-1]).encode("utf-8"))
                f_n.write('</reviews>\n')
            root.destroy()

        def positiveClick():
            global now
            positive_comments.append(now)
            now += 1
            if now <= annotation_end:
                text.set(xmlroot[now].text[1:-1])
            else:
                terminate()
        def neutralClick():
            global now
            now += 1
            if now <= annotation_end:
                text.set(xmlroot[now].text[1:-1])
            else:
                terminate()
        def negativeClick():
            global now
            negative_comments.append(now)
            now += 1
            if now <= annotation_end:
                text.set(xmlroot[now].text[1:-1])
            else:
                terminate()
        def rollbackClick():
            global now
            if now > annotation_start:
                now -= 1
                if positive_comments != [] and positive_comments[-1] == now:
                    positive_comments.pop()
                if negative_comments != [] and negative_comments[-1] == now:
                    negative_comments.pop()
            text.set(xmlroot[now].text[1:-1])
            
        text = StringVar()
        text.set(xmlroot[now].text[1:-1])
        label = Label(canvas, textvariable = text, width = 40, height = 10, wraplength=600, justify='center', font = ("宋体", 24, 'normal'))
        label.place(x = 200, y = 150)
        positiveButton = Button(canvas, text = '好评', font = ("宋体", 50, 'normal'), command = positiveClick)
        positiveButton.place(x = 237, y = 530)
        neutralButton = Button(canvas, text = '中立', font = ("宋体", 50, 'normal'), command = neutralClick)
        neutralButton.place(x = 432, y = 530)
        negativeButton = Button(canvas, text = '差评', font = ("宋体", 50, 'normal'), command = negativeClick)
        negativeButton.place(x = 627, y = 530)
        rollbackButton = Button(canvas, text = '撤销', font = ("宋体", 30, 'normal'), command = rollbackClick)
        rollbackButton.place(x = 467, y = 660)

root = Tk()
root.title("Annotation")
root.geometry("1024x768")
canvas = Canvas(root, height = 768, width = 1024, bg = 'cyan')
canvas.pack()

prompt_0 = Label(canvas, text = '请输入组号(0-4):', bg = 'cyan', font = ("宋体", 14, 'normal'))
prompt_0.place(x = 447, y = 210)
input_0 = StringVar()
input_0.set(str(groupID))
entry_0 = Entry(canvas, textvariable = input_0)
entry_0.place(x = 450, y = 240)
prompt_1 = Label(canvas, text = '标注起始编号:', bg = 'cyan', font = ("宋体", 14, 'normal'))
prompt_1.place(x = 247, y = 360)
input_1 = StringVar()
input_1.set(str(annotation_start))
entry_1 = Entry(canvas, textvariable = input_1)
entry_1.place(x = 250, y = 390)
prompt_2 = Label(canvas, text = '标注结束编号:', bg = 'cyan', font = ("宋体", 14, 'normal'))
prompt_2.place(x = 647, y = 360)
input_2 = StringVar()
input_2.set(str(annotation_end))
entry_2 = Entry(canvas, textvariable = input_2)
entry_2.place(x = 650, y = 390)
startButton = Button(canvas, text = '开始标注', command = annoteStart)
startButton.place(x = 487, y = 490)
root.mainloop()
