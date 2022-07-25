from PIL import Image
from PIL import ImageTk
from PIL import ImageFilter
from PIL import ImageEnhance

import tkinter as tk  # Tkinter: python内置GUI
from tkinter.filedialog import askopenfilename
import tkinter.ttk
import tkinter.messagebox
from tkinter import simpledialog

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import collections

#import os
#os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

class picture(object):
    """description of class"""
    # 打开图像调用
    def open_pic(self, address):
        self.pic_get = Image.open(address).convert('RGBA')
        wid, hei = self.pic_get.size
        if wid > 600 or hei > 400:
            if tk.messagebox.askokcancel('提示', '图片可能过大,是否压缩?'):
                needShow_pic = self.openResize()
                return needShow_pic
            return self.pic_get
        else:
            return self.pic_get

    # 打开图像时的图像压缩展示
    def openResize(self):
        w, h = self.pic_get.size
        w_hope = 500
        h_hope = 300
        f1 = 1.0 * w_hope / w
        f2 = 1.0 * h_hope / h
        factor = min([f1, f2])
        width = int(w * factor)
        height = int(h * factor)
        pic_show = self.pic_get.resize((width, height))
        return pic_show
 
# 截图处理
    def Cutpic(self, pic_preCut, p1, p2, p3, p4):
        cropped_pic = pic_preCut.crop((p1, p2, p3, p4))#截图
        return cropped_pic

# 尺寸大小变化
    def changeResize(self, pic_reshow, newWidth, newHeight):
        reesizeNew_pic = pic_reshow.resize((newWidth, newHeight))#修改尺寸
        # print('3')
        return reesizeNew_pic


# 镜像左右
    def MirrorPic_leftOrright(self, pic_mir_lr):
        Mirror_lrFinish = pic_mir_lr.transpose(Image.Transpose.FLIP_LEFT_RIGHT)# 镜像左右
        return Mirror_lrFinish

# 镜像上下
    def MirrorPic_topOrbuttom(self, pic_mir_tp):
        Mirror_tbFinish = pic_mir_tp.transpose(Image.Transpose.FLIP_LEFT_RIGHT)# 镜像上下
        return Mirror_tbFinish
# 旋转
    def rotatePic(self, pic_prerotate, rodegreee):
        rotateNew_pic = pic_prerotate.rotate(rodegreee, expand=True)
        return rotateNew_pic
# 亮度
    def brightPic(self, pic_prebright, n):
        pic_brighted = ImageEnhance.Brightness(pic_prebright).enhance(n)# 亮度
        return pic_brighted

# 色彩度
    def colornPic(self, pic_preColor, n):
        pic_colored = ImageEnhance.Color(pic_preColor).enhance(n)# 色彩度
        return pic_colored

# 对比度
    def constractPic(self, pic_preCon, n):
        enh_con = ImageEnhance.Contrast(pic_preCon)
        contrast = n
        pic_contrasted = enh_con.enhance(contrast)# 对比度
        return pic_contrasted

# 锐度调整
    def sharpPic(self, pic_preSharp, n):
        pic_sharped = ImageEnhance.Sharpness(pic_preSharp).enhance(n)
        return pic_sharped


class modules(object):

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std


    """description of class"""
    # 加了滤镜的拓展功能，用的ImageFilter库
    def blurPic(Imf):
        Im2 = Imf.filter(ImageFilter.BLUR)  # 图像模糊
        return Im2

    def edge(Imf):
        Im4 = Imf.filter(ImageFilter.EDGE_ENHANCE)  # 边界增强
        return Im4

    def gaussianBlur(Imf):
        Im6 = Imf.filter(ImageFilter.GaussianBlur)  # 高斯模糊
        return Im6

    def emboss(Imf):
        Im8 = Imf.filter(ImageFilter.EMBOSS)  # 浮雕滤镜，
        return Im8

    # 线性灰度转换
    def linearization(Imf, a, c):
        Im12=np.array(Imf)
        r,g,b = Im12[:,:,0], Im12[:,:,1], Im12[:,:,2]
        Im12 = 0.2989*r + 0.5870*g + 0.1140*b
        Im12=float(a)*Im12+float(c) # 对矩阵类型计算,a是对比度，c是亮度，由k和b传入
        # 进行数据截断，大于255的值要截断为255
        Im12[Im12>255]=255
        # 数据类型转化
        Im12=np.round(Im12)
        Im12=Im12.astype(np.uint8)
        return Image.fromarray(Im12)
    # 非线性log灰度转换
    def tologpic(Imf, c):
        Im14=np.array(Imf)
        r,g,b = Im14[:,:,0], Im14[:,:,1], Im14[:,:,2]
        Im14 = 0.2989*r + 0.5870*g + 0.1140*b
        Im14 = c * np.log(1.0 + Im14) # 对数运算
        Im14[Im14>255]=255 #
        Im14 = np.uint8(Im14 + 0.5)
        return Image.fromarray(Im14)
    # n值灰度转换
    def tonpic(Imf, n):
        Im16=np.array(Imf)
        r,g,b = Im16[:,:,0], Im16[:,:,1], Im16[:,:,2]
        Im16 = 0.2989*r + 0.5870*g + 0.1140*b
        Im16=float(n)*Im16
        
        Im16[Im16>255]=255
        # 数据类型转化
        Im16=np.round(Im16)
        Im16=Im16.astype(np.uint8)
        return Image.fromarray(Im16)

    def calc_hist(gray):
        # 计算彩色图单通道的直方图
        hist_new = []
        num = []
        hist_result = []
        hist_key = []
        gray1 = list(gray.ravel()) # 将读取出来的数组转化为一维列表方便循环遍历
        obj = dict(collections.Counter(gray1)) # 计算每个灰度级出现的次数
        obj = sorted(obj.items(),key=lambda item:item[0])
        # 初始化hist数组
        for each in obj:
            hist1 = []
            key = list(each)[0]
            cnt = list(each)[1]
            hist_key.append(key)
            hist1.append(cnt)
            hist_new.append(hist1)
        # 检查从0-255每个通道是否都有个数，没有的话添加并将值设为0
        for i in range (0, 256):
            if i in hist_key:
                num = hist_key.index(i)
                hist_result.append(hist_new[num])
            else:
                hist_result.append([0])
        hist_result = np.array(hist_result)
        return hist_result

    # 计算直方图
    def showhist(image):
        image= np.array(image)
        r,g,b = image[:,:,0], image[:,:,1], image[:,:,2]

        hist_new_b = modules.calc_hist(b)
        hist_new_g = modules.calc_hist(g)
        hist_new_r = modules.calc_hist(r)

        # 绘制直方图
        plt.plot(hist_new_b, color='b')
        plt.plot(hist_new_g, color='g')
        plt.plot(hist_new_r, color='r')
        plt.show()

    # 图像相加函数
    def IMG_PLUS(img1, img2):
        # 先修改img1尺寸和img2相同
        img1 = cv.resize(img1, (img2.shape[1], img2.shape[0]))
        # 矩阵相加
        newimg = img1*0.5 + img2*0.5
        newimg = newimg.astype(np.uint8)
        return newimg

    # 图像相加
    def Add(img1, img2):
        first = np.array(img1)
        second = np.array(img2)
        newimg = modules.IMG_PLUS(first, second)
        #return ImageTk.PhotoImage(Image.fromarray(newimg))
        return Image.fromarray(newimg)

    # 均值滤波处理函数
    def mean_filter(img, b=3):
        padnum = (b-1)//2# 填充数量
        pad = ((padnum, padnum), (padnum, padnum), (0,0))# 填充格式
        Filter = np.ones((b, b, img.shape[2]), img.dtype)# 方阵滤波器
        padnumImg= np.pad(img, pad, 'constant', constant_values=(0, 0))
        # 用滤波器对图像中像素依次计算取均值
        for i in range(padnum, padnumImg.shape[0] - padnum):
            for j in range(padnum, padnumImg.shape[1] - padnum):
                padnumImg[i][j] = (Filter * padnumImg[i-padnum:i+padnum+1, j-padnum:j+padnum+1]).sum(axis = 0).sum(axis = 0)//(b ** 2)
        newimg = padnumImg[padnum:padnumImg.shape[0] - padnum, padnum:padnumImg.shape[1] - padnum]  # 剪切使尺寸一样
        return newimg

    # 中值滤波处理函数
    def median_filter(img, b=3):
        padnum = (b-1)//2# 填充数量
        pad = ((padnum, padnum), (padnum, padnum), (0,0))# 填充格式
        padImg= np.pad(img, pad, 'constant', constant_values=(0, 0))# 方阵滤波器
        # 按通道计算中值函数
        def DimensionAdd(img):
            blank = np.zeros((img.shape[2]))
            for i in range(img.shape[2]):
                blank[i] = np.median(img[:,:,i])
            return blank
        # 用滤波器对图像中像素依次计算中值
        for i in range(padnum, padImg.shape[0] - padnum):
            for j in range(padnum, padImg.shape[1] - padnum):
                padImg[i][j] = DimensionAdd(padImg[i-padnum:i+padnum+1, j-padnum:j+padnum+1])
        newimg = padImg[padnum:padImg.shape[0] - padnum, padnum:padImg.shape[1] - padnum]  # 把操作完多余的0去除，保证尺寸一样大
        return newimg

    # 均值滤波
    def filter1(img):
        l = simpledialog.askinteger(title='滤波核size', prompt='边长L', initialvalue=3, minvalue=0, maxvalue=99)
        img = modules.mean_filter(np.array(img), l)
        return ImageTk.PhotoImage(Image.fromarray(img))

    # 中值滤波
    def filter2(img):
        l = simpledialog.askinteger(title='滤波核size', prompt='边长L', initialvalue=3, minvalue=0, maxvalue=99)
        img = modules.median_filter(np.array(img), l)
        return ImageTk.PhotoImage(Image.fromarray(img))

    # sobel锐化
    def sharpen(img):
        
        img = np.array(img)
        r,g,b = img[:,:,0], img[:,:,1], img[:,:,2]
        img = 0.2989*r + 0.5870*g + 0.1140*b
        # sobel算子
        G_x = np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]])
        G_y = np.array([[-1, -2, -1],[0, 0, 0],[1, 2, 1]])
        rows = np.size(img, 0)
        columns = np.size(img, 1)
        mag = np.zeros(img.shape)
        # 分别检测水平和垂直，在计算每个pixel的时候，将水平和垂直的值作一次平方和的处理
        for i in range(0, rows - 2):
            for j in range(0, columns - 2):
                v = sum(sum(G_x * img[i:i+3, j:j+3]))  # vertical
                h = sum(sum(G_y * img[i:i+3, j:j+3]))  # horizon
                mag[i+1, j+1] = np.sqrt((v ** 2) + (h ** 2))
        # 设置阈值
        threshold=120
        mag[mag<threshold] = 0

        mag = mag.astype(np.uint8)
        return Image.fromarray(mag)


    def style_transfer(content_img, style_img, steps):
        result = solver.run(content_img, style_img, steps)
        return result

        
# ---风格转换相关---
# 内容损失
class ContentLoss(nn.Module):
    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        # 将目标内容从用于动态计算梯度的树中“分离”出来
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input


def gram_matrix(input):
    a, b, c, d = input.size()
    # a:=batch size(=1)
    # b:=number of feature maps
    # (c,d):=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)

    G = torch.mm(features, features.t())  # 计算格拉姆矩阵

    # 矩阵标准化
    return G.div(a * b * c * d)

# 风格损失
class StyleLoss(nn.Module):

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input


# 创建一个模块来标准化输入的图像，使其能被放在nn.Sequential中
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = mean.clone().detach().view(-1, 1, 1)
        self.std = std.clone().detach().view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std

class solver(object):
    content_layers_default = ['conv_4']
    style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
    unloader = transforms.ToPILImage()  # 转换到PIL格式
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 期望输出图像的大小
    imsize = 512
    loader = transforms.Compose([
        # 缩放图片大小
        transforms.Resize(imsize),
        # 转换成pytorch tensor形式
        transforms.ToTensor()])

    def load_image(image_name):
        image = solver.loader(image_name).unsqueeze(0)
        return image.to(solver.device, torch.float)

    # 打印图像
    def show_image(tensor, title=None):
        image = tensor.cpu().clone()  # 复制参数tensor
        image = image.squeeze(0) # 削减无用维度
        image = solver.unloader(image)
        plt.imshow(image)
        if title is not None:
            plt.title(title)
        plt.pause(1)
        return image

    def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                                   style_img, content_img,
                                   content_layers=content_layers_default,
                                   style_layers=style_layers_default):
        # 标准化模块
        normalization = Normalization(normalization_mean, normalization_std).to(solver.device)

        content_losses = []
        style_losses = []

        # 使用的是nn.Sequential, 则建立一个新的nn.Sequential
        # 放入模块，会按顺序激活
        model = nn.Sequential(normalization)

        i = 0  # 每次看到一个conv层就增加
        for layer in cnn.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = 'conv_{}'.format(i)
            elif isinstance(layer, nn.ReLU):
                name = 'relu_{}'.format(i)
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = 'pool_{}'.format(i)
            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}'.format(i)
            else:
                raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

            model.add_module(name, layer)

            if name in content_layers:
                # 加入内容损失:
                target = model(content_img).detach()
                content_loss = ContentLoss(target)
                model.add_module("content_loss_{}".format(i), content_loss)
                content_losses.append(content_loss)

            if name in style_layers:
                # 加入风格损失:
                target_feature = model(style_img).detach()
                style_loss = StyleLoss(target_feature)
                model.add_module("style_loss_{}".format(i), style_loss)
                style_losses.append(style_loss)

        for i in range(len(model) - 1, -1, -1):
            if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
                break
            
        model = model[:(i + 1)]
        return model, style_losses, content_losses

    def get_input_optimizer(input_img):
        optimizer = optim.LBFGS([input_img])
        return optimizer

    def run_style_transfer(cnn, normalization_mean, normalization_std,
                           content_img, style_img, input_img, num_steps=300,
                           style_weight=1000000, content_weight=1):
        """Run the style transfer."""
        print('Building the style transfer model..')
        model, style_losses, content_losses = solver.get_style_model_and_losses(cnn,
            normalization_mean, normalization_std, style_img, content_img)

        # 优化输入而不是模型参数，因此更新所有requires_grad字段
        input_img.requires_grad_(True)
        model.requires_grad_(False)

        optimizer = solver.get_input_optimizer(input_img)

        print('Optimizing..')
        run = [0]
        while run[0] <= num_steps:

            def closure():
                # 修正更新后的输入图像的值
                with torch.no_grad():
                    input_img.clamp_(0, 1)

                optimizer.zero_grad()
                model(input_img)
                style_score = 0
                content_score = 0

                for sl in style_losses:
                    style_score += sl.loss
                for cl in content_losses:
                    content_score += cl.loss

                style_score *= style_weight
                content_score *= content_weight

                loss = style_score + content_score
                loss.backward()

                run[0] += 1
                if run[0] % 50 == 0:
                    print("run {}:".format(run))
                    print('Style Loss: {:4f} Content Loss: {:4f}'.format(
                        style_score.item(), content_score.item()))
                    print()

                return style_score + content_score

            optimizer.step(closure)

        with torch.no_grad():
            input_img.clamp_(0, 1)

        return input_img

    def run(content_img, style_img, steps):

        content_img = solver.load_image(content_img.convert('RGB'))
        style_img = solver.load_image(style_img.convert('RGB'))

        # 风格图像和内容图像大小要保持一致
        assert style_img.size() == content_img.size(), "Error"

        #plt.ion()
        #plt.figure()
        #solver.show_image(style_img, title='Style Image')

        #plt.figure()
        #solver.show_image(content_img, title='Content Image')

        cnn = models.vgg19(weights='VGG19_Weights.DEFAULT').features.to(solver.device).eval()
        cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(solver.device)
        cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(solver.device)

        input_img = content_img.clone()
        # input_img = torch.randn(content_img.data.size(), device=device)

        output = solver.run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                            content_img, style_img, input_img, steps)

        plt.figure()
        result = solver.show_image(output, title='Output Image')


        
        # plt.ioff()
        # plt.show()


        return result
        

class Win(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('图像处理期末demo')
        self.geometry('1080x720')
        self.picture = picture()  # self.picture将作为picture类的实例化对象
        self.img = None  # self.img将作为窗口类中一直变动的PIL对象图片

        self.setupUI()

    def setupUI(self):
        # 右边菜单栏
        right_f = tk.Frame(self, height=720, width=360,bg="#FFFFFF")
        right_f.pack(side=tk.RIGHT)

        # 各种功能按钮名称及位置
        btn1 = tk.Button(right_f, text='打开图像',fg='#FFFFFF', bg="#7C3338",command=self.open_picture)
        btn1.place(y=25, x=30, width=300, height=40)
        btn2 = tk.Button(right_f, text='基本调整', bg="#FFE498", command=self.basic_proc)
        btn2.place(y=85, x=30, width=144, height=40)
        btn3 = tk.Button(right_f, text='调整大小', bg="#FFE498", command=self.picture_resize)
        btn3.place(y=85, x=186, width=144,height=40)
        btn4 = tk.Button(right_f, text='旋转操作', bg='#FFE498', command=self.picture_rotate)
        btn4.place(y=145, x=30, width=144, height=40)
        btn5 = tk.Button(right_f, text='镜像操作', bg='#FFE498',command=self.picture_mirror)
        btn5.place(y=145, x=186, width=144, height=40)
        btn6 = tk.Button(right_f, text='灰度变换', bg='#FFE498',command=self.picture_gray)
        btn6.place(y=205, x=30, width=144, height=40)
        btn7 = tk.Button(right_f, text='滤镜', bg='#FFE498',command=self.picture_filters)
        btn7.place(y=205, x=186, width=144, height=40)
        btn8 = tk.Button(right_f, text='显示直方图', bg='#FFE498',command=self.show_hist)
        btn8.place(y=265, x=30, width=144, height=40)
        btn9 = tk.Button(right_f, text='图像相加', bg='#FFE498',command=self.add_picture)
        btn9.place(y=265, x=186, width=144, height=40)
        btn10 = tk.Button(right_f, text='滤波', bg='#FFE498',command=self.filtering)
        btn10.place(y=325, x=30, width=144, height=40)
        btn11 = tk.Button(right_f, text='Sobel锐化', bg='#FFE498',command=self.sharpening)
        btn11.place(y=325, x=186, width=144, height=40)
        btn12 = tk.Button(right_f, text='风格转换', bg='#FFE498',command=self.style_transfer)
        btn12.place(y=385, x=30, width=144, height=40)

        # 底部恢复、保存、对比
        btn13 = tk.Button(right_f, text='保存图像',fg='#FFFFFF', command=self.save_pic,bg="#7C3338")
        btn13.place(y=590, x=230, width=90, height=30)
        btn14 = tk.Button(right_f, text='恢复图像',fg='#FFFFFF', command=self.replay,bg="#7C3338")
        btn14.place(y=590, x=30, width=90, height=30)
        btn15 = tk.Button(right_f, text='对比图像',fg='#FFFFFF', command=self.compare,bg="#7C3338")
        btn15.place(y=590, x=130, width=90, height=30)


        # 图像显示栏
        right_f = tk.Frame(self, height=720, width=720)
        right_f.pack(side=tk.RIGHT)
        self.image_l = tk.Label(right_f, relief='ridge')
        self.image_l.place(x=0, y=0, width=720, height=720)

    def filtering(self):
        filter_win = tk.Toplevel()
        filter_win.title('滤波选择')
        filter_win.geometry('150x150')
        b1 = tk.Button(filter_win, text='均值滤波', command=self.filter1)
        b1.place(y=30, x=35, width=75)
        b2 = tk.Button(filter_win, text='中值滤波', command=self.filter2)
        b2.place(y=60, x=35, width=75)
        b3 = tk.Button(filter_win, text='完成', command=filter_win.destroy)
        b3.place(y=110, x=80, width=40)

    def sharpening(self):
        sharpen_pic = self.img
        sharpen_pic = modules.sharpen(sharpen_pic)
        self.show_img(sharpen_pic)
    # 打开图片时使用，传值(图)给展示函数
    def open_picture(self):
        address = self.getAddress()
        self.open_picToimg = self.picture.open_pic(address)
        self.firstPic(self.open_picToimg)
        self.show_img(self.open_picToimg)

    # 打开图片时使用，传值(图)给展示函数
    def add_picture(self):
        address = self.getAddress()
        self.open_picToimg = self.picture.open_pic(address)
        self.add_pic(self.open_picToimg)
        
    def filter1(self):
        img_show = modules.filter1(self.img)
        self.image_l.config(image=img_show)
        self.image_l.image = img_show
        return self.img

    def filter2(self):
        img_show = modules.filter2(self.img)
        self.image_l.config(image=img_show)
        self.image_l.image = img_show
        return self.img

    # 打开图片时使用，获得地址
    def getAddress(self):
        path = tk.StringVar()
        file_entry = tk.Entry(self, state='readonly', text=path)
        path_ = askopenfilename()
        path.set(path_)

        return file_entry.get()

    # 展示函数
    def show_img(self, n_img):
        self.img = n_img  # self.img PIL对象方便传值给picture类以及本类中其他需要使用PIL图像的地方
        img_show = ImageTk.PhotoImage(self.img)
        self.image_l.config(image=img_show)
        self.image_l.image = img_show
        return self.img

    def add_pic(self, n_img):
        img_show = n_img
        img_show = modules.Add(img_show, self.img)
        self.show_img(img_show)
        
    
    # 保存函数
    def save_pic(self):
        fname = tkinter.filedialog.asksaveasfilename(title='保存文件', filetypes=[("PNG", ".png")])
        self.img.save(str(fname))  # PIL保存

    # 原图储存
    def firstPic(self, pic):
        self.Fpic = pic
        return self.Fpic
    
    # 打开图像时的图像压缩展示
    def openResize(self):
        w, h = self.pic_get.size
        w_hope = 500
        h_hope = 300
        f1 = 1.0 * w_hope / w
        f2 = 1.0 * h_hope / h
        factor = min([f1, f2])
        width = int(w * factor)
        height = int(h * factor)
        pic_show = self.pic_get.resize((width, height))
        return pic_show


    # 基本操作
    def basic_proc(self):
        Bas_win = tk.Toplevel()
        Bas_win.title('基本调整')
        Bas_win.geometry('250x250')

        l0 = tk.Label(Bas_win, text="%")
        l0.place(y=30, x=100)
        self.i0 = tk.Entry(Bas_win, width=13)#输入参数n
        self.i0.place(y=30, x=20)
        bt1 = tk.Button(Bas_win, text='亮度调整', command=self.brightnessPic)
        bt1.place(y=25, x=130, width=80)

        l1 = tk.Label(Bas_win, text="%")
        l1.place(y=65, x=100)
        self.i1 = tk.Entry(Bas_win, width=13)#输入参数n
        self.i1.place(y=70, x=20)
        bt2 = tk.Button(Bas_win, text='色彩度调整', command=self.colorPic)
        bt2.place(y=65, x=130, width=80)

        l2 = tk.Label(Bas_win, text="%")
        l2.place(y=105, x=100)
        self.i2 = tk.Entry(Bas_win, width=13)#输入参数n
        self.i2.place(y=110, x=20)
        bt3 = tk.Button(Bas_win, text='对比度调整', command=self.contrastPic)
        bt3.place(y=105, x=130, width=80)

        l3 = tk.Label(Bas_win, text="%")
        l3.place(y=145, x=100)
        self.i3 = tk.Entry(Bas_win, width=13)#输入参数n
        self.i3.place(y=150, x=20)
        bt4 = tk.Button(Bas_win, text='锐度调整', command=self.sharpnessPic)
        bt4.place(y=145, x=130, width=80)

        bt6 = tk.Button(Bas_win, text='完成', command=Bas_win.destroy)
        bt6.place(y=200, x=100)

    # 大小尺寸操作窗口
    def picture_resize(self):
        Size_win = tk.Toplevel()
        Size_win.title('尺寸操作')
        Size_win.geometry('300x180')
        l1 = tk.Label(Size_win, text="宽:")
        l1.place(y=30, x=80)
        self.text1 = tk.Entry(Size_win, width=10)
        self.text1.place(y=30, x=100)
        l1_1 = tk.Label(Size_win, text="px")
        l1_1.place(y=30, x=170)
        l2 = tk.Label(Size_win, text="高:")
        l2.place(y=60, x=80)
        self.text2 = tk.Entry(Size_win, width=10)
        self.text2.place(y=60, x=100)
        l2_1 = tk.Label(Size_win, text="px")
        l2_1.place(y=60, x=170)
        b1 = tk.Button(Size_win, text='确定', command=self.getSize_change)
        b1.place(y=100, x=100, width=40)
        b2 = tk.Button(Size_win, text='完成', command=Size_win.destroy)
        b2.place(y=145, x=100, width=40)


    # 获得输入尺寸
    def getSize_change(self):
        sizeNum_w = int(self.text1.get())
        sizeNum_h = int(self.text2.get())
        # print('1')
        self.showSize_change(sizeNum_w, sizeNum_h)
        # print(sizeNum_w, sizeNum_h)

    # 尺寸修改并展示图片
    def showSize_change(self, renewSize_w, renewSize_h):
        # print('2')
        
        needResize_pic = self.img
        show_resizePic = self.picture.changeResize(needResize_pic, renewSize_w, renewSize_h)
        self.show_img(show_resizePic)

    # 灰度转换功能窗口
    def picture_gray(self):
        Word_win = tk.Toplevel()
        Word_win.title('灰度变化')
        Word_win.geometry('300x400')
        l1 = tk.Label(Word_win, text='请选择灰度转换方法:')
        l1.place(y=10, x=20, width=130)
        l0 = tk.Label(Word_win, text="n(n值默认1.5):")
        l0.place(y=40, x=20)
        self.i0 = tk.Entry(Word_win, width=13)#输入参数n
        self.i0.place(y=40, x=110)
        bt1 = tk.Button(Word_win, text='n值化', command=self.getn)
        bt1.place(y=70, x=20, width=80)

        l2 = tk.Label(Word_win, text="k(对比度默认1.5):")
        l2.place(y=110, x=20)
        self.i1 = tk.Entry(Word_win, width=13)#输入线性化参数k
        self.i1.place(y=110, x=110)
        l3 = tk.Label(Word_win, text="b(亮度默认1.5):")
        l3.place(y=130, x=20)
        self.i2 = tk.Entry(Word_win, width=13)#输入线性化参数b
        self.i2.place(y=130, x=110)
        bt2 = tk.Button(Word_win, text='线性化', command=self.getkb)
        bt2.place(y=160, x=20, width=80)

        
        l3 = tk.Label(Word_win, text="c(尺度比较常数默认30.0):")
        l3.place(y=200, x=20)
        self.i3 = tk.Entry(Word_win, width=13)#输入参数c
        self.i3.place(y=200, x=150)
        bt3 = tk.Button(Word_win, text='非线性化(对数转换)', command=self.getc)
        bt3.place(y=230, x=20, width=120)
        b7 = tk.Button(Word_win, text='完成', command=Word_win.destroy)
        b7.place(y=300, x=210)
    # 获得输入的n值转换值
    def getn(self):
        npic = self.img
        #n = float(self.i0.get())
        # 图像线性化获取展示
        npic= modules.tonpic(npic,
                            1.5 if self.i0.get()=="" else float(self.i0.get()))
        self.show_img(npic)
    # 获得输入的线性转换值
    def getkb(self):
        linearpic = self.img
        #k = float(self.i1.get())
        #b = float(self.i2.get())
        # 图像线性化获取展示
        linearpic= modules.linearization(linearpic,
                                         1.5 if self.i1.get()=="" else float(self.i1.get()),
                                         1.5 if self.i2.get()=="" else float(self.i2.get()))
        self.show_img(linearpic)
    # 获得输入的对数转换值
    def getc(self):
        logpic = self.img
        #c = float(self.i3.get())
        # 图像对数化获取展示
        logpic= modules.tologpic(logpic,
                                 30.0 if self.i3.get()=="" else float(self.i3.get()))
        self.show_img(logpic)

    # 图像旋转操作窗口
    def picture_rotate(self):
        Rot_win = tk.Toplevel()
        Rot_win.title('旋转操作')
        Rot_win.geometry('225x220')
        l1 = tk.Label(Rot_win, text="逆时针角度:")
        l1.place(y=20, x=10)
        self.inpt = tk.Entry(Rot_win, width=15)
        self.inpt.place(y=20, x=70)
        b0 = tk.Button(Rot_win, text='确定', command=lambda: self.getDegree('1'))
        b0.place(y=15, x=170, width=40)
        b1 = tk.Button(Rot_win, text='+90', command=lambda: self.getDegree('2'))
        b1.place(y=85, x=70, width=95)
        b2 = tk.Button(Rot_win, text='-90', command=lambda: self.getDegree('3'))
        b2.place(y=115, x=70, width=95)
        b3 = tk.Button(Rot_win, text='180', command=lambda: self.getDegree('4'))
        b3.place(y=145, x=70, width=95)
        b4 = tk.Button(Rot_win, text='完成', command=Rot_win.destroy)
        b4.place(y=180, x=100, width=40)

    # 旋转角度获取
    def getDegree(self, n):

        needRotate_pic = self.img
        # print('99')
        # print(n)
        if n == '1':
            inputDegree = float(self.inpt.get())
            showRotate_pic = self.picture.rotatePic(needRotate_pic, inputDegree)
        elif n == '2':
            # print('34')
            showRotate_pic = self.picture.rotatePic(needRotate_pic, +90)
        elif n == '3':
            showRotate_pic = self.picture.rotatePic(needRotate_pic, -90)
        elif n == '4':
            showRotate_pic = self.picture.rotatePic(needRotate_pic, 180)
        else:
            return 0
        self.show_img(showRotate_pic)

    # 滤镜选择窗口
    def picture_filters(self):
        Sty_win = tk.Toplevel()
        Sty_win.title('滤镜选择')
        Sty_win.geometry('230x180')
        bt1 = tk.Button(Sty_win, text='图像模糊', command=self.sty_1)
        bt1.place(y=25, x=25, width=80)
        bt2 = tk.Button(Sty_win, text='轮廓滤波', command=self.sty_2)
        bt2.place(y=25, x=115, width=80)
        bt3 = tk.Button(Sty_win, text='高斯模糊', command=self.sty_3)
        bt3.place(y=65, x=25, width=80)
        bt4 = tk.Button(Sty_win, text='浮雕滤镜', command=self.sty_4)
        bt4.place(y=65, x=115, width=80)
        bt6 = tk.Button(Sty_win, text='完成', command=Sty_win.destroy)
        bt6.place(y=140, x=160)

    # 图像模糊获取展示
    def sty_1(self):
        sty_1_pic = self.img
        relSty_1 = modules.blurPic(sty_1_pic)
        self.show_img(relSty_1)

    # 边界增强获取展示
    def sty_2(self):
        sty_2_pic = self.img
        reSty_2 = modules.edge(sty_2_pic)
        self.show_img(reSty_2)

    # 高斯模糊获取展示
    def sty_3(self):
        sty_3_pic = self.img
        reSty_3 = modules.gaussianBlur(sty_3_pic)
        self.show_img(reSty_3)

    # 浮雕滤镜获取展示
    def sty_4(self):
        sty_4_pic = self.img
        reSty_4 = modules.emboss(sty_4_pic)
        self.show_img(reSty_4)

    # 亮度调整
    def brightnessPic(self):
        
        needBright_pic = self.img
        b_num = float(self.i0.get())
        briNum = b_num / 100
        showBright_pic = self.picture.brightPic(needBright_pic, briNum)
        self.show_img(showBright_pic)

    # 色彩度调整
    def colorPic(self):
        
        needColor_pic = self.img
        co_num = float(self.i1.get())
        colNum = co_num / 100
        showColor_pic = self.picture.colornPic(needColor_pic, colNum)
        self.show_img(showColor_pic)

    # 对比度调整
    def contrastPic(self):
        
        needCon_pic = self.img
        c_num = float(self.i2.get())
        ConNum = c_num / 100
        showContrast_pic = self.picture.constractPic(needCon_pic, ConNum)
        self.show_img(showContrast_pic)

    # 锐度调整
    def sharpnessPic(self):
        
        needSharp_pic = self.img
        s_num = float(self.i3.get())
        ShNum = s_num / 100
        showSharp_pic = self.picture.constractPic(needSharp_pic, ShNum)
        self.show_img(showSharp_pic)


    def open_style_img_and_run(self):
        address = self.getAddress()
        self.style_img = self.picture.open_pic(address)
        content_img = self.img

        self.style_img = self.style_img.resize((content_img.width, content_img.height))

        # prompt
        Prompt_win = tk.Toplevel()
        Prompt_win.title('提示')
        Prompt_win.geometry('230x150')
        l1 = tk.Label(Prompt_win, text="output image生成中,保持终端打开...")
        l1.place(y=30, x=25)
        self.update()

        result = modules.style_transfer(content_img, self.style_img, int(300 if len(self.inpt.get())==0 else self.inpt.get()))
        self.show_img(result)

        Prompt_win.destroy()


    # 风格转换
    def style_transfer(self):
        Sty_win = tk.Toplevel()
        Sty_win.title('风格转换')
        Sty_win.geometry('300x200')

        bt1 = tk.Button(Sty_win, text='选择风格图像并运行(主窗口显示图像为内容图像)', command=self.open_style_img_and_run)
        bt1.place(y=60, x=25, width=250)
        l1 = tk.Label(Sty_win, text="步数(越大越慢,默认300)")
        l1.place(y=25, x=25)

        self.inpt = tk.Entry(Sty_win, width=10)
        self.inpt.place(y=25, x=150)
        
        bt2 = tk.Button(Sty_win, text='完成', command=Sty_win.destroy)
        bt2.place(y=150, x=130)


    # 镜像操作窗口
    def picture_mirror(self):
        Mir_win = tk.Toplevel()
        Mir_win.title('镜像操作')
        Mir_win.geometry('300x150')
        b1 = tk.Button(Mir_win, text='左右', command=self.MirrorImg_lr)
        b1.place(y=30, x=100, width=75)
        b2 = tk.Button(Mir_win, text='上下', command=self.MirrorImg_tb)
        b2.place(y=60, x=100, width=75)
        b3 = tk.Button(Mir_win, text='完成', command=Mir_win.destroy)
        b3.place(y=110, x=100, width=40)

    # 镜像左右调用展示
    def MirrorImg_lr(self):
        
        Mirror_img_lr = self.img
        MittotImg_lrFinish = self.picture.MirrorPic_leftOrright(Mirror_img_lr)
        self.show_img(MittotImg_lrFinish)

    # 镜像上下调用展示
    def MirrorImg_tb(self):
        
        Mirror_img_tb = self.img
        MittotImg_tbFinish = self.picture.MirrorPic_topOrbuttom(Mirror_img_tb)
        self.show_img(MittotImg_tbFinish)

    # 恢复图像
    def replay(self):
        self.show_img(self.Fpic)

    # 对比图像
    def compare(self):
        Image._show(self.Fpic)

    def show_hist(self):
        #gethist = modules.showhist(self.img)
        #self.show_img(gethist)
        modules.showhist(self.img)


if __name__ == '__main__':
    root = Win()
    # 窗体主循环
    root.mainloop()
