from kivy.app import App
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.slider import Slider
from kivy.graphics import Color, Bezier, Line
import numpy as np
import GA_StockTrader as GA
from GA_StockTrader import *
from kivy.clock import Clock
import threading
#import kivy.graphics.Callback as Callback
from thread import start_new_thread
import time

#points = []

from kivy.core.text import Label as CoreLabel
#from kivy.graphics import *
from kivy.config import Config
#from kivy.graphics import Rectangle
from kivy.graphics.vertex_instructions import Rectangle
#from kivy.core.window import Window
Config.set('graphics', 'width', '1450')
Config.set('graphics', 'height', '600')
from kivy.properties import StringProperty
from kivy.uix.label import Label as CoreLabel2

class BezierTest(FloatLayout):
    def __init__(self, points=[], loop=False, *args, **kwargs):
        super(BezierTest, self).__init__(*args, **kwargs)

        self.d = 10  # pixel tolerance when clicking on a point
        self.points = points
        self.loop = loop
        self.saved_points=[]
        self.line_height = 0
        self.current_point = None  # index of point being dragged
        self.GA = threading.Thread(target=RunGA, args=(self,))
        self.GA.start()
        print "Spin thread."

        self.label = CoreLabel(text=LSTM.trade_stocks[0], font_size=20)
        self.label.refresh()

        self.label2 = CoreLabel(text=LSTM.trade_stocks[0], font_size=10)
        self.label2.refresh()

        self.label3 = CoreLabel(text=LSTM.trade_stocks[0], font_size=10)
        self.label3.refresh()


        with self.canvas:
            Color(1.0, 0.0, 0.0)
            #self.cb = Callback(self.my_callback)

            Color(0.1, 0.1, 0.1)

            seg = 1300/6

            Line(
                    points=[75,0,75,800],
                    width=0.8,
                    close=False)

            Line(
                    points=[seg,0,seg,800],
                    width=0.8,
                    close=False)

            Line(
                    points=[seg*2,0,seg*2,800],
                    width=0.8,
                    close=False)

            Line(
                    points=[seg*3,0,seg*3,800],
                    width=0.8,
                    close=False)
            Line(
                    points=[seg*4,0,seg*4,800],
                    width=0.8,
                    close=False)

            Line(
                    points=[seg*5,0,seg*5,800],
                    width=0.8,
                    close=False)

            Line(
                    points=[seg*6,0,seg*6,800],
                    width=0.8,
                    close=False)

            Line(
                    points=[1335,0,1335,800],
                    width=5,
                    close=False)

            '''self.bezier = Bezier(
                    points=self.points,
                    segments=150,
                    loop=self.loop,
                    dash_length=100,
                    dash_offset=10)
            '''
            Color(0.0, 0.0, 1.0)
            self.line1 = Line(
                    points=self.points,
                    width=0.8,
                    close=False)

        """s = Slider(y=0, pos_hint={'x': .3}, size_hint=(.3, None), height=20)
        s.bind(value=self._set_bezier_dash_offset)
        self.add_widget(s)

        s = Slider(y=20, pos_hint={'x': .3}, size_hint=(.3, None), height=20)
        s.bind(value=self._set_line_dash_offset)
        self.add_widget(s)"""

    def my_callback(self, instr):
        print('I have been called!')


    def render(self):
        self.rect = Rectangle(size=self.size, pos=self.pos)
        self.canvas.add(self.rect)
        label = CoreLabel(text="Text Lable here", font_size=20)
        label.refresh()
        text = label.texture
        #self.canvas.add(Color(self.colour, 1-self.colour,0, 1))
        pos = [150,150]#self.pos[i] + (self.size[i] - text.size[i]) / 2 for i in range(2))
        self.canvas.add(Rectangle(size=text.size, pos=pos, texture=text))
        self.canvas.ask_update()

    def update(self, dt):
        with self.canvas:
            self.line_height

            Color(0.4, 0.4, 0.4)
            self.line1.points = self.saved_points
            self.line1 = Line(
                    points=self.saved_points,
                    width=0.8,
                    close=False)

            Line(
                    points=[0,self.line_height,2500,self.line_height],
                    width=0.5,
                    close=False)

            Color(.4, 0.4, .4)

        #self.line1.points = self.saved_points
        #self.cb.ask_update()
        #print "redraw line"

    def draw_point(self, pos, sell):
        with self.canvas.after:
            if sell:
                Color(0.0, 1.0, 0.0)
            else:
                Color(0.0, 0.5, 1.0)
            Line(circle=(pos[0], pos[1], 4), width=1.6)

    def draw_rect(self, pos, sell, count, count2, umoney, dmoney):
        with self.canvas:
            #if sell>0:
                Color(0.0, 1.0, 0.0)
                #Rectangle(pos=(pos[0]+count*2, pos[1]+8), size=(2, 10))
                Rectangle(pos=(pos[0], pos[1]+8), size=(count, 7))
                #Rectangle(pos=(pos[0], pos[1]+20), size=(umoney, 7))
            #else:
                Color(1.0, 0.0, 0.0)
                #Rectangle(pos=(pos[0]+count*2, pos[1]-13), size=(2, 10))
                Rectangle(pos=(pos[0], pos[1]-13), size=(count2, 7))
                #Rectangle(pos=(pos[0], pos[1]-21), size=(dmoney, 7))

    def draw_text(self, txt, pos, txt_size, color=[.7,.7,.7]):
        with self.canvas:
            CoreLabel2(text=str(txt), pos=pos, font_size=txt_size)
            #Color(1.0, 1.0, 1.0)
            #text = self.label.texture
            #pos2 = list(pos[i] + (self.size[i] - text.size[i]) / 2 for i in range(2))
            #Rectangle(size=text.size, pos=pos, texture=text)


    def update_lines(self):
        #self.line1.points = self.saved_points
        #with self.canvas:
        #    Color((.8+random.random()*.2)*.5, (.8+random.random()*.2)*.5, (.8+random.random()*.2)*.5)
        print "updating line"

    def _set_bezier_dash_offset(self, instance, value):
        pass
        # effect to reduce length while increase offset
        #self.bezier.dash_length = 100 - value
        #self.bezier.dash_offset = value

    def _set_line_dash_offset(self, instance, value):
        pass
        # effect to reduce length while increase offset
        #self.line.dash_length = 100 - value
        #self.line.dash_offset = value

    def on_touch_down(self, touch):
        with self.canvas:
                self.line1.points = self.saved_points

        if self.collide_point(touch.pos[0], touch.pos[1]):
            for i, p in enumerate(list(zip(self.points[::2],
                                           self.points[1::2]))):
                if (abs(touch.pos[0] - self.pos[0] - p[0]) < self.d and
                    abs(touch.pos[1] - self.pos[1] - p[1]) < self.d):
                    self.current_point = i + 1
                    return True
            return super(BezierTest, self).on_touch_down(touch)

    def on_touch_up(self, touch):
        with self.canvas:
            self.line1.points = self.saved_points

        if self.collide_point(touch.pos[0], touch.pos[1]):
            if self.current_point:
                self.current_point = None
                return True
            return super(BezierTest, self).on_touch_up(touch)

    def on_touch_move(self, touch):
        if self.collide_point(touch.pos[0], touch.pos[1]):
            c = self.current_point
            if c:
                self.points[(c - 1) * 2] = touch.pos[0] - self.pos[0]
                self.points[(c - 1) * 2 + 1] = touch.pos[1] - self.pos[1]
                self.bezier.points = self.points
                self.line.points = self.points + self.points[:2]
                return True
            return super(BezierTest, self).on_touch_move(touch)

import random

class AI_Market_Trader(App):

    def build(self):
        from math import cos, sin, radians
        x = y = 150
        l = 100

        tmppoints=[]
        #for i in range(45, 360, 45):
        """for i in range(10):
            y = 0.5+random.random()
            x = i
            tmppoints.extend([x*40, y*100])"""
        ui = BezierTest(points=tmppoints, loop=False)
        ui.saved_points = tmppoints
        Clock.schedule_interval(ui.update, 0.1)
        return ui

AI_Market_Trader().run()