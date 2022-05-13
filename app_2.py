from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re

import math


# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image
import mediapipe as mp
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
#%matplotlib qt
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
# Flask utils
from flask import Flask, redirect, url_for, request, render_template, Response
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

app = Flask(__name__)


#prediction_model_1 = 'C:/Users/HP/Desktop/oxvi/deep_learning/Breathing/cnn_lstm_model_new5666.h5'

# Load your trained model
#model = load_model(prediction_model_1)
#model._make_predict_function()

cap = cv2.VideoCapture(0)


@app.route('/predict', methods=['POST'])
def model_predict( ):
	#folder_path = os.path.join(path, video_folder_name)
	#video_list = os.listdir(folder_path)
	prediction_list_1 = []
	number = []

	with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
		b = 0
		graph = []
		score = 0
		i = 0
		i_arr = []

		# landmarks = []
		while cap.isOpened():

			ret, frame = cap.read()
			frame = cv2.resize(frame, (500, 500))

			# Recolor image to RGB
			image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			image.flags.writeable = False
			# Make detection
			results = pose.process(image)
			# Recolor back to BGR
			image.flags.writeable = True
			image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
			# Extract landmarks
			try:
				landmarks = results.pose_landmarks.landmark
			# print(landmarks)
			except:
				pass
			a = landmarks[11].y
			if a > b:
				ans = "Exhale"
				score = score - 1
			# print(ans)
			elif a < b:
				ans = "Inhale"
				score = score + 1
			else:
				ans = "hold"
				score = score

			graph.append(score)
			i_arr.append(i)


			plt.style.use('dark_background')
			# plt.plot(i_arr,graph)
			# plt.xlabel("i_arr")
			# plt.ylabel("graph")
			ax = plt.gca()
			ax.axes.xaxis.set_visible(False)
			ax.axes.yaxis.set_visible(False)
			# print("abc")
			ax.patch.set_visible(False)
			ax.axis('off')

			# plt.xlim(i-1,i+30)
			# ax.set_ylim(graph[i]-1, graph[i]+100)
			plt.plot(i_arr, graph, scaley=True, scalex=True, color="red")

			# anim = FuncAnimation(fig, animate, interval=100)
			# print("ABC")

			rows, cols, channels = frame.shape
			roi = frame[400:rows, 400:cols]

			plt.savefig('C:/Users/HP/Desktop/oxvi/deep_learning/pose/graph.png')

			# Render detections
			mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
									  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
									  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
									  )
			b = a
			cv2.putText(frame, 'Status: %s' % ans,
						(10, 20), cv2.FONT_HERSHEY_TRIPLEX, 0.8, (50, 50, 50), 2)

			graph_img = cv2.imread('C:/Users/HP/Desktop/oxvi/deep_learning/pose/graph.png')
			graph_img = cv2.resize(graph_img, (100, 100))
			copy_img = graph_img
			small_img_gray = cv2.cvtColor(graph_img, cv2.COLOR_RGB2GRAY)
			# cv2.imshow("small_img_gray",small_img_gray)
			ret2, mask = cv2.threshold(small_img_gray, 0, 255, cv2.THRESH_BINARY)
			dkernel = np.ones((5, 5), np.uint8)
			mask = cv2.dilate(mask, dkernel, iterations=1)
			color = (0, 0, 255)  # sample of a color
			copy_img = np.full((100, 100, 3), color, np.uint8)
			masked = cv2.bitwise_and(copy_img, copy_img, mask=mask)

			# anim = FuncAnimation(fig, animate, interval=100)

			bg = cv2.add(roi, masked)
			cv2.imshow("bg", bg)
			y_offset = 400
			x_offset = 400
			frame[y_offset: y_offset + bg.shape[0], x_offset: x_offset + bg.shape[1]] = bg
			cv2.imshow("frame", frame)

			i = i + 1
			cv2.imshow('Mediapipe Feed', image)
			# cv2.imshow('original', frame)
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break

		cap.release()
		cv2.destroyAllWindows()



@app.route('/', methods=['GET'])
def index():
	# Main page
    return render_template('index.html')


def generate_frames():
	while True:

		## read the camera frame
		success, frame = cap.read()
		if not success:
			break
		else:
			ret, buffer = cv2.imencode('.jpg', frame)
			frame = buffer.tobytes()

		yield (b'--frame\r\n'
			   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video')
def video():
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=True,threaded=True)

