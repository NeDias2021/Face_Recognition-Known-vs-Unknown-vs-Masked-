# import necessary libraries
import face_recognition
import cv2
from datetime import datetime, timedelta
import numpy as np
import platform
import pickle
import os
import dlib

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision 
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

transform = transforms.Compose(
	[transforms.Resize((512,512)),
	 transforms.ToTensor(),
	 transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])

# recreate the CNN and update the weights using the trained CNN (for more comments/annotations of the CNN refer to cnn.py)
class ConvNet(nn.Module):
	def __init__(self):
		super(ConvNet, self).__init__()
		self.conv1 = nn.Conv2d(3, 6, 5)
		self.pool = nn.MaxPool2d(2,2)
		self.conv2 = nn.Conv2d(6,16,5)
		self.fc1 = nn.Linear(125*125*16, 120)
		self.fc2 = nn.Linear(120, 84)
		self.fc3 = nn.Linear(84, 2)

	def forward(self, x):
		x = self.pool(F.relu(self.conv1(x)))
		x = self.pool(F.relu(self.conv2(x)))
		x = x.view(-1, 125*125*16) 
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)   
		return x

model = ConvNet()
model.load_state_dict(torch.load("/home/realtimeml/model.pth"))
model.eval()

# reading camera
video_capture = cv2.VideoCapture(0)

process_this_frame = True

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
known_face_encodings = []
known_face_names = []
unknown_face_encodings = []
unknown_face_metadata = []
mask_metadata = []
images = []
mask_metadata.append({
		"last_seen": datetime.now()})

# Iterating through files
directory = '/home/realtimeml/known_people/'

for filename in os.listdir(directory):
	images.append(filename)
	known_face_names.append(filename[: len(filename)-4])

for x in range (len(images)):
	name = known_face_names[x]
	name_image = face_recognition.load_image_file("/home/realtimeml/known_people/"+images[x])
	name_face_encoding = face_recognition.face_encodings(name_image)[0]
	known_face_encodings.append(name_face_encoding)

# helpers
def find_visits(image_name, face_encoding, unknown_number):
	unknown_face_encodings.append(face_encoding)
	unknown_face_metadata.append({
		"first_seen_this_interaction": datetime.now(),
		"last_seen": datetime.now(),
		"seen_count": 1})
	visit = str(1)
	unknown_number = unknown_number+1
	return unknown_number, visit

# Main Loop
def main_loop():
	
	mask_time = mask_metadata[0]

	unknown_number = 1
	mask_number = 1
	visits = 0
	while True:
		# Grab a single frame of video
		ret, frame = video_capture.read()

		# Resize frame of video to 1/4 size for faster face recognition processing
		small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
		
		# Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
		rgb_small_frame = small_frame[:, :, ::-1]

	
		# Find all the face locations and face encodings in the current frame of the video
		face_locations = face_recognition.face_locations(rgb_small_frame)
		face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

		# Loop through each detected face and see if it is one we have seen before
		# If so, we'll give it a label that we'll draw on top of the video
		if process_this_frame:

			image1 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			im_pil = Image.fromarray(image1)
			image_tr = transform(im_pil)
			image_tr = image_tr.unsqueeze(0)
			output = model(image_tr)
			prediction = torch.argmax(output)
			if prediction == 1:
				mask = True
			else:
				mask = False
		# Find all the faces and face encodings in the current frame of video
			face_locations = face_recognition.face_locations(rgb_small_frame)
			face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
			face_names = []
			for face_encoding in face_encodings:

				
			
			# See if the face is a match for the known face(s)
				matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
				name = "Unknown"

				
				
				# If matches use the known face with the smallest distance to the new face
				face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
				best_match_index = np.argmin(face_distances)
				if matches[best_match_index] and mask == False:
					name = known_face_names[best_match_index]
				
				if mask == False:
					
					image = frame
					unknown_number_string = str(unknown_number)
					image_name = "unknown" + unknown_number_string

					if unknown_number == 1:
						visits = str(1)
						
						cv2.imwrite(os.path.join(r"/home/realtimeml/unknown_people/", image_name + "_NumberOfVisits:" + visits + ".jpg"), image)
						unknown_number = unknown_number+1
						unknown_face_encodings.append(face_encoding)
						unknown_face_metadata.append({ 
							"first_seen_this_interaction": datetime.now(),
							"last_seen": datetime.now(),
							"seen_count": 1})
	
					else:
						# Get previous image info
						match = face_recognition.compare_faces(unknown_face_encodings, face_encoding)
						if True in match:
							face_distance = face_recognition.face_distance(unknown_face_encodings, face_encoding)
							best_match = np.argmin(face_distance)
							if match[best_match]:
								image_name = "unknown" + str(best_match+1)
							
								metadata = unknown_face_metadata[best_match]	
								if datetime.now() - metadata["first_seen_this_interaction"] > timedelta(minutes = 5):
									previous_visit = str(metadata["seen_count"])			
									metadata["seen_count"] += 1
									visits = str(metadata["seen_count"])
									os.remove("/home/realtimeml/unknown_people/"+ image_name + "_NumberOfVisits:" + previous_visit + ".jpg")
				
								metadata["last_seen"] = datetime.now()
								metadata["first_seen_this_interaction"] = datetime.now()
							
						else:	
		 					unknown_number, visits = find_visits(image_name, face_encoding, unknown_number)
							

						cv2.imwrite(os.path.join(r"/home/realtimeml/unknown_people/", image_name + "_NumberOfVisits:" + visits + ".jpg" ), image)
						name = image_name + "_NumberOfVisits:" + visits
						

				# mask code to save image into folder
				if mask == True: 
					image = frame
					if mask_number == 1 or (datetime.now()-mask_time["last_seen"]> timedelta(minutes = 1)):
						mask_time["last_seen"] = datetime.now()					
						mask_number_string = str(mask_number)
						mask_name = "mask" + mask_number_string + ":"
						x = 0
						for face_distance in face_distances:
							if x < 10:
								index = np.argmin(face_distances)
								if mask_name.count(known_face_names[index]) == 0:
									mask_name = mask_name + known_face_names[index] + "," 
								face_distances = np.delete(face_distances, face_distances.argmin())
								x = x+1
						mask_name = mask_name[: len(mask_name)-1]
						mask_name = mask_name + ".jpg"
						(cv2.imwrite(os.path.join(r"/home/realtimeml/mask_people/", mask_name), image))
						mask_number = mask_number+1
					name = mask_name[: len(mask_name)-4]

				face_names.append(name)

		# Draw a box around each face and label each face
		for (top, right, bottom, left), face_label in zip(face_locations, face_names):
			# Scale back up face locations since the frame detected was scaled to 1/4 size
			top *= 4
			right *= 4
			bottom *= 4
			left *= 4
		
			# Draw a box around face
			cv2.rectangle(frame, (left,top),(right,bottom), (0,0,255),2)

			# Draw a label with a name below the face
			cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255),  cv2.FILLED)
			cv2.putText(frame, face_label, (left + 6, bottom - 6),  cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)
			
			# Display the final frame of video with boxes drawn around each detected frames
		cv2.imshow('Video', frame)

			# Hit 'q' on the keyboard to quit
		if cv2.waitKey(1) & 0xFF == ord('q'):
			save_known_faces()
			break
	
	# Release handle to the webcam
	video_capture.release()
	cv2.destroyAllWindows()

if __name__ == "__main__":
	# load known_faces()
	main_loop()