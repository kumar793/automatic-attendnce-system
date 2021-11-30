Name = str(input("Enter your Name : "))
Roll_Number = int(input("Enter your Roll_Number : "))
def register():

        import PySimpleGUI as sg
        import numpy as np
        import imutils
        import pickle
        import time
        import cv2
        import csv
        import serial
        import os
        from sklearn.preprocessing import LabelEncoder
        from sklearn.svm import SVC
        import pickle
        from imutils import paths
        import numpy as np
        import pickle
        def dataset():
                cascade = 'haarcascade_frontalface_default.xml'
                detector = cv2.CascadeClassifier(cascade)

                
                dataset = 'dataset'
                sub_data = Name
                path = os.path.join(dataset, sub_data)

                if not os.path.isdir(path):
                        os.mkdir(path)
                        print(sub_data)

                info = [str(Name), str(Roll_Number)]
                with open('student.csv', 'a') as csvFile:
                        write = csv.writer(csvFile)
                        write.writerow(info)
                csvFile.close()

                sg.theme("LightGreen")

        # Define the window layout
                layout = [
                [sg.Text("dataset creation", size=(60, 1), justification="center")],
                [sg.Image(filename="", key="-IMAGE-")],
                [sg.Button("Exit", size=(10, 1))]
               
        ]

        # Create the window and show it without the plot
                window = sg.Window("dataset creation", layout, location=(800, 400))

                print("Starting video stream...")
                cam = cv2.VideoCapture(0)
                time.sleep(2.0)
                total = 0

                while total < 30:
                        event, values = window.read(timeout=20)
                        if event == "Exit" or event == sg.WIN_CLOSED:
                                break
                        
                        elif event == "preprocess" :
                                preprocess()
                                train()
                        print(total)
                        _, frame = cam.read()
                        img = imutils.resize(frame, width=600)
                        rects = detector.detectMultiScale(
                                cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), scaleFactor=1.1,
                                minNeighbors=5, minSize=(30, 30))

                        for (x, y, w, h) in rects:
                                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                                p = os.path.sep.join([path, "{}.png".format(
                                        str(total).zfill(5))])
                                cv2.imwrite(p, img)
                                total += 1

                        imgbytes = cv2.imencode(".png", frame)[1].tobytes()
                        window["-IMAGE-"].update(data=imgbytes)
                window.close()
        def preprocess():
                dataset = "dataset"

                embeddingFile = "output/embeddings.pickle" #initial name for embedding file
                embeddingModel = "openface_nn4.small2.v1.t7" #initializing model for embedding Pytorch

                #initialization of caffe model for face detection
                prototxt = "model/deploy.prototxt"
                model =  "model/res10_300x300_ssd_iter_140000.caffemodel"

                #loading caffe model for face detection
                #detecting face from Image via Caffe deep learning
                detector = cv2.dnn.readNetFromCaffe(prototxt, model)

                #loading pytorch model file for extract facial embeddings
                #extracting facial embeddings via deep learning feature extraction
                embedder = cv2.dnn.readNetFromTorch(embeddingModel)

                #gettiing image paths
                imagePaths = list(paths.list_images(dataset))

                #initialization
                knownEmbeddings = []
                knownNames = []
                total = 0
                conf = 0.5

                #we start to read images one by one to apply face detection and embedding
                for (i, imagePath) in enumerate(imagePaths):
                        print("Processing image {}/{}".format(i + 1,len(imagePaths)))
                        name = imagePath.split(os.path.sep)[-2]
                        image = cv2.imread(imagePath)
                        image = imutils.resize(image, width=600)
                        (h, w) = image.shape[:2]
                        #converting image to blob for dnn face detection
                        imageBlob = cv2.dnn.blobFromImage(
                                cv2.resize(image, (300, 300)), 1.0, (300, 300),(104.0, 177.0, 123.0), swapRB=False, crop=False)

                        #setting input blob image
                        detector.setInput(imageBlob)
                        #prediction the face
                        detections = detector.forward()

                        if len(detections) > 0:
                                i = np.argmax(detections[0, 0, :, 2])
                                confidence = detections[0, 0, i, 2]

                                if confidence > conf:
                                        #ROI range of interest
                                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                                        (startX, startY, endX, endY) = box.astype("int")
                                        face = image[startY:endY, startX:endX]
                                        (fH, fW) = face.shape[:2]

                                        if fW < 20 or fH < 20:
                                                continue
                                        #image to blob for face
                                        faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
                                        #facial features embedder input image face blob
                                        embedder.setInput(faceBlob)
                                        vec = embedder.forward()
                                        knownNames.append(name)
                                        knownEmbeddings.append(vec.flatten())
                                        total += 1

                print("Embedding:{0} ".format(total))
                data = {"embeddings": knownEmbeddings, "names": knownNames}
                f = open(embeddingFile, "wb")
                f.write(pickle.dumps(data))
                f.close()
                print("Process Completed")
                print("click back for live attendance")
        def train():
                

                #initilizing of embedding & recognizer
                embeddingFile = "output/embeddings.pickle"
                #New & Empty at initial
                recognizerFile = "output/recognizer.pickle"
                labelEncFile = "output/le.pickle"

                print("Loading face embeddings...")
                data = pickle.loads(open(embeddingFile, "rb").read())

                print("Encoding labels...")
                labelEnc = LabelEncoder()
                labels = labelEnc.fit_transform(data["names"])


                print("Training model...")
                recognizer = SVC(C=1.0, kernel="linear", probability=True)
                recognizer.fit(data["embeddings"], labels)

                f = open(recognizerFile, "wb")
                f.write(pickle.dumps(recognizer))
                f.close()

                f = open(labelEncFile, "wb")
                f.write(pickle.dumps(labelEnc))
                f.close()
                print("model trained")

        dataset()
        preprocess()
        train()

register()



