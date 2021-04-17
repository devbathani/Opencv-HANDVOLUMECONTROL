import  cv2
import mediapipe as mp
import  time

class handDetector():
    def __init__(self, mode=False, maxHands = 2, detectionCon = 0.7, trackCon=0.5):
        self.mode = mode
        self.maxhands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mphands = mp.solutions.hands
        self.hands = self.mphands.Hands(self.mode,
                                        self.maxhands,
                                        self.detectionCon,
                                        self.trackCon)
        self.mpdraw = mp.solutions.drawing_utils


    def findHands(self, img, draw=True):
         imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
         self.results = self.hands.process(imgRGB)
         # print(results.multi_hand_landmarks)

         if self.results.multi_hand_landmarks:
             for handLms in self.results.multi_hand_landmarks:
                 if draw:
                    self.mpdraw.draw_landmarks(img, handLms, self.mphands.HAND_CONNECTIONS)
         return img

    def findposition(self, img, handNo = 0 , draw = True):

        lmlist = []
        if self.results.multi_hand_landmarks:
            myhand = self.results.multi_hand_landmarks[handNo]

            for id, lm in enumerate(myhand.landmark):
              # print(id,lm)
              h, w, c = img.shape
              cx, cy = int(lm.x * w), int(lm.y * h)
              print(id, cx, cy)
              lmlist.append([id,cx,cy])
              if draw:
                 cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

        return  lmlist








def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()

    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmlist = detector.findposition(img)
        if len(lmlist) != 0:
            print(lmlist[4])

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, f'FPS : {int(fps)}', (30, 50), cv2.FONT_ITALIC, 2, (255, 0, 0), 3)

        cv2.imshow("IMAGE", img)
        cv2.waitKey(1)





if __name__ == "__main__":
    main()
