import cv2

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

smile_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_smile.xml"
)


cam = cv2.VideoCapture(0)

print("Pulsa 'q' para salir")

while True:
    ret, frame = cam.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.3, minNeighbors=5
    )

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]

        smiles = smile_cascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.7,
            minNeighbors=20
        )

        # Determinar emoción
        if len(smiles) > 0:
            emocion = "FELIZ"
            color = (0, 255, 0)   # Verde
        else:
            emocion = "TRISTE"
            color = (255, 0, 0)   # Azul

        # Dibujar cuadro
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
        cv2.putText(
            frame,
            emocion,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            color,
            2
        )

    cv2.imshow("Detector de Emociones", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()

