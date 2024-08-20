import tkinter as tk
from PIL import Image, ImageDraw
import mnist_loader
from Network import Network
import numpy as np

class DigitClassifierApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Draw a digit")

        self.canvas = tk.Canvas(self, width=280, height=280, bg="black")
        self.canvas.pack()

        self.button_clear = tk.Button(self, text="Clear", command=self.clear_canvas)
        self.button_clear.pack()

        self.button_predict = tk.Button(self, text="Predict", command=self.predict_digit)
        self.button_predict.pack()

        self.label_result = tk.Label(self, text="Draw a digit and click Predict", font=("Helvetica", 16))
        self.label_result.pack()

        self.canvas.bind("<B1-Motion>", self.paint)
        self.image = Image.new("L", (28, 28), color=0)  # Imagen en escala de grises, fondo negro
        self.draw = ImageDraw.Draw(self.image)

    def paint(self, event):
        x1, y1 = (event.x - 10), (event.y - 10)
        x2, y2 = (event.x + 10), (event.y + 10)
        self.canvas.create_oval(x1, y1, x2, y2, fill="white", outline="white")
        self.draw.ellipse([x1/10, y1/10, x2/10, y2/10], fill="white", outline="white")

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (28, 28), color=0)
        self.draw = ImageDraw.Draw(self.image)
        self.label_result.config(text="Draw a digit and click Predict")

    def predict_digit(self):
        image = self.image.resize((28, 28)).convert("L")
        image = np.array(image).reshape(784, 1)
        image = image.astype(np.float32)
        image /= 255.0  # Normaliza los valores a [0, 1]

        prediction = np.argmax(net.feedforward(image))
        self.label_result.config(text=f"Prediction: {prediction}")

if __name__ == "__main__":
    # Cargar datos y entrenar la red
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    net = Network([784, 20, 10])
    net.SGD(training_data, 15, 10, 3.0, test_data=test_data)
    
    app = DigitClassifierApp()
    app.mainloop()
