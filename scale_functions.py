

class ScaleFunctions:
    def __init__(self, y_min, y_max):
        self.y_min = y_min
        self.y_max = y_max

    def scale_logistic(self, y):
        return (y - self.y_min) / (self.y_max - self.y_min)

    def scale_tanh(self, y):
        return 2.0 * ((y - self.y_min) / (self.y_max - self.y_min)) - 1.0

    def descale_logistic(self, y_scaled):
        return y_scaled * (self.y_max - self.y_min) + self.y_min

    def descale_tanh(self, y_scaled):
        return ((y_scaled + 1.0) / 2.0) * (self.y_max - self.y_min) + self.y_min