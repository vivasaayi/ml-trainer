class BaseKerasDataSet():
    def __init__(self):
        print("Initializing DataSet")
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None

    def load_data(self):
        print("This function should be overridden")