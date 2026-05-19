class Classifier:
    def __init__(self, output_control):
        self.output = output_control

    def execute(self, class_idx):
        if class_idx == 0:
            self.output.blinker_links()
        elif class_idx == 1:
            self.output.blinker_rechts()
        elif class_idx == 2:
            self.output.innenlicht_an()
        elif class_idx == 3:
            self.output.innenlicht_aus()
        elif class_idx == 4:
            self.output.licht_an()
        elif class_idx == 5:
            self.output.licht_aus()

