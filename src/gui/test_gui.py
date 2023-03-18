import sys
from typing import NamedTuple

from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QGridLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QWidget,
)

from src.core import hc
from src.test import main
from src.utils.torch_utils import *


class Args(NamedTuple):
    type: str
    hair: str
    eye: str
    sample_dir: str
    batch_size: int
    epoch: int
    check_point_number: str
    extra_generator_layers: int
    gen_model_dir: str


class GeneratorGUI(QWidget):
    def __init__(self):
        super().__init__()

        # Initialize the input widgets
        self.type_label = QLabel("Type:")
        # self.type_input = QLineEdit()
        self.type_input = QComboBox()
        self.type_input.addItems(
            [
                "fix_noise",
                "fix_hair_eye",
                "change_hair",
                "change_eye",
                "interpolate",
            ]
        )

        self.type_input.setText("fix_noise")
        self.hair_label = QLabel("Hair:")
        # self.hair_input = QLineEdit()
        # self.hair_input.setText(hair_mapping[2])
        self.hair_input = QComboBox()
        self.hair_input.addItems(hair_mapping)

        self.eye_label = QLabel("Eye:")
        self.eye_input = QComboBox()
        self.eye_input.addItems(eye_mapping)

        self.sample_dir_label = QLabel("Sample Directory:")
        self.sample_dir_input = QLineEdit()
        self.sample_dir_input.setText(f"{hc.DIR}results/generated")
        self.batch_size_label = QLabel("Batch Size:")
        self.batch_size_input = QLineEdit()
        self.batch_size_input.setText("64")
        self.epoch_label = QLabel("Epoch:")
        self.epoch_input = QLineEdit()
        self.epoch_input.setText("100")
        self.check_point_number_label = QLabel("Check Point Number:")
        self.check_point_number_input = QLineEdit()
        self.check_point_number_input.setText("100")
        self.extra_generator_layers_label = QLabel("Extra Generator Layers:")
        self.extra_generator_layers_input = QLineEdit()
        self.extra_generator_layers_input.setText("1")
        self.generate_button = QPushButton("Generate")
        self.generate_button.clicked.connect(self.generate_images)

        # Create the layout and add the widgets
        grid = QGridLayout()
        grid.addWidget(self.type_label, 0, 0)
        grid.addWidget(self.type_input, 0, 1)
        grid.addWidget(self.hair_label, 1, 0)
        grid.addWidget(self.hair_input, 1, 1)
        grid.addWidget(self.eye_label, 2, 0)
        grid.addWidget(self.eye_input, 2, 1)
        grid.addWidget(self.sample_dir_label, 3, 0)
        grid.addWidget(self.sample_dir_input, 3, 1)
        grid.addWidget(self.batch_size_label, 4, 0)
        grid.addWidget(self.batch_size_input, 4, 1)
        grid.addWidget(self.epoch_label, 5, 0)
        grid.addWidget(self.epoch_input, 5, 1)
        grid.addWidget(self.check_point_number_label, 6, 0)
        grid.addWidget(self.check_point_number_input, 6, 1)
        grid.addWidget(self.extra_generator_layers_label, 7, 0)
        grid.addWidget(self.extra_generator_layers_input, 7, 1)
        grid.addWidget(self.generate_button, 8, 0, 1, 2)

        self.setLayout(grid)

    def generate_images(self):
        # Get the values from the input widgets
        type = self.type_input.text()
        hair = self.hair_input.text()
        eye = self.eye_input.text()
        sample_dir = self.sample_dir_input.text()
        batch_size = int(self.batch_size_input.text())
        epoch = int(self.epoch_input.text())
        check_point_number = self.check_point_number_input.text()
        extra_generator_layers = int(self.extra_generator_layers_input.text())

        # Compute the generator model directory based on the input values
        if check_point_number == "best":
            gen_model_dir = f"{hc.DIR}results/checkpoints/ACGAN-[{batch_size}]-[{epoch}]/G_best_.ckpt"
        else:
            gen_model_dir = f"{hc.DIR}results/checkpoints/ACGAN-[{batch_size}]-[{epoch}]/G_{check_point_number}.ckpt"

        # Create the arguments object
        args = Args(
            type=type,
            hair=hair,
            eye=eye,
            sample_dir=sample_dir,
            batch_size=batch_size,
            epoch=epoch,
            check_point_number=check_point_number,
            extra_generator_layers=extra_generator_layers,
            gen_model_dir=gen_model_dir,
        )

        main(args)


def run():
    app = QApplication(sys.argv)
    gui = GeneratorGUI()
    gui.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    run()
