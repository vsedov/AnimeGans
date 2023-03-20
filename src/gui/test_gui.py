import sys

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


class AnimeGenerationWindow(QWidget):
    def __init__(self):
        super().__init__()

        # Initialize the UI
        self.initUI()

    def initUI(self):
        # Create the input fields and buttons
        type_label = QLabel("Type of anime generation:")
        self.type_input = QComboBox()
        self.type_input.addItems(
            [
                "fix_noise",
                "fix_hair_eye",
                "change_hair",
                "change_eye",
                "interpolate",
                "generate",
            ]
        )

        hair_label = QLabel("Hair color:")
        self.hair_input = QComboBox()
        self.hair_input.addItems(hair_mapping)

        eye_label = QLabel("Eye color:")
        self.eye_input = QComboBox()
        self.eye_input.addItems(eye_mapping)

        sample_dir_label = QLabel("Output directory:")
        self.sample_dir_input = QLineEdit()
        self.sample_dir_input.setText(f"{hc.DIR}results/generated")

        batch_size_label = QLabel("Batch size:")
        self.batch_size_input = QLineEdit()
        self.batch_size_input.setText("64")

        epoch_label = QLabel("Epoch:")
        self.epoch_input = QLineEdit()
        self.epoch_input.setText("100")

        check_point_number_label = QLabel("Checkpoint number:")
        self.check_point_number_input = QLineEdit()
        self.check_point_number_input.setText("100")

        extra_generator_layers_label = QLabel("Extra generator layers:")
        self.extra_generator_layers_input = QLineEdit()
        self.extra_generator_layers_input.setText("1")

        range_label = QLabel("Range of checkpoint numbers:")
        self.range_input = QLineEdit()

        num_images_label = QLabel("Number of images:")
        self.num_images_input = QLineEdit()
        self.num_images_input.setText("2")

        image_size_label = QLabel("Image size:")
        self.image_size_input = QLineEdit()
        self.image_size_input.setText("128")

        saturation_label = QLabel("Saturation:")
        self.saturation_input = QLineEdit()
        self.saturation_input.setText("0.9")

        qual_label = QLabel("Qual:")
        self.qual_input = QLineEdit()
        self.qual_input.setText("0.9")

        generate_button = QPushButton("Generate Images")
        generate_button.clicked.connect(self.generate_images)

        # Create the layout
        layout = QGridLayout()
        layout.addWidget(type_label, 0, 0)
        layout.addWidget(self.type_input, 0, 1)

        layout.addWidget(hair_label, 1, 0)
        layout.addWidget(self.hair_input, 1, 1)

        layout.addWidget(eye_label, 2, 0)
        layout.addWidget(self.eye_input, 2, 1)

        layout.addWidget(sample_dir_label, 3, 0)
        layout.addWidget(self.sample_dir_input, 3, 1)

        layout.addWidget(batch_size_label, 4, 0)
        layout.addWidget(self.batch_size_input, 4, 1)

        layout.addWidget(epoch_label, 5, 0)
        layout.addWidget(self.epoch_input, 5, 1)
        layout.addWidget(check_point_number_label, 6, 0)
        layout.addWidget(self.check_point_number_input, 6, 1)

        layout.addWidget(extra_generator_layers_label, 7, 0)
        layout.addWidget(self.extra_generator_layers_input, 7, 1)

        layout.addWidget(range_label, 8, 0)
        layout.addWidget(self.range_input, 8, 1)

        layout.addWidget(num_images_label, 9, 0)
        layout.addWidget(self.num_images_input, 9, 1)

        layout.addWidget(image_size_label, 10, 0)
        layout.addWidget(self.image_size_input, 10, 1)

        layout.addWidget(saturation_label, 11, 0)
        layout.addWidget(self.saturation_input, 11, 1)

        layout.addWidget(qual_label, 12, 0)
        layout.addWidget(self.qual_input, 12, 1)

        layout.addWidget(generate_button, 13, 0, 1, 2)

        self.setLayout(layout)

        # Set the window properties
        self.setWindowTitle("Anime Generation Tool")
        self.setGeometry(100, 100, 500, 500)
        self.show()

    def generate_images(self):
        # Get the input values from the UI
        type = self.type_input.currentText()
        hair = self.hair_input.currentText()
        eye = self.eye_input.currentText()
        sample_dir = self.sample_dir_input.text()
        batch_size = int(self.batch_size_input.text())
        epoch = int(self.epoch_input.text())
        check_point_number = self.check_point_number_input.text()
        extra_generator_layers = int(self.extra_generator_layers_input.text())
        range = self.range_input.text()
        num_images = int(self.num_images_input.text())
        image_size = int(self.image_size_input.text())
        saturation = float(self.saturation_input.text())
        qual = float(self.qual_input.text())

        # Call the main function with the input values
        args = namedtuple(
            "Args",
            [
                "type",
                "hair",
                "eye",
                "sample_dir",
                "batch_size",
                "epoch",
                "check_point_number",
                "extra_generator_layers",
                "range",
                "num_images",
                "image_size",
                "saturation",
                "qual",
                "gen_model_dir",
            ],
        )
        args = args(
            type,
            hair,
            eye,
            sample_dir,
            batch_size,
            epoch,
            check_point_number,
            extra_generator_layers,
            range,
            num_images,
            image_size,
            saturation,
            qual,
        )

        args = parser.parse_args()
        if args.check_point_number == "best":
            args.gen_model_dir = f"{hc.DIR}results/checkpoints/ACGAN-[{args.batch_size}]-[{args.epoch}]/G_best_.ckpt"
        else:
            args.gen_model_dir = f"{hc.DIR}results/checkpoints/ACGAN-[{args.batch_size}]-[{args.epoch}]/G_{args.check_point_number}.ckpt"

        main(args)


def run():
    app = QApplication(sys.argv)
    gui = AnimeGenerationWindow()
    gui.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    run()
