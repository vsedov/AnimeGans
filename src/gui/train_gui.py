import sys

from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QGridLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QWidget,
)

# Define the default options
from src.train import main, parse_args

default_options = parse_args()


class GeneratorGUI(QWidget):
    def __init__(self):
        super().__init__()

        # Create the labels and input widgets for each option
        self.iterations_label = QLabel("Iterations:")
        self.iterations_input = QLineEdit(str(default_options.iterations))

        self.extra_generator_layers_label = QLabel("Extra Generator Layers:")
        self.extra_generator_layers_input = QLineEdit(
            str(default_options.extra_generator_layers)
        )

        self.extra_discriminator_layers_label = QLabel(
            "Extra Discriminator Layers:"
        )
        self.extra_discriminator_layers_input = QLineEdit(
            str(default_options.extra_discriminator_layers)
        )

        self.cp_per_save_label = QLabel("Checkpoints per Save:")
        self.cp_per_save_input = QLineEdit(str(default_options.cp_per_save))

        self.batch_size_label = QLabel("Batch Size:")
        self.batch_size_input = QLineEdit(str(default_options.batch_size))

        self.sample_dir_label = QLabel("Sample Directory:")
        self.sample_dir_input = QLineEdit(default_options.sample_dir)

        self.checkpoint_dir_label = QLabel("Checkpoint Directory:")
        self.checkpoint_dir_input = QLineEdit(default_options.checkpoint_dir)

        self.sample_label = QLabel("Sample every n steps:")
        self.sample_input = QLineEdit(str(default_options.sample))

        self.lr_label = QLabel("Learning Rate:")
        self.lr_input = QLineEdit(str(default_options.lr))

        self.beta_label = QLabel("Momentum term in Adam optimizer:")
        self.beta_input = QLineEdit(str(default_options.beta))

        self.lambda_gp_label = QLabel("Gradient Penalty Lambda:")
        self.lambda_gp_input = QLineEdit(str(default_options.lambda_gp))

        self.wandb_checkbox = QCheckBox("Use wandb")
        self.wandb_checkbox.setChecked(default_options.wandb == "true")

        self.wandb_project_label = QLabel("Wandb Project:")
        self.wandb_project_input = QLineEdit(default_options.wandb_project)

        self.wandb_name_label = QLabel("Wandb Name:")
        self.wandb_name_input = QLineEdit(default_options.wandb_name)

        self.overwrite_label = QLabel("Overwrite:")
        self.overwrite_input = QLineEdit(default_options.overwrite)

        self.extra_train_model_type_label = QLabel("Extra train model type:")
        self.extra_train_model_type_input = QLineEdit(
            default_options.extra_train_model_type
        )

        self.generate_button = QPushButton("Generate Images")
        self.generate_button.clicked.connect(self.generate_images)

        # Set the window properties
        self.setWindowTitle("Generator GUI")
        self.setGeometry(100, 100, 800, 600)

        # Add the widgets to the layout
        grid = QGridLayout()
        grid.addWidget(self.iterations_label, 0, 0)
        grid.addWidget(self.iterations_input, 0, 1)
        grid.addWidget(self.extra_generator_layers_label, 1, 0)
        grid.addWidget(self.extra_generator_layers_input, 1, 1)
        grid.addWidget(self.extra_discriminator_layers_label, 2, 0)
        grid.addWidget(self.extra_discriminator_layers_input, 2, 1)
        grid.addWidget(self.cp_per_save_label, 3, 0)
        grid.addWidget(self.cp_per_save_input, 3, 1)
        grid.addWidget(self.batch_size_label, 4, 0)
        grid.addWidget(self.batch_size_input, 4, 1)
        grid.addWidget(self.sample_dir_label, 5, 0)
        grid.addWidget(self.sample_dir_input, 5, 1)
        grid.addWidget(self.checkpoint_dir_label, 6, 0)
        grid.addWidget(self.checkpoint_dir_input, 6, 1)
        grid.addWidget(self.sample_label, 7, 0)
        grid.addWidget(self.sample_input, 7, 1)
        grid.addWidget(self.lr_label, 8, 0)
        grid.addWidget(self.lr_input, 8, 1)
        grid.addWidget(self.beta_label, 9, 0)
        grid.addWidget(self.beta_input, 9, 1)
        grid.addWidget(self.lambda_gp_label, 10, 0)
        grid.addWidget(self.lambda_gp_input, 10, 1)
        grid.addWidget(self.wandb_checkbox, 11, 0)
        grid.addWidget(self.wandb_project_label, 12, 0)
        grid.addWidget(self.wandb_project_input, 12, 1)
        grid.addWidget(self.wandb_name_label, 13, 0)
        grid.addWidget(self.wandb_name_input, 13, 1)
        grid.addWidget(self.overwrite_label, 14, 0)
        grid.addWidget(self.overwrite_input, 14, 1)
        grid.addWidget(self.extra_train_model_type_label, 15, 0)
        grid.addWidget(self.extra_train_model_type_input, 15, 1)
        grid.addWidget(self.generate_button, 16, 0, 1, 2)

        self.setLayout(grid)

    def generate_images(self):
        # Get the values from the input widgets
        iterations = int(self.iterations_input.text())
        extra_generator_layers = int(self.extra_generator_layers_input.text())
        extra_discriminator_layers = int(
            self.extra_discriminator_layers_input.text()
        )
        cp_per_save = int(self.cp_per_save_input.text())
        batch_size = int(self.batch_size_input.text())
        sample_dir = self.sample_dir_input.text()
        checkpoint_dir = self.checkpoint_dir_input.text()
        sample = int(self.sample_input.text())
        lr = float(self.lr_input.text())
        beta = float(self.beta_input.text())
        lambda_gp = float(self.lambda_gp_input.text())
        wandb_enabled = "true" if self.wandb_checkbox.isChecked() else "false"
        wandb_project = self.wandb_project_input.text()
        wandb_name = self.wandb_name_input.text()
        overwrite = self.overwrite_input.text()
        extra_train_model_type = self.extra_train_model_type_input.text()

        # TODO: Call the generator with the given arguments
        print("Generating images with the following options:")
        print(f"Iterations: {iterations}")
        print(f"Extra Generator Layers: {extra_generator_layers}")
        print(f"Extra Discriminator Layers: {extra_discriminator_layers}")
        print(f"Checkpoints per Save: {cp_per_save}")
        print(f"Batch Size: {batch_size}")
        print(f"Sample Directory: {sample_dir}")
        print(f"Checkpoint Directory: {checkpoint_dir}")
        print(f"Sample every n steps: {sample}")
        print(f"Learning Rate: {lr}")
        print(f"Momentum term in Adam optimizer: {beta}")
        print(f"Gradient Penalty Lambda: {lambda_gp}")
        print(f"Wandb enabled: {wandb_enabled}")
        print(f"Wandb Project: {wandb_project}")
        print(f"Wandb Name: {wandb_name}")
        print(f"Overwrite: {overwrite}")
        print(f"Extra train model type: {extra_train_model_type}")

        args = {
            "iterations": iterations,
            "extra_generator_layers": extra_generator_layers,
            "extra_discriminator_layers": extra_discriminator_layers,
            "cp_per_save": cp_per_save,
            "batch_size": batch_size,
            "sample_dir": sample_dir,
            "checkpoint_dir": checkpoint_dir,
            "sample": sample,
            "lr": lr,
            "beta": beta,
            "lambda_gp": lambda_gp,
            "wandb_enabled": wandb_enabled,
            "wandb_project": wandb_project,
            "wandb_name": wandb_name,
            "overwrite": overwrite,
            "extra_train_model_type": extra_train_model_type,
        }

        main(args)


if __name__ == "__main__":
    # Initialize the application and GUI
    app = QApplication(sys.argv)
    generator_gui = GeneratorGUI()
    generator_gui.show()
    sys.exit(app.exec_())
