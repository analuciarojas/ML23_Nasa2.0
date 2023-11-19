import datetime as dt
import pathlib
import matplotlib.pyplot as plt

file_path = pathlib.Path(__file__).parent.absolute()

# trazar y guardar gráficos de pérdidas
class PlotLosses():
    def __init__(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.fig, self.ax = plt.subplots()
        self.logs = []

    def on_epoch_end(self, epoch, train_loss, val_loss):        
        self.x.append(self.i)
        self.losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.i += 1
        self.ax.cla()
        self.ax.plot(self.x, self.losses, label="Costo de entrenamiento promedio")
        self.ax.plot(self.x, self.val_losses, label="Costo de validación promedio")
        self.ax.set_xlabel('epochs')
        self.ax.set_ylabel('Loss')
        self.ax.legend()
        plt.pause(0.1)

    def on_train_end(self):
        today = dt.datetime.now().strftime("%Y-%m-%d")
        losses_file = file_path / f'figures/losses_{today}.png'
        self.fig.savefig(losses_file)

# Example usage:
# plotter = PlotLosses()
# plotter.on_epoch_end(1, 0.5, 0.3)  # Call this at the end of each epoch
# plotter.on_train_end()  # Call this after training is complete
