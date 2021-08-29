from tkinter import *
from PIL import ImageTk
import torch
from torchvision import transforms as T
from torchvision.utils import make_grid
from pl_bolts.models.gans import DCGAN
from typing import Callable

class Application:

    def __init__(self, ckpt_path: str) -> None:
        self.model = DCGAN.load_from_checkpoint(ckpt_path)
        self.latent_space = torch.randn(25, self.model.hparams.latent_dim)

        self.root = Tk()

        outer_slider_frame = Frame(self.root)
        slider_scroll = Scrollbar(outer_slider_frame)
        slider_canvas = Canvas(outer_slider_frame, bg="white",
                               height=640, width=600,
                               yscrollcommand=slider_scroll.set)
        slider_scroll.config(command=slider_canvas.yview)
        inner_slider_frame = Frame(outer_slider_frame)

        slider_scroll.pack(side=RIGHT, fill=Y)
        slider_canvas.pack(side=LEFT)
        outer_slider_frame.pack(side=LEFT)
        
        slider_canvas.create_window(
            (4, 4), window=inner_slider_frame, anchor=NW
        )
        
        inner_slider_frame.bind("<Configure>",
        lambda _: slider_canvas.configure(
            scrollregion=slider_canvas.bbox("all")
        ))

        for dim in range(self.model.hparams.latent_dim):
            slider = Scale(
                inner_slider_frame, from_=-4, to=4, orient=HORIZONTAL,
                length=600, showvalue=False, resolution=0.01,
                command=self.change_value_func(dim)
            )
            slider.set(self.latent_space[0, dim].item())
            slider.pack(side=BOTTOM)

        self.first_change = [True] * self.model.hparams.latent_dim

        self.flower_canvas = Canvas(self.root, bg="white",
                                    height=640, width=640)
        img = self.draw()
        self.img = self.flower_canvas.create_image(320, 320, image=img)
        self.flower_canvas.pack(side=RIGHT)

        mainloop()
        
    def change_value_func(self, slider_idx: int) -> Callable[[str], None]:
        def change_value(value: str) -> None:
            if self.first_change[slider_idx]:
                self.first_change[slider_idx] = False
                return
            for idx in range(25):
                self.latent_space[idx, slider_idx] = float(value) ** 3
            new_img = self.draw()
            self.flower_canvas.itemconfig(self.img, image=new_img)
        return change_value

    def draw(self) -> ImageTk.PhotoImage:
        img_raw = self.model(self.latent_space)
        img_raw = make_grid(img_raw, nrow=5, normalize=True)
        pipeline = T.Compose([
            T.Resize(640),
            T.ToPILImage()
        ])
        img_raw = pipeline(img_raw)

        self.root.img = img = ImageTk.PhotoImage(img_raw)
        return img

ckpt_path = "checkpoints\model_2.ckpt"
Application(ckpt_path)
