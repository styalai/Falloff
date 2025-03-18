import arcade
import random

window = arcade.Window(title="Window")
window.center_window()

bg_start_color = (255, 255, 255, 95) # last=transparence

fg_start_color = [
    arcade.color.WHITE,
    arcade.color.BABY_BLUE,
    arcade.color.AQUA,
    arcade.color.BUFF,
    arcade.color.ALIZARIN_CRIMSON,
]

def create_starfield(batch, color, random_color=False):
    for i in range(200):
        x = random.randint(0, 1280)
        y = random.randint(0, 720)
        w = random.randint(1, 3)
        h = random.randint(1, 3)

        if random_color:
            color = random.choice(fg_start_color)
        star = arcade.shape_list.create_rectangle_filled(x, y, w, h, color)
        batch.append(star)


class GameView(arcade.View):
    def __init__(self):
        super().__init__()
        self.fg_star1 = arcade.shape_list.ShapeElementList()
        create_starfield(self.fg_star1, 0, random_color=True)

    def on_draw(self):
        self.clear()
        self.fg_star1.draw()

    def on_update(self, delta_time):
        pass

game = GameView()
window.show_view(game)
arcade.run()