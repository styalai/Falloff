import arcade
from arcade.gui import UIManager, UIFlatButton

window = arcade.Window(1080/2.4, 1920/2.4, "Falloff")
window.center_window()

class GameView(arcade.View):
    def __init__(self):
        super().__init__()
        ### Player sprite
        self.player_s = 0.3
        self.player_x = self.width//2
        self.player_y = self.height - 150
        self.sprites = arcade.SpriteList()
        self.player = arcade.Sprite("../assets/rectanglerose.webp",
                                    scale=self.player_s,
                                    angle=90
                                    )
        self.player.position = self.player_x, self.player_y
        self.sprites.append(self.player)
        
        ### Button
        self.button_h = 70
        self.button_w = 190
        self.manager = UIManager()
        self.humain = UIFlatButton(
            x=(self.width-self.button_w)//2,
            y=10,
            width=self.button_w,
            height=self.button_h,
            color=(92, 225, 230),
        )
        self.manager.add(self.humain)
    
    def on_draw(self):
        self.clear((56, 182, 255))
        self.sprites.draw()
        self.manager.draw()

game = GameView()
window.show_view(game)
arcade.run()