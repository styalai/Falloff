import arcade
from arcade.gui import UIManager, UITextureButton
from PIL import Image

window = arcade.Window(1080/2.4, 1920/2.4, "Falloff")
window.center_window()

class MenuView(arcade.View):
    def __init__(self):
        super().__init__()
        ### Player sprite
        self.player_s = 0.3
        self.player_x = self.width//2
        self.player_y = self.height - 150
        self.sprites = arcade.SpriteList()
        self.player = arcade.Sprite("assets/rectanglerose.webp",
                                    scale=self.player_s,
                                    angle=90
                                    )
        self.player.position = self.player_x, self.player_y
        self.sprites.append(self.player)
        
        self.manager = UIManager()
        ### Button humain
        self.button_h_h = 90
        self.button_h_w = 220
        self.texture_h = arcade.Texture(Image.open("assets/button_humain.png"))
        self.texture_h_pressed = arcade.Texture(Image.open("assets/button_humain_pressed.png"))
        self.humain = UITextureButton(
            x=(self.width-self.button_h_w)//2,
            y=50,
            width=self.button_h_w,
            height=self.button_h_h,
            texture=self.texture_h,
            texture_hovered=self.texture_h_pressed,
            texture_pressed=self.texture_h_pressed,
        )
        self.humain.on_click = self.on_click
        self.manager.add(self.humain)
    
    def on_click():
        print("click")
    
    def on_draw(self):
        self.clear((56, 182, 255))
        self.sprites.draw()
        self.manager.draw()


menu = MenuView()
window.show_view(menu)
arcade.run()