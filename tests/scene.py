class Scene(arcade.Scene):
    def __init__(self, scene):
        super().__init__()
        self.scene = scene
        self.player_s = 0.5
        self.player_x = self.width//2
        self.player_y = self.hieght - 10
        
        self.sprites = arcade.SpriteList()
        
        self.player = arcade.Sprite("../assets/rectanglerose.webp",
                                    scale=self.player_s,
                                    angle=90
                                    )
        self.player.position = 0, 0
        
        self.sprites.append(self.player)
    
    def draw(self):
        super().draw()
        self.sprites.draw()
        
        
    def update(self, delta_time):
        super().update(delta_time)
