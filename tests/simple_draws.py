import arcade

window = arcade.Window(title="Window")


class GameView(arcade.View):
    def __init__(self):
        super().__init__()

        self.batch = arcade.shape_list.ShapeElementList()

        # create circle
        cercle = arcade.shape_list.create_ellipse_filled(center_x=self.width//2, 
                                                         center_y=self.height//2,
                                                         width=60,
                                                         height=60,
                                                         color=(255, 102, 196)
                                                         )
        
        contour = arcade.shape_list.create_ellipse_outline(center_x=self.width//2, 
                                                           center_y=self.height//2, 
                                                           width=80,
                                                           height=80,
                                                           color=(255, 102, 196)
                                                           )
        
        # create triangle
        triangle = arcade.shape_list.create_polygon([(0,0), (100,0), (50,100)],
                                                    color=arcade.color.BLUE
                                                    )
        # create rectangle
        rect = arcade.shape_list.create_rectangle_filled(center_x=100, 
                                                              center_y=360,
                                                              width=100,
                                                              height=150,
                                                              color=arcade.color.VIOLET
                                                            )

        self.batch.append(cercle)
        self.batch.append(contour)
        self.batch.append(triangle)
        self.batch.append(rect)

    def on_draw(self): # run 60 time by second
        self.clear((56, 182, 255))
        self.batch.draw()
        
game = GameView()
window.show_view(game)
arcade.run()
