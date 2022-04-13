"""
Map Class

Simple program to visualise the drone and its destination location.
"""
import arcade
import math

# Constants
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 1000
SCREEN_TITLE = "Map"


class Map(arcade.Window):
    """
    Main application class.
    """

    def __init__(self, dest_pos=(600, 700)):

        # Call the parent class and set up the window
        super().__init__(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_TITLE)

        # Separate variable that holds the drone sprite
        self.drone_sprite = None
        self.destination_point = None

        self.dest_pos = dest_pos
        self.reach_goal = False

        # Create a variable to hold our speed.
        self.speed = 0

        self.pos_direction = None
        self.rot_direction = None
        self.red_light = True
        self.target_angle_is_close = False

        self.point_list = ()

        arcade.set_background_color(arcade.csscolor.WHITE)

    def setup(self):
        """Set up the map here. Call this function to restart the map."""
        # Set up the drone, specifically placing it at these coordinates.
        drone_image_source = "Images/drone.png"
        self.drone_sprite = arcade.Sprite(drone_image_source, 0.1)
        self.drone_sprite.center_x = SCREEN_WIDTH / 2
        self.drone_sprite.center_y = SCREEN_HEIGHT / 2

        destination_image_source = "Images/finish-flag.png"
        self.destination_point = arcade.Sprite(destination_image_source, 0.06)
        self.destination_point.center_x = self.dest_pos[0]
        self.destination_point.center_y = self.dest_pos[1]

        # Reset value for restarting
        self.reach_goal = False
        self.point_list = ()

    def on_draw(self):
        """Render the screen."""

        # Clear the screen to the background color
        self.clear()

        # Draw a line strip
        arcade.draw_line_strip(self.point_list, arcade.color.RED, 3)

        # Code to draw the screen goes here
        self.destination_point.draw()
        self.drone_sprite.draw()

        # Put the text on the screen.
        output = f"Position: {self.drone_sprite.center_x}, {self.drone_sprite.center_y}"
        arcade.draw_text(output, 10, 20, arcade.color.BLACK, 14)

    def update(self, delta_time: float):

        if not self.reach_goal:

            # Add a line strip point
            new_pos = self.drone_sprite.position
            if new_pos not in self.point_list:
                point_list = list(self.point_list)
                point_list.append(new_pos)
                self.point_list = tuple(point_list)

            self.rot_direction = None
            self.target_angle_is_close = False

            # Position the start at our current location
            start_x = self.drone_sprite.center_x
            start_y = self.drone_sprite.center_y

            # Get the destination location
            dest_x = self.destination_point.center_x
            dest_y = self.destination_point.center_y

            # Do math to calculate how to get the sprite to the destination.
            # Calculation the angle in radians between the start points
            # and end points. This is the angle the player will travel.
            x_diff = dest_x - start_x
            y_diff = dest_y - start_y
            target_angle_radians = math.atan2(y_diff, x_diff)
            if target_angle_radians < 0:
                target_angle_radians += 2 * math.pi

            # What angle are we at now in radians?
            rot = open('E:\\GitHub\\Dissertation\\.temp\\rot').read()
            if rot != '':
                self.drone_sprite.angle = int(rot)
            actual_angle_radians = math.radians(self.drone_sprite.angle + 90)

            if actual_angle_radians > 7:
                actual_angle_radians -= 6

            # How fast can we rotate?
            rot_speed_radians = math.radians(self.speed)

            # What is the difference between what we want, and where we are?
            angle_diff_radians = target_angle_radians - actual_angle_radians

            if abs(angle_diff_radians) > 6.2 or abs(angle_diff_radians) < 0.1:
                self.target_angle_is_close = True

            # Figure out if we rotate clockwise or counter-clockwise
            if abs(angle_diff_radians) <= rot_speed_radians:
                # Close enough, let's set our angle to the target
                actual_angle_radians = target_angle_radians
                clockwise = None
            elif angle_diff_radians > 0 and abs(angle_diff_radians) < math.pi:
                clockwise = False
            elif angle_diff_radians > 0 and abs(angle_diff_radians) >= math.pi:
                clockwise = True
            elif angle_diff_radians < 0 and abs(angle_diff_radians) < math.pi:
                clockwise = True
            else:
                clockwise = False

            # Rotate the proper direction if needed
            if actual_angle_radians != target_angle_radians and clockwise:
                actual_angle_radians -= rot_speed_radians
                self.rot_direction = 'right'
            elif actual_angle_radians != target_angle_radians:
                actual_angle_radians += rot_speed_radians
                self.rot_direction = 'left'

            # Keep in a range of 0 to 2pi
            if actual_angle_radians > 2 * math.pi:
                actual_angle_radians -= 2 * math.pi
            elif actual_angle_radians < 0:
                actual_angle_radians += 2 * math.pi

            # Are we close to the correct angle? If so, move.
            # Angle tolerance: 22.5 degrees
            if abs(angle_diff_radians) < math.pi / 8:
                self.red_light = False

                if self.pos_direction == 'forward':
                    self.drone_sprite.center_x += math.cos(actual_angle_radians) * self.speed
                    self.drone_sprite.center_y += math.sin(actual_angle_radians) * self.speed

                elif self.pos_direction == 'left':
                    angle_rad = math.radians(self.drone_sprite.angle)
                    self.drone_sprite.center_x += -self.speed * math.cos(angle_rad)
                    self.drone_sprite.center_y += -self.speed * math.sin(angle_rad)

                elif self.pos_direction == 'right':
                    angle_rad = math.radians(self.drone_sprite.angle)
                    self.drone_sprite.center_x += self.speed * math.cos(angle_rad)
                    self.drone_sprite.center_y += self.speed * math.sin(angle_rad)
            else:
                self.red_light = True

            # When we reach to destination point's 10cm x 10cm square area.
            arrived = False
            # print('x: ', abs(self.drone_sprite.center_x - dest_x))
            # print('y: ', abs(self.drone_sprite.center_y - dest_y))
            if abs(self.drone_sprite.center_x - dest_x) < 10 and abs(self.drone_sprite.center_y - dest_y) < 10:
                self.drone_sprite.center_x = dest_x
                self.drone_sprite.center_y = dest_y
                arrived = True

            # If we have arrived, then cancel our destination point
            if arrived:
                # self.destination_point.remove_from_sprite_lists() # Destroy current goal and move on to next goal
                self.reach_goal = True

    def on_mouse_press(self, x, y, button, key_modifiers):
        """
        Called when the user presses a mouse button.
        """
        if button == arcade.MOUSE_BUTTON_LEFT:
            self.destination_point.position = (x, y)
            self.reach_goal = False

    def get_rot_direction(self):
        return self.rot_direction

    def set_pos_direction(self, direction):
        if direction == 'forward' and self.pos_direction in ['left', 'right']:
            self.pos_direction = direction
        if direction != 'forward' and self.pos_direction not in ['left', 'right']:
            self.pos_direction = direction

    def get_red_light_status(self):
        return self.red_light

    def get_reach_goal(self):
        return self.reach_goal

    def set_speed(self, speed):
        self.speed = speed

    def get_target_angle_is_close(self):
        return self.target_angle_is_close