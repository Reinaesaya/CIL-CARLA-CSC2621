#!/usr/bin/env python3

# Copyright (c) 2017 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# Keyboard controlling for CARLA. Please refer to client_example.py for a simpler
# and more documented example.

"""
Welcome to CARLA manual control.

Use ARROWS or WASD keys for control.

    W            : throttle
    S            : brake
    AD           : steer
    Q            : toggle reverse
    Space        : hand-brake
    P            : toggle autopilot

    R            : restart level

STARTING in a moment...
"""

from __future__ import print_function

import argparse
import logging
import random
import time
import math
import os
import h5py
script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in

from configparser import SafeConfigParser

try:
    import pygame
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

from configparser import SafeConfigParser

from carla import image_converter
from carla import sensor
from carla.client import make_carla_client, VehicleControl
from carla.planner.map import CarlaMap
from carla.settings import CarlaSettings
from carla.tcp import TCPConnectionError
from carla.util import print_over_same_line

WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
MINI_WINDOW_WIDTH = 320
MINI_WINDOW_HEIGHT = 180
H5_WINDOW_WIDTH = 200
H5_WINDOW_HEIGHT = 88

class HumanDriver(object):
    def __init__(self):
        pygame.init()
        pygame.joystick.init()
        self.js = pygame.joystick.Joystick(0)
        self.js.init()
        self._is_on_reverse = False
        self.control = VehicleControl()
        self.parser = SafeConfigParser()
        self.parser.read('wheel_config.ini')
        self.steer_idx = int(self.parser.get('G29 Racing Wheel', 'steering_wheel'))
        self.throttle_idx = int(self.parser.get('G29 Racing Wheel', 'throttle'))
        self.brake_idx = int(self.parser.get('G29 Racing Wheel', 'brake'))
        self.reverse_idx = int(self.parser.get('G29 Racing Wheel', 'reverse'))
        self.handbrake_idx = int(self.parser.get('G29 Racing Wheel', 'handbrake'))

        # ADDITION: Added high level command inputs to be taken from joystick
        self.cmd_left_idx = int(self.parser.get('G29 Racing Wheel', 'cmd_left'))
        self.cmd_right_idx = int(self.parser.get('G29 Racing Wheel', 'cmd_right'))
        self.cmd_straight_idx = int(self.parser.get('G29 Racing Wheel', 'cmd_straight'))
        self.cmd = 2

        self.savetoggle = int(self.parser.get('G29 Racing Wheel', 'save_toggle'))
        self._savemode = False
        
    def reset_joystick(self):
        self.js = pygame.joystick.Joystick(0)
        self.js.init()
        jsInit = self.js.get_init()
        jsId = self.js.get_id()
        axis = self.js.get_axis(1)
        print('Joystick ID: %d Init status: %s Axis(1): %d' % (jsId, jsInit, axis))

    def computeControl(self):
        pygame.event.pump()
        numAxes = self.js.get_numaxes()
        jsInputs = [ float(self.js.get_axis(i)) for i in range(numAxes)]
        #print (jsInputs)
        jsButtons = [ float(self.js.get_button(i)) for i in range(self.js.get_numbuttons())]
        #print(jsButtons)

        # print('Steering angle: %f Throttle: %f Brake: %f' % (jsInputs[self.steer_idx], jsInputs[self.throttle_idx], jsInputs[self.brake_idx]))

        # Custom function to map range of inputs [1, -1] to outputs [0, 1] i.e 1 from inputs means nothing is pressed
        # For the steering, it seems fine as it is
        steerCmd = 0.55 * math.tan(1.1 * jsInputs[self.steer_idx])
        self.control.steer = steerCmd

        throttleCmd = 1.6 + (2.05 * math.log10(-0.7 * jsInputs[self.throttle_idx] + 1.4) - 1.2)/0.92
        if throttleCmd <= 0:
            throttleCmd = 0
        elif throttleCmd > 1:
            throttleCmd = 1

        brakeCmd = 1.6 + (2.05 * math.log10(-0.7 * jsInputs[self.brake_idx] + 1.4) - 1.2)/0.92
        if brakeCmd <= 0:
            brakeCmd = 0
        elif brakeCmd > 1:
            brakeCmd = 1

        #print("Steer Cmd, ", steerCmd, "Brake Cmd", brakeCmd, "ThrottleCmd", throttleCmd)
        self.control.brake = brakeCmd
        self.control.throttle = throttleCmd
        #print(len(jsButtons))
        toggle = jsButtons[self.reverse_idx]
        #print('reverse: %d toggle: %d' % (self._is_on_reverse, toggle))

        if toggle == 1:
            self._is_on_reverse += 1
        if self._is_on_reverse%2 == 0:
            self.control.reverse = 0
        if self._is_on_reverse > 1:
            self._is_on_reverse = 0

        if self._is_on_reverse:
            self.control.reverse = 1

        self.control.hand_brake = jsButtons[self.handbrake_idx]

        # ADDITION: Keep track of high-level commands from joystick
        if jsButtons[self.cmd_left_idx]:
            self.cmd = 3
        elif jsButtons[self.cmd_right_idx]:
            self.cmd = 4
        elif jsButtons[self.cmd_straight_idx]:
            self.cmd = 5
        else:
            self.cmd = 2

        savetoggle = jsButtons[self.savetoggle]
        if savetoggle == 1:
            self._savemode = True
        else:
            self._savemode = False

        # if self.control is None:
        #     self.reset_joystick()
        # else:
        return self.control

def make_carla_settings(args):
    """Make a CarlaSettings object with the settings we need."""
    settings = CarlaSettings()
    settings.set(
        SynchronousMode=False,
        SendNonPlayerAgentsInfo=True,
        NumberOfVehicles=10,
        NumberOfPedestrians=15,
        WeatherId=random.choice([1, 3, 7, 8, 14]),
        QualityLevel=args.quality_level)
    settings.randomize_seeds()
    camera0 = sensor.Camera('CameraRGB')
    camera0.set_image_size(WINDOW_WIDTH, WINDOW_HEIGHT)
    camera0.set_position(2.0, 0.0, 1.4)
    camera0.set_rotation(0.0, 0.0, 0.0)
    settings.add_sensor(camera0)
    camera1 = sensor.Camera('CameraDepth', PostProcessing='Depth')
    camera1.set_image_size(MINI_WINDOW_WIDTH, MINI_WINDOW_HEIGHT)
    camera1.set_position(2.0, 0.0, 1.4)
    camera1.set_rotation(0.0, 0.0, 0.0)
    settings.add_sensor(camera1)
    camera2 = sensor.Camera('CameraSemSeg', PostProcessing='SemanticSegmentation')
    camera2.set_image_size(MINI_WINDOW_WIDTH, MINI_WINDOW_HEIGHT)
    camera2.set_position(2.0, 0.0, 1.4)
    camera2.set_rotation(0.0, 0.0, 0.0)
    settings.add_sensor(camera2)
    camera3 = sensor.Camera('CameraRGB_h5')
    camera3.set_image_size(H5_WINDOW_WIDTH, H5_WINDOW_HEIGHT)
    camera3.set_position(2.0, 0.0, 1.4)
    camera3.set_rotation(0.0, 0.0, 0.0)
    settings.add_sensor(camera3)
    if args.lidar:
        lidar = sensor.Lidar('Lidar32')
        lidar.set_position(0, 0, 2.5)
        lidar.set_rotation(0, 0, 0)
        lidar.set(
            Channels=32,
            Range=50,
            PointsPerSecond=100000,
            RotationFrequency=10,
            UpperFovLimit=10,
            LowerFovLimit=-30)
        settings.add_sensor(lidar)
    return settings

class Timer(object):
    def __init__(self):
        self.step = 0
        self._lap_step = 0
        self._lap_time = time.time()

    def tick(self):
        self.step += 1

    def lap(self):
        self._lap_step = self.step
        self._lap_time = time.time()

    def ticks_per_second(self):
        return float(self.step - self._lap_step) / self.elapsed_seconds_since_lap()

    def elapsed_seconds_since_lap(self):
        return time.time() - self._lap_time


class CarlaGame(object):
    def __init__(self, carla_client, args):
        self.client = carla_client
        self._carla_settings = make_carla_settings(args)
        self._timer = None
        self._display = None
        self._main_image = None
        self._mini_view_image1 = None
        self._mini_view_image2 = None
        self._enable_autopilot = args.autopilot
        self._lidar_measurement = None
        self._map_view = None
        self._is_on_reverse = 0
        self._city_name = args.map_name
        self._map = CarlaMap(self._city_name, 16.43, 50.0) if self._city_name is not None else None
        self._map_shape = self._map.map_image.shape if self._city_name is not None else None
        self._map_view = self._map.get_map(WINDOW_HEIGHT) if self._city_name is not None else None
        self._position = None
        self._agent_positions = None
        self.rev_count = 0

        # ADDITION: Limit H5 files to 200 images each, keep track of image and measurement arrays
        self._countFrames = 0
        self._globalCount = 0
        self._sizeBatch = 200
        self._numTarget = 28
        self._h5_image = None
        self._image_array = np.zeros((self._sizeBatch,88,200,3))
        self._meas_array = np.zeros((self._sizeBatch,self._numTarget))

    def execute(self):
        """Launch the PyGame."""
        pygame.init()
        self._initialize_game()
        try:
            while True:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return
                self._on_loop()
                self._on_render()
        finally:
            pygame.quit()

    def _initialize_game(self):
        if self._city_name is not None:
            self._display = pygame.display.set_mode(
                (WINDOW_WIDTH + int((WINDOW_HEIGHT/float(self._map.map_image.shape[0]))*self._map.map_image.shape[1]), WINDOW_HEIGHT),
                pygame.HWSURFACE | pygame.DOUBLEBUF)
        else:
            self._display = pygame.display.set_mode(
                (WINDOW_WIDTH, WINDOW_HEIGHT),
                pygame.HWSURFACE | pygame.DOUBLEBUF)

        logging.debug('pygame started')
        self._on_new_episode()

    def _on_new_episode(self):
        self._carla_settings.randomize_seeds()
        self._carla_settings.randomize_weather()
        scene = self.client.load_settings(self._carla_settings)
        number_of_player_starts = len(scene.player_start_spots)
        player_start = np.random.randint(number_of_player_starts)
        print('Starting new episode...')
        self.client.start_episode(player_start)
        self._timer = Timer()
        # self._is_on_reverse = False
		# # Adding Joystick controls here
        # self.js = pygame.joystick.Joystick(0)
        # self.js.init()
        # axis = self.js.get_axis(1)
        # jsInit = self.js.get_init()
        # jsId = self.js.get_id()
        # print('Joystick ID: %d Init status: %s Axis(1): %d' % (jsId, jsInit, axis))

    def _on_loop(self):
        self._timer.tick()

        measurements, sensor_data = self.client.read_data()

        # ADDITION: Get 200 images for H5 files and extract images and measurements to put in them
        self._main_image = sensor_data.get('CameraRGB', None)
        self._mini_view_image1 = sensor_data.get('CameraDepth', None)
        self._mini_view_image2 = sensor_data.get('CameraSemSeg', None)
        self._lidar_measurement = sensor_data.get('Lidar32', None)
        self._h5_image = sensor_data.get('CameraRGB_h5', None)

        # Initiate measurement and image arrays
        if self._countFrames == self._sizeBatch:
            print ("Saving h5 {}".format(self._globalCount))
            rel_path = "h5/train{}.h5".format(self._globalCount)
            abs_file_path = os.path.join(script_dir, rel_path)     

            with h5py.File(abs_file_path, 'w') as f:
                dset_image = f.create_dataset("rgb", data=self._image_array, dtype='u1')
                dset_meas = f.create_dataset("targets", data=self._meas_array, dtype='f4')

            self._countFrames = 0
            self._globalCount += 1
            self._image_array = np.zeros((self._sizeBatch,88,200,3))
            self._meas_array = np.zeros((self._sizeBatch,self._numTarget))

        # numAxes = self.js.get_numaxes()
        # print("numAxes", numAxes)
        # jsInputs = [ float(self.js.get_axis(i)) for i in range(numAxes)]
        # jsButtons = [ float(self.js.get_button(i)) for i in range(self.js.get_numbuttons())]

        # Extract control variables from joystick
        driver = HumanDriver()
        control = driver.computeControl()

        # Print measurements every second.
        timediff = self._timer.elapsed_seconds_since_lap()
        if (timediff > 0.2) and (self._h5_image is not None) and (driver._savemode):
            print("Adding frame {}/{} to h5 {}".format(self._countFrames+1, self._sizeBatch, self._globalCount))
            # Construct image array
            self._image_array[self._countFrames,:,:,:] = image_converter.to_rgb_array(self._h5_image)
            # Construct measurement array
            self._meas_array[self._countFrames,0] = control.steer
            self._meas_array[self._countFrames,1] = control.throttle
            self._meas_array[self._countFrames,2] = control.brake
            self._meas_array[self._countFrames,3] = control.hand_brake
            self._meas_array[self._countFrames,4] = control.reverse
            self._meas_array[self._countFrames,5] = control.steer
            self._meas_array[self._countFrames,6] = control.throttle
            self._meas_array[self._countFrames,7] = control.brake
            self._meas_array[self._countFrames,8] = measurements.player_measurements.transform.location.x
            self._meas_array[self._countFrames,9] = measurements.player_measurements.transform.location.y
            self._meas_array[self._countFrames,10] = measurements.player_measurements.forward_speed
            self._meas_array[self._countFrames,11] = measurements.player_measurements.collision_other
            self._meas_array[self._countFrames,12] = measurements.player_measurements.collision_pedestrians
            self._meas_array[self._countFrames,13] = measurements.player_measurements.collision_vehicles
            self._meas_array[self._countFrames,14] = measurements.player_measurements.intersection_otherlane
            self._meas_array[self._countFrames,15] = measurements.player_measurements.intersection_offroad
            self._meas_array[self._countFrames,16] = measurements.player_measurements.acceleration.x
            self._meas_array[self._countFrames,17] = measurements.player_measurements.acceleration.y
            self._meas_array[self._countFrames,18] = measurements.player_measurements.acceleration.z
            self._meas_array[self._countFrames,19] = measurements.platform_timestamp
            self._meas_array[self._countFrames,20] = measurements.game_timestamp
            self._meas_array[self._countFrames,21] = measurements.player_measurements.transform.orientation.x
            self._meas_array[self._countFrames,22] = measurements.player_measurements.transform.orientation.y
            self._meas_array[self._countFrames,23] = measurements.player_measurements.transform.orientation.z
            self._meas_array[self._countFrames,24] = driver.cmd
            self._meas_array[self._countFrames,25] = 0.0
            self._meas_array[self._countFrames,26] = 0.0
            self._meas_array[self._countFrames,27] = 0.0
            
            self._countFrames += 1
            #print(driver.cmd)

        if timediff > 1.0:
            if self._city_name is not None:
                # Function to get car position on map.
                map_position = self._map.convert_to_pixel([
                    measurements.player_measurements.transform.location.x,
                    measurements.player_measurements.transform.location.y,
                    measurements.player_measurements.transform.location.z])
                # Function to get orientation of the road car is in.
                lane_orientation = self._map.get_lane_orientation([
                    measurements.player_measurements.transform.location.x,
                    measurements.player_measurements.transform.location.y,
                    measurements.player_measurements.transform.location.z])

                self._print_player_measurements_map(
                    measurements.player_measurements,
                    map_position,
                    lane_orientation)
            else:
                self._print_player_measurements(measurements.player_measurements)

            # Plot position on the map as well.

            self._timer.lap()
        
        #self._countFrames += 1

    # Set the player position
        if self._city_name is not None:
            self._position = self._map.convert_to_pixel([
            measurements.player_measurements.transform.location.x,
            measurements.player_measurements.transform.location.y,
            measurements.player_measurements.transform.location.z])
            self._agent_positions = measurements.non_player_agents

        if control is None:
            self._on_new_episode()
        elif self._enable_autopilot:
            self.client.send_control(measurements.player_measurements.autopilot_control)
        else:
            self.client.send_control(control)

    def _print_player_measurements_map(
            self,
            player_measurements,
            map_position,
            lane_orientation):
        message = 'Step {step} ({fps:.1f} FPS): '
        message += 'Map Position ({map_x:.1f},{map_y:.1f}) '
        message += 'Lane Orientation ({ori_x:.1f},{ori_y:.1f}) '
        message += '{speed:.2f} km/h, '
        message += '{other_lane:.0f}% other lane, {offroad:.0f}% off-road'
        message = message.format(
            map_x=map_position[0],
            map_y=map_position[1],
            ori_x=lane_orientation[0],
            ori_y=lane_orientation[1],
            step=self._timer.step,
            fps=self._timer.ticks_per_second(),
            speed=player_measurements.forward_speed * 3.6,
            other_lane=100 * player_measurements.intersection_otherlane,
            offroad=100 * player_measurements.intersection_offroad)
        print_over_same_line(message)

    def _print_player_measurements(self, player_measurements):
        message = 'Step {step} ({fps:.1f} FPS): '
        message += '{speed:.2f} km/h, '
        message += '{other_lane:.0f}% other lane, {offroad:.0f}% off-road'
        message = message.format(
            step=self._timer.step,
            fps=self._timer.ticks_per_second(),
            speed=player_measurements.forward_speed * 3.6,
            other_lane=100 * player_measurements.intersection_otherlane,
            offroad=100 * player_measurements.intersection_offroad)
        print_over_same_line(message)

    def _on_render(self):
        gap_x = (WINDOW_WIDTH - 2 * MINI_WINDOW_WIDTH) / 3
        mini_image_y = WINDOW_HEIGHT - MINI_WINDOW_HEIGHT - gap_x

        if self._main_image is not None:
            array = image_converter.to_rgb_array(self._main_image)
            surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
            self._display.blit(surface, (0, 0))

        if self._mini_view_image1 is not None:
            array = image_converter.depth_to_logarithmic_grayscale(self._mini_view_image1)
            surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
            self._display.blit(surface, (gap_x, mini_image_y))

        if self._mini_view_image2 is not None:
            array = image_converter.labels_to_cityscapes_palette(
                self._mini_view_image2)
            surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))

            self._display.blit(
                surface, (2 * gap_x + MINI_WINDOW_WIDTH, mini_image_y))

        if self._lidar_measurement is not None:
            lidar_data = np.array(self._lidar_measurement.data[:, :2])
            lidar_data *= 2.0
            lidar_data += 100.0
            lidar_data = np.fabs(lidar_data)
            lidar_data = lidar_data.astype(np.int32)
            lidar_data = np.reshape(lidar_data, (-1, 2))
            #draw lidar
            lidar_img_size = (200, 200, 3)
            lidar_img = np.zeros(lidar_img_size)
            lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
            surface = pygame.surfarray.make_surface(lidar_img)
            self._display.blit(surface, (10, 10))

        if self._map_view is not None:
            array = self._map_view
            array = array[:, :, :3]

            new_window_width = \
                (float(WINDOW_HEIGHT) / float(self._map_shape[0])) * \
                float(self._map_shape[1])
            surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))

            w_pos = int(self._position[0]*(float(WINDOW_HEIGHT)/float(self._map_shape[0])))
            h_pos = int(self._position[1] *(new_window_width/float(self._map_shape[1])))

            pygame.draw.circle(surface, [255, 0, 0, 255], (w_pos, h_pos), 6, 0)
            for agent in self._agent_positions:
                if agent.HasField('vehicle'):
                    agent_position = self._map.convert_to_pixel([
                        agent.vehicle.transform.location.x,
                        agent.vehicle.transform.location.y,
                        agent.vehicle.transform.location.z])

                    w_pos = int(agent_position[0]*(float(WINDOW_HEIGHT)/float(self._map_shape[0])))
                    h_pos = int(agent_position[1] *(new_window_width/float(self._map_shape[1])))

                    pygame.draw.circle(surface, [255, 0, 255, 255], (w_pos, h_pos), 4, 0)

            self._display.blit(surface, (WINDOW_WIDTH, 0))

        pygame.display.flip()

def main():
    argparser = argparse.ArgumentParser(
        description='CARLA Manual Control Client')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='localhost',
        help='IP of the host server (default: localhost)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-a', '--autopilot',
        action='store_true',
        help='enable autopilot')
    argparser.add_argument(
        '-l', '--lidar',
        action='store_true',
        help='enable Lidar')
    argparser.add_argument(
        '-q', '--quality-level',
        choices=['Low', 'Epic'],
        type=lambda s: s.title(),
        default='Epic',
        help='graphics quality level, a lower level makes the simulation run considerably faster.')
    argparser.add_argument(
        '-m', '--map-name',
        metavar='M',
        default=None,
        help='plot the map of the current city (needs to match active map in '
             'server, options: Town01 or Town02)')

    args = argparser.parse_args()
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)
    logging.info('listening to server %s:%s', args.host, args.port)

    print(__doc__)

    while True:
        try:
            with make_carla_client(args.host, args.port) as client:
                game = CarlaGame(client, args)
                game.execute()
                break
        except TCPConnectionError as error:
            logging.error(error)
            time.sleep(1)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
