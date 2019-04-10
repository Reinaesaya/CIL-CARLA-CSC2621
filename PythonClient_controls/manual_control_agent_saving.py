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
import os
import h5py
import copy
script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in

try:
    import pygame
    from pygame.locals import K_DOWN
    from pygame.locals import K_LEFT
    from pygame.locals import K_RIGHT
    from pygame.locals import K_SPACE
    from pygame.locals import K_UP
    from pygame.locals import K_a
    from pygame.locals import K_d
    from pygame.locals import K_p
    from pygame.locals import K_q
    from pygame.locals import K_r
    from pygame.locals import K_s
    from pygame.locals import K_w
    from pygame.locals import K_n
    from pygame.locals import K_i
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

from carla import image_converter
from carla import sensor
from carla.client import make_carla_client, VehicleControl
from carla.planner.map import CarlaMap
from carla.settings import CarlaSettings
from carla.tcp import TCPConnectionError
from carla.util import print_over_same_line

from agents.imitation.imitation_learning_new import ImitationLearning

WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
MINI_WINDOW_WIDTH = 320
MINI_WINDOW_HEIGHT = 180
H5_WINDOW_WIDTH = 200
H5_WINDOW_HEIGHT = 88

def make_carla_settings(args):
    """Make a CarlaSettings object with the settings we need."""
    settings = CarlaSettings()
    settings.set(
        SynchronousMode=True,
        SendNonPlayerAgentsInfo=True,
        NumberOfVehicles=10,
        NumberOfPedestrians=100,
        WeatherId=random.choice([1, 3, 7, 8, 14]),
        QualityLevel=args.quality_level)
    settings.randomize_seeds()
    # camera0 = sensor.Camera('CameraRGB_main')
    # camera0.set_image_size(WINDOW_WIDTH, WINDOW_HEIGHT)
    # camera0.set_position(2.0, 0.0, 1.4)
    # camera0.set_rotation(0.0, 0.0, 0.0)
    # settings.add_sensor(camera0)
    # camera1 = sensor.Camera('CameraDepth', PostProcessing='Depth')
    # camera1.set_image_size(MINI_WINDOW_WIDTH, MINI_WINDOW_HEIGHT)
    # camera1.set_position(2.0, 0.0, 1.4)
    # camera1.set_rotation(0.0, 0.0, 0.0)
    # settings.add_sensor(camera1)
    # camera2 = sensor.Camera('CameraSemSeg', PostProcessing='SemanticSegmentation')
    # camera2.set_image_size(MINI_WINDOW_WIDTH, MINI_WINDOW_HEIGHT)
    # camera2.set_position(2.0, 0.0, 1.4)
    # camera2.set_rotation(0.0, 0.0, 0.0)
    # settings.add_sensor(camera2)
    cameraNN = sensor.Camera('CameraRGB')
    cameraNN.set_image_size(H5_WINDOW_WIDTH, H5_WINDOW_HEIGHT)
    cameraNN.set_position(2.0, 0.0, 1.4)
    cameraNN.set_rotation(0.0, 0.0, 0.0)
    settings.add_sensor(cameraNN)
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
    def __init__(self, carla_client, nn_agent, saveid, args):
        self.client = carla_client
        self.NNagent = nn_agent
        self._carla_settings = make_carla_settings(args)
        self._timer = None
        self._display = None
        self._main_image = None
        self._mini_view_image1 = None
        self._mini_view_image2 = None
        self._enable_autopilot = args.autopilot
        self._NNagent_drives = False
        self._lidar_measurement = None
        self._map_view = None
        self._is_on_reverse = False
        self._city_name = args.map_name
        self._map = CarlaMap(self._city_name, 0.1643, 50.0) if self._city_name is not None else None
        self._map_shape = self._map.map_image.shape if self._city_name is not None else None
        self._map_view = self._map.get_map(WINDOW_HEIGHT) if self._city_name is not None else None
        self._position = None
        self._agent_positions = None
        self._at_intesection = False
        self.brake_count = 0
        self.saveid = saveid
        self._savecount = 0

        # ADDITION: Limit H5 files to 200 images each, keep track of image and measurement arrays
        self._countFrames = 0
        self._globalCount = 0
        #self._sizeBatch = 200
        self._numTarget = 28
        self._h5_image = None
        self._save_holder = []
        self._save_holder_stop = []
        self._cmd = 5
        self._drivestate = 0

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
        self._is_on_reverse = False

    def _on_loop(self):
        self._timer.tick()

        measurements, sensor_data = self.client.read_data()

        self._main_image = sensor_data.get('CameraRGB_main', None)
        self._mini_view_image1 = sensor_data.get('CameraDepth', None)
        self._mini_view_image2 = sensor_data.get('CameraSemSeg', None)
        self._lidar_measurement = sensor_data.get('Lidar32', None)
        self._h5_image = sensor_data.get('CameraRGB_h5', None)

        timediff = self._timer.elapsed_seconds_since_lap()

        # Print measurements every second.
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

        control = self._get_keyboard_control(pygame.key.get_pressed(), measurements, sensor_data)
        # Set the player position
        if self._city_name is not None:
            self._position = self._map.convert_to_pixel([
                measurements.player_measurements.transform.location.x,
                measurements.player_measurements.transform.location.y,
                measurements.player_measurements.transform.location.z])
            self._agent_positions = measurements.non_player_agents

        #print(control)
        if control is None:
            self._on_new_episode()
        elif self._enable_autopilot:
            self.client.send_control(measurements.player_measurements.autopilot_control)
        else:
            self.client.send_control(control)

        ### Saving on autopilot ###

        if (self._enable_autopilot) and (timediff > 0.03) and (self._h5_image is not None):

            steer = measurements.player_measurements.autopilot_control.steer 
            throttle = measurements.player_measurements.autopilot_control.steer
            brake = measurements.player_measurements.autopilot_control.brake
            hand_brake = measurements.player_measurements.autopilot_control.hand_brake
            reverse = measurements.player_measurements.autopilot_control.reverse

            s_image = image_converter.to_rgb_array(self._h5_image)
            s_measurements = [
                steer,
                throttle,
                brake,
                hand_brake,
                reverse,
                0.0,
                0.0,
                0.0,
                measurements.player_measurements.transform.location.x,
                measurements.player_measurements.transform.location.y,
                measurements.player_measurements.forward_speed,
                measurements.player_measurements.collision_other,
                measurements.player_measurements.collision_pedestrians,
                measurements.player_measurements.collision_vehicles,
                measurements.player_measurements.intersection_otherlane,
                measurements.player_measurements.intersection_offroad,
                measurements.player_measurements.acceleration.x,
                measurements.player_measurements.acceleration.y,
                measurements.player_measurements.acceleration.z,
                measurements.platform_timestamp,
                measurements.game_timestamp,
                measurements.player_measurements.transform.orientation.x,
                measurements.player_measurements.transform.orientation.y,
                measurements.player_measurements.transform.orientation.z,
                -1,
                0.0,
                0.0,
                0.0,
            ]


            if brake == 1.0:
                self.brake_count += 1
            else:
                self.brake_count = 0
            # Need to handle intersection data separately
            # When stopping at stop light, labels can be random
            # When turning, driving labels need to be populated retrospectively
            # For lane following, just populate in large batch
            if self._at_intersection:
                if self._drivestate == 0:
                    self.saveh5s()
                    #print("Saved command {}".format(2))
                    self._drivestate = 1

                self._save_holder.append([copy.deepcopy(s_image), copy.deepcopy(s_measurements)])
                self._save_holder_stop.append([copy.deepcopy(s_image), copy.deepcopy(s_measurements)])

                if self.brake_count > 3:
                    #print("Stopped at light")
                    self._save_holder[-1][1][24] = -2
                    self._save_holder_stop[-1][1][24] = -2
                elif steer < -0.01:
                    #print("Turning left")
                    self._cmd = 3
                elif steer > 0.01:
                    #print("Turning right")
                    self._cmd = 4
                else:
                    #print("Going straight")
                    pass
            else:
                if self._drivestate == 1:
                    for i in range(len(self._save_holder)):
                        if self._save_holder[i][1][24] == -1:
                            self._save_holder[i][1][24] = self._cmd
                            self._save_holder_stop[i][1][24] = self._cmd
                        else:
                            self._save_holder[i][1][24] = self._cmd
                            self._save_holder_stop[i][1][24] = 6

                    self.saveh5s()
                    #print("Saved command {}".format(self._cmd))
                    self._cmd = 5
                    self._drivestate = 0

                self._save_holder.append([copy.deepcopy(s_image), copy.deepcopy(s_measurements)])
                self._save_holder_stop.append([copy.deepcopy(s_image), copy.deepcopy(s_measurements)])

                #print("Following lane")
                self._save_holder[-1][1][24] = 2
                if self.brake_count > 3:
                    self._save_holder_stop[-1][1][24] = 6
                else:
                    self._save_holder_stop[-1][1][24] = 2

    def saveh5s(self):
        assert len(self._save_holder) == len(self._save_holder_stop)
        dpcount = len(self._save_holder)
        if dpcount == 0:
            print("Nothing to save")
            return

        rel_path = "h5/Town2/nostop/dp{}_{}_{}{}_{}.h5".format(self.saveid, self._savecount, self._drivestate, self._cmd, dpcount)
        abs_file_path = os.path.join(script_dir, rel_path)
        rel_path_s = "h5/Town2/stop/dp{}s_{}_{}{}_{}.h5".format(self.saveid, self._savecount, self._drivestate, self._cmd, dpcount)
        abs_file_path_s = os.path.join(script_dir, rel_path_s)

        img_array = np.zeros((dpcount,88,200,3))
        meas_array = np.zeros((dpcount,self._numTarget))
        img_array_s = np.zeros((dpcount,88,200,3))
        meas_array_s = np.zeros((dpcount,self._numTarget))

        for i in range(dpcount):
            img_array[i,:,:,:] = self._save_holder[i][0]
            for j in range(28):
                meas_array[i,j] = self._save_holder[i][1][j]
            img_array_s[i,:,:,:] = self._save_holder_stop[i][0]
            for j in range(28):
                meas_array_s[i,j] = self._save_holder_stop[i][1][j]
        with h5py.File(abs_file_path, 'w') as f:
            dset_image = f.create_dataset("rgb", data=img_array, dtype='u1')
            dset_meas = f.create_dataset("targets", data=meas_array, dtype='f4')
        with h5py.File(abs_file_path_s, 'w') as f:
            dset_image_s = f.create_dataset("rgb", data=img_array_s, dtype='u1')
            dset_meas_s = f.create_dataset("targets", data=meas_array_s, dtype='f4')

        if self._drivestate == 0:
            print("Saved {} lane following data points to {}".format(dpcount, rel_path))
        if self._drivestate == 1:
            print("Saved {} intersection data points to {}".format(dpcount, rel_path))

        self._save_holder = []
        self._save_holder_stop = []
        self._savecount += 1


    def _get_keyboard_control(self, keys, measurements, sensor_data):
        """
        Return a VehicleControl message based on the pressed keys. Return None
        if a new episode was requested.
        """
        self._at_intersection = False
        if keys[K_r]:
            return None

        if self._NNagent_drives:
            if keys[K_LEFT] or keys[K_a]:
                control = self.NNagent.run_step(measurements, sensor_data, 3, None)
            elif keys[K_RIGHT] or keys[K_d]:
                control = self.NNagent.run_step(measurements, sensor_data, 4, None)
            elif keys[K_UP] or keys[K_w]:
                control = self.NNagent.run_step(measurements, sensor_data, 2, None)
            else:
                control = self.NNagent.run_step(measurements, sensor_data, -1, None)

            if keys[K_q]:
                self._is_on_reverse = not self._is_on_reverse
            if keys[K_p]:
                self._enable_autopilot = True
                self._NNagent_drives = not self._NNagent_drives
            if keys[K_n]:
                print("Turning off")
                self._NNagent_drives = not self._NNagent_drives
            if keys[K_i]:
                self._at_intersection = True
            control.reverse = self._is_on_reverse
        else:
            control = VehicleControl()
            if keys[K_LEFT] or keys[K_a]:
                control.steer = -1.0
            if keys[K_RIGHT] or keys[K_d]:
                control.steer = 1.0
            if keys[K_UP] or keys[K_w]:
                control.throttle = 1.0
            if keys[K_DOWN] or keys[K_s]:
                control.brake = 1.0
            if keys[K_SPACE]:
                control.hand_brake = True
            if keys[K_q]:
                self._is_on_reverse = not self._is_on_reverse
            if keys[K_p]:
                self._enable_autopilot = not self._enable_autopilot
            if keys[K_n]:
                print("Turning on")
                self._NNagent_drives = not self._NNagent_drives
                self._enable_autopilot = False
            if keys[K_i]:
                self._at_intersection = True
            control.reverse = self._is_on_reverse
        return control

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

        # if self._mini_view_image2 is not None:
        #     array = image_converter.labels_to_cityscapes_palette(
        #         self._mini_view_image2)
        #     surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))

        #     self._display.blit(
        #         surface, (2 * gap_x + MINI_WINDOW_WIDTH, mini_image_y))

        if self._h5_image is not None:
            array = image_converter.to_rgb_array(self._h5_image)
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
    argparser.add_argument(
        '--avoid-stopping',
        default=True,
        action='store_false',
        help=' Uses the speed prediction branch to avoid unwanted NN agent stops'
    )
    argparser.add_argument(
        '--saveid',
        required=True,
        help=' Save ID'
    )
    args = argparser.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    print(__doc__)

    while True:
        try:

            with make_carla_client(args.host, args.port) as client:
                client.load_settings(CarlaSettings())
                client.start_episode(0)

                NNagent = ImitationLearning(args.map_name, args.avoid_stopping)
                game = CarlaGame(client, NNagent, args.saveid, args)
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
