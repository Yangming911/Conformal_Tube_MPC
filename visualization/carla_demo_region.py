#!/usr/bin/env python

"""
CARLA Episode Demo with Proper Pygame Integration
基于manual_control.py的架构，集成CBF控制器和行人预测
"""

import carla
from carla import ColorConverter as cc
import argparse
import collections
import datetime
import logging
import math
import random
import re
import weakref
import numpy as np
import sys
import os

import pygame
from pygame.locals import KMOD_CTRL
from pygame.locals import KMOD_SHIFT
from pygame.locals import K_0
from pygame.locals import K_9
from pygame.locals import K_BACKQUOTE
from pygame.locals import K_BACKSPACE
from pygame.locals import K_COMMA
from pygame.locals import K_DOWN
from pygame.locals import K_ESCAPE
from pygame.locals import K_F1
from pygame.locals import K_LEFT
from pygame.locals import K_PERIOD
from pygame.locals import K_RIGHT
from pygame.locals import K_SLASH
from pygame.locals import K_SPACE
from pygame.locals import K_TAB
from pygame.locals import K_UP
from pygame.locals import K_a
from pygame.locals import K_b
from pygame.locals import K_c
from pygame.locals import K_d
from pygame.locals import K_f
from pygame.locals import K_g
from pygame.locals import K_h
from pygame.locals import K_i
from pygame.locals import K_l
from pygame.locals import K_m
from pygame.locals import K_n
from pygame.locals import K_o
from pygame.locals import K_p
from pygame.locals import K_q
from pygame.locals import K_r
from pygame.locals import K_s
from pygame.locals import K_t
from pygame.locals import K_v
from pygame.locals import K_w
from pygame.locals import K_x
from pygame.locals import K_z
from pygame.locals import K_MINUS
from pygame.locals import K_EQUALS


# 导入我们的模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cbf.current_cbf_controller import cbf_controller_multi_pedestrian
from models.predictor import WalkerActionPredictor
from envs.dynamics_social_force import get_vehicle_contour_and_influence_point, walker_logic_SF
from utils.constants import dt,num_pedestrians,WALKER_DESTINATION_Y
import utils.constants as C


def get_actor_display_name(actor, truncate=250):
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name


# ==============================================================================
# -- World ---------------------------------------------------------------------
# ==============================================================================

class World(object):
    def __init__(self, carla_world, hud, args):
        self.world = carla_world
        self.sync = args.sync
        self.hud = hud
        self.player = None
        self.walkers = []
        self.camera_manager = None
        self._gamma = args.gamma
        self.predictor = None
        self.step_count = 0
        self.max_steps = args.steps
        self.controller_state_transform_x = -50
        self.controller_state_transform_y = -60
        
        # 行人控制相关
        self.walker_velocities = []  # 存储每个行人的速度
        self.walker_destination_y = 80.0  # 行人目的地Y坐标 (CARLA坐标)
        
        # 控制模式相关
        self.control_mode = "CBF"  # "CBF" 或 "MANUAL" 或 "STRAIGHT" 或 "LEFT_TURN"
        self.manual_control = {
            'throttle': 0.0,
            'brake': 0.0,
            'steer': 0.0
        }
        
        # 左转控制器相关
        self.turn_controller = {
            'enabled': False,
            'start_x': 0.0,
            'start_y': 0.0,
            'start_yaw': 0.0,
            'turn_radius': 15.0,  # 转弯半径
            'turn_speed': 8.0,    # 转弯时的速度
            'turn_progress': 0.0,  # 转弯进度 (0-1)
            'target_yaw': 0.0,    # 目标航向角
            'turn_completed': False
        }
        
        # 视频录制相关
        self.recording = False
        self.recorded_frames = []
        self.video_filename = f"simulation_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        self.auto_recording = args.record  # 自动录制开关
        self.show_debug = True  # debug绘制开关
        
        # 初始化预测器（固定选择已提供的v2_fixed权重）
        self.predictor = WalkerActionPredictor(model_path=os.path.join('assets', 'walker_speed_predictor_v2_fixed.pth'))
        
        # 生成车辆和行人
        self.spawn_actors()
        
        # 初始化行人控制，确保所有行人都有初始速度
        for i, walker in enumerate(self.walkers):
            if i < len(self.walker_velocities):
                vx, vy = self.walker_velocities[i]
                walker_control = carla.WalkerControl()
                speed = math.sqrt(vx**2 + vy**2)
                if speed > 0.01:
                    direction_x = vx / speed
                    direction_y = vy / speed
                    walker_control.direction = carla.Vector3D(x=direction_x, y=direction_y, z=0.0)
                    walker_control.speed = speed
                    walker.apply_control(walker_control)
                    print(f"行人{i}初始控制应用: 速度={speed:.2f}, 方向=({direction_x:.2f}, {direction_y:.2f})")
        
        # 设置摄像头
        self.camera_manager = CameraManager(self.player, self.hud, self._gamma)
        self.camera_manager.set_sensor(0, notify=False)  # RGB摄像头
        
        # 自动开始录制
        if self.auto_recording:
            self.start_recording()
        else:
            print("视频录制已禁用 (--record=False)")
        
        self.world.on_tick(hud.on_world_tick)

    def spawn_actors(self):
        """生成车辆和行人"""
        blueprint_library = self.world.get_blueprint_library()

        # 车辆 - 使用指定蓝图
        # veh_bp = blueprint_library.find('vehicle.dodge.charger_police_2020')
        veh_bp = blueprint_library.find('vehicle.tesla.model3')
        # 将(50,70)投影到道路
        base_z = self.world.get_map().get_spawn_points()[0].location.z
        desired_loc = carla.Location(x=60.0, y=70.0, z=base_z)
        wp = self.world.get_map().get_waypoint(desired_loc, project_to_road=True, lane_type=carla.LaneType.Driving)
        if wp is None:
            spawn_tf = self.world.get_map().get_spawn_points()[0]
        else:
            spawn_tf = wp.transform
            spawn_tf.location.z = base_z
        
        # 尝试生成车辆
        vehicle = None
        forward_vec = spawn_tf.get_forward_vector()
        for k in range(10):
            try_tf = carla.Transform(
                carla.Location(
                    x=spawn_tf.location.x + forward_vec.x * 1.0 * k,
                    y=spawn_tf.location.y + forward_vec.y * 1.0 * k,
                    z=spawn_tf.location.z + 0.1
                ),
                spawn_tf.rotation
            )
            vehicle = self.world.try_spawn_actor(veh_bp, try_tf)
            if vehicle is not None:
                break
        
        if vehicle is None:
            raise RuntimeError("Vehicle spawn failed")
        
        self.player = vehicle
        print(f"车辆生成成功: {get_actor_display_name(vehicle)}")

        # 行人 - 生成在(90,60)到(90,77)之间，均匀分散，随机外观
        import random
        
        # 获取所有可用的行人蓝图
        walker_blueprints = blueprint_library.filter('walker.pedestrian.*')
        num_walkers = 5
        y_start = 60.0
        y_end = 67.0
        
        for i in range(num_walkers):
            # 随机选择一个行人蓝图
            walker_bp = random.choice(walker_blueprints)
            
            wx = 90.0
            # 在60到77之间均匀分布
            wy = y_start + (y_end - y_start) * i / (num_walkers - 1) if num_walkers > 1 else (y_start + y_end) / 2
            desired = carla.Location(x=wx, y=wy, z=1.1)

            ped_point = carla.Transform(desired)
            walker = self.world.spawn_actor(walker_bp, ped_point)
            # 在导航网格上寻找合适位置
            # candidate_tf = None
            # best_dist = 1e9
            # for _ in range(100):
            #     rnd = self.world.get_random_location_from_navigation()
            #     d = (rnd.x - desired.x)**2 + (rnd.y - desired.y)**2
            #     if d < best_dist:
            #         best_dist = d
            #         candidate_tf = carla.Transform(rnd)
            
            # candidate_tf.location.z = spawn_tf.location.z + 0.5
            # walker = self.world.try_spawn_actor(walker_bp, candidate_tf)
            if walker is not None:
                self.walkers.append(walker)
                self.walker_velocities.append([0.0, 0.0])  # 初始化行人速度为0
                print(f"行人{i}生成成功")

    def to_controller_state(self):
        """将CARLA状态转换为控制器状态"""
        car_tf = self.player.get_transform()
        car_vel = self.player.get_velocity()
        
        state = {
            'car_x': car_tf.location.x + self.controller_state_transform_x,
            'car_y': car_tf.location.y + self.controller_state_transform_y,
            'car_vx': car_vel.x,
            'car_vy': car_vel.y,
            'car_v': math.sqrt(car_vel.x**2 + car_vel.y**2),
            'car_theta': math.radians(car_tf.rotation.yaw),
            'walker_x': [],
            'walker_y': [],
            'walker_vx': [],
            'walker_vy': []
        }
        
        for walker in self.walkers:
            walker_tf = walker.get_transform()
            walker_vel = walker.get_velocity()
            state['walker_x'].append(walker_tf.location.x + self.controller_state_transform_x)
            state['walker_y'].append(walker_tf.location.y + self.controller_state_transform_y)
            state['walker_vx'].append(walker_vel.x)
            state['walker_vy'].append(walker_vel.y)
        
        return state

    def apply_control(self, u):
        """应用CBF控制"""
        # 将控制量u转换为CARLA控制
        current_vel = self.player.get_velocity()
        current_speed = math.sqrt(current_vel.x**2 + current_vel.y**2)
        target_speed = u
        
        error = target_speed - current_speed
        throttle = float(np.clip(error * 0.3, 0.0, 1.0))
        brake = 0.0 if error > 0 else float(np.clip(-error * 0.3, 0.0, 1.0))
        
        ctrl = carla.VehicleControl(throttle=throttle, brake=brake)
        self.player.apply_control(ctrl)
    
    def apply_manual_control(self):
        """应用手动控制"""
        ctrl = carla.VehicleControl(
            throttle=self.manual_control['throttle'],
            brake=self.manual_control['brake'],
            steer=self.manual_control['steer']
        )
        self.player.apply_control(ctrl)
    
    def apply_straight_control(self):
        """应用直行控制 - 以4m/s速度直行"""
        current_speed = math.sqrt(self.player.get_velocity().x**2 + self.player.get_velocity().y**2)
        target_speed = 4.0  # 目标速度4m/s（原来8m/s的一半）
        
        # 速度控制 - 降低加速度
        if current_speed < target_speed * 0.9:
            throttle = 0.3  # 原来0.6的一半
            brake = 0.0
        elif current_speed > target_speed * 1.1:
            throttle = 0.0
            brake = 0.15  # 原来0.3的一半
        else:
            throttle = 0.2  # 原来0.4的一半
            brake = 0.0
        
        # 直行控制 - 不转向
        steer = 0.0
        
        # 应用控制
        ctrl = carla.VehicleControl(
            throttle=throttle,
            brake=brake,
            steer=steer
        )
        self.player.apply_control(ctrl)
    
    def handle_keyboard_input(self, keys):
        """处理键盘输入"""
        if self.control_mode == "MANUAL":
            # 油门控制
            if keys[K_w] or keys[K_UP]:
                self.manual_control['throttle'] = min(1.0, self.manual_control['throttle'] + 0.1)
            else:
                self.manual_control['throttle'] = max(0.0, self.manual_control['throttle'] - 0.1)
            
            # 刹车控制
            if keys[K_s] or keys[K_DOWN]:
                self.manual_control['brake'] = min(1.0, self.manual_control['brake'] + 0.1)
            else:
                self.manual_control['brake'] = max(0.0, self.manual_control['brake'] - 0.1)
            
            # 转向控制
            if keys[K_a] or keys[K_LEFT]:
                self.manual_control['steer'] = max(-1.0, self.manual_control['steer'] - 0.1)
            elif keys[K_d] or keys[K_RIGHT]:
                self.manual_control['steer'] = min(1.0, self.manual_control['steer'] + 0.1)
            else:
                # 自动回正
                if self.manual_control['steer'] > 0:
                    self.manual_control['steer'] = max(0.0, self.manual_control['steer'] - 0.1)
                elif self.manual_control['steer'] < 0:
                    self.manual_control['steer'] = min(0.0, self.manual_control['steer'] + 0.1)
    
    def start_left_turn(self, car_x, car_y, car_yaw):
        """启动左转控制器"""
        self.turn_controller['enabled'] = True
        self.turn_controller['start_x'] = car_x
        self.turn_controller['start_y'] = car_y
        self.turn_controller['start_yaw'] = car_yaw
        self.turn_controller['turn_progress'] = 0.0
        
        # 计算目标航向角 - 确保是左转90度
        # 在CARLA中，Y轴正方向对应航向角0度，左转90度意味着航向角增加90度
        target_yaw = car_yaw + math.pi / 2
        # 标准化角度到 [0, 2π]
        while target_yaw > 2 * math.pi:
            target_yaw -= 2 * math.pi
        while target_yaw < 0:
            target_yaw += 2 * math.pi
        
        # 调试：打印角度信息
        print(f"角度计算: 当前航向={math.degrees(car_yaw):.1f}°, 目标航向={math.degrees(target_yaw):.1f}°, 差值={math.degrees(target_yaw - car_yaw):.1f}°")
            
        self.turn_controller['target_yaw'] = target_yaw
        self.turn_controller['turn_completed'] = False
        
        print(f"启动左转控制器: 起始位置({car_x:.2f}, {car_y:.2f})")
        print(f"起始航向: {math.degrees(car_yaw):.1f}°, 目标航向: {math.degrees(target_yaw):.1f}°")
    
    def apply_left_turn_control(self):
        """应用左转控制 - 改进版本，转弯完成后停止修正"""
        if not self.turn_controller['enabled']:
            return
        
        car_tf = self.player.get_transform()
        car_x = car_tf.location.x
        car_y = car_tf.location.y
        car_yaw = math.radians(car_tf.rotation.yaw)
        
        start_yaw = self.turn_controller['start_yaw']
        target_yaw = self.turn_controller['target_yaw']
        
        # 计算已经转过的角度（从开始到现在的角度变化）
        yaw_turned = car_yaw - start_yaw
        # 标准化到 [-π, π]
        while yaw_turned > math.pi:
            yaw_turned -= 2 * math.pi
        while yaw_turned < -math.pi:
            yaw_turned += 2 * math.pi
        
        # 计算转弯进度 (0-1)
        total_turn = math.pi / 2  # 90度转弯
        turn_progress = abs(yaw_turned) / total_turn
        turn_progress = max(0.0, min(1.0, turn_progress))
        self.turn_controller['turn_progress'] = turn_progress
        
        # 检查是否完成转弯 - 基于已转角度而不是角度差
        if abs(yaw_turned) >= total_turn * 0.99:  # 转了90度的95%以上认为完成
            self.turn_controller['turn_completed'] = True
            self.turn_controller['enabled'] = False
            print(f"左转完成! 最终位置: ({car_x:.2f}, {car_y:.2f}), 最终航向: {math.degrees(car_yaw):.1f}°, 已转角度: {math.degrees(yaw_turned):.1f}°")
            return
        
        # 左转控制 - 基于已转角度
        if abs(yaw_turned) < total_turn * 0.9:  # 还没转到90度的90%
            # 继续左转
            if abs(yaw_turned) > total_turn * 0.7:  # 接近完成时减速转向
                steer_intensity = 0.2
            else:
                steer_intensity = 0.3
            steer = -steer_intensity  # 负值表示左转
        else:
            # 接近完成，轻微转向
            steer = -0.1
        
        # 速度控制 - 转弯时减速
        current_speed = math.sqrt(self.player.get_velocity().x**2 + self.player.get_velocity().y**2)
        target_speed = 7.0  # 进一步降低目标速度
        
        if current_speed < target_speed * 0.8:
            throttle = 0.2
            brake = 0.0
        elif current_speed > target_speed * 1.2:
            throttle = 0.0
            brake = 0.3
        else:
            throttle = 0.1
            brake = 0.0
        
        # 应用控制
        ctrl = carla.VehicleControl(
            throttle=throttle,
            brake=brake,
            steer=steer
        )
        self.player.apply_control(ctrl)
        
        # 调试信息
        if self.step_count % 5 == 0:  # 每5步打印一次，减少输出
            print(f"左转中 - 进度: {turn_progress:.2f}, 已转角度: {math.degrees(yaw_turned):.1f}°, 转向: {steer:.2f}, 速度: {current_speed:.1f}m/s, 位置: ({car_x:.1f}, {car_y:.1f})")
    
    def start_recording(self):
        """开始录制视频"""
        self.recording = True
        self.recorded_frames = []
        print(f"开始录制视频: {self.video_filename}")
    
    def stop_recording(self):
        """停止录制并保存视频"""
        if not self.recording:
            return
        
        self.recording = False
        print(f"停止录制，共录制了 {len(self.recorded_frames)} 帧")
        
        # 保存视频
        self.save_video()
    
    def save_video(self):
        """保存录制的视频 - 高帧率高清晰度版本"""
        if not self.recorded_frames:
            print("没有录制的帧，无法保存视频")
            return
        
        try:
            import cv2
            import numpy as np
            
            # 获取第一帧的尺寸
            first_frame = self.recorded_frames[0]
            height, width = first_frame.shape[:2]
            
            # 使用更高质量的编码器和合适的帧率
            fourcc = cv2.VideoWriter_fourcc(*'H264')  # 使用H264编码器，质量更好
            fps = 20.0  # 降低到20FPS，让视频播放更自然
            out = cv2.VideoWriter(self.video_filename, fourcc, fps, (width, height))
            
            # 写入所有帧
            for frame in self.recorded_frames:
                # 转换颜色格式 (RGB -> BGR)
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)
            
            out.release()
            print(f"高质量视频已保存: {self.video_filename} (20FPS, H264编码, 1080p)")
            
        except ImportError:
            print("错误: 需要安装 opencv-python 库来保存视频")
            print("请运行: pip install opencv-python")
            print("录制的帧数据已丢失，请安装库后重新录制")
        except Exception as e:
            print(f"保存视频时出错: {e}")
            print("录制的帧数据已丢失")
    
    def capture_frame(self, display):
        """捕获当前帧"""
        if self.recording:
            # 获取显示表面的像素数据
            frame_array = pygame.surfarray.array3d(display)
            # 转换维度 (width, height, channels) -> (height, width, channels)
            frame_array = frame_array.swapaxes(0, 1)
            self.recorded_frames.append(frame_array)

    def draw_debug_info(self):
        """绘制调试信息"""
        if not self.show_debug:
            return
            
        debug = self.world.debug
        
        # 绘制行人预测位置的五边形
        state = self.to_controller_state()
        if len(state['walker_x']) > 0:
            # 调试信息：每10步打印一次
            if self.step_count % 10 == 0:
                print(f"绘制五边形: 行人数量={len(state['walker_x'])}")
            # 逐个行人预测位置并绘制五边形
            for i, walker in enumerate(self.walkers):
                walker_tf = walker.get_transform()
                walker_x = walker_tf.location.x
                walker_y = walker_tf.location.y
                walker_z = walker_tf.location.z
                
                # 使用神经网络预测每个行人的下一步速度
                next_walker_vx, next_walker_vy = self.predictor.predict(
                    state['car_x'], state['car_y'], state['car_v'],
                    state['walker_x'][i], state['walker_y'][i],
                    state['walker_vx'][i], state['walker_vy'][i]
                )
                
                # 预测下一步位置（使用控制器状态坐标）
                dt = float(C.dt)
                # next_walker_x_controller = state['walker_x'][i] + next_walker_vx * dt
                # next_walker_y_controller = state['walker_y'][i] + next_walker_vy * dt
                next_walker_x_controller = state['walker_x'][i] + next_walker_vx 
                next_walker_y_controller = state['walker_y'][i] + next_walker_vy 
                
                # 将控制器坐标转换回CARLA坐标
                next_walker_x_carla = next_walker_x_controller - self.controller_state_transform_x
                next_walker_y_carla = next_walker_y_controller - self.controller_state_transform_y
                
                # 定义五边形的五个顶点（使用CARLA坐标，贴地显示）
                ground_z = 0.0  # 贴地显示，避免浮空
                
                # 顶点1: 当前行人位置
                v1 = carla.Location(x=walker_x, y=walker_y, z=ground_z)
                
                # 顶点2: (next_walker_x_carla-0.1446, next_walker_y_carla-0.7179)
                v2 = carla.Location(x=next_walker_x_carla-0.1446, y=next_walker_y_carla-0.7179, z=ground_z)
                
                
                # 顶点3: (next_walker_x_carla+0.1446, next_walker_y_carla-0.7179)
                v3 = carla.Location(x=next_walker_x_carla+0.1446, y=next_walker_y_carla-0.7179, z=ground_z)
                
                # 顶点4: (next_walker_x_carla+0.1446, next_walker_y_carla+0.7179)
                v4 = carla.Location(x=next_walker_x_carla+0.1446, y=next_walker_y_carla+0.7179, z=ground_z)
                
                # 顶点5: (next_walker_x_carla-0.1446, next_walker_y_carla+0.7179)
                v5 = carla.Location(x=next_walker_x_carla-0.1446, y=next_walker_y_carla+0.7179, z=ground_z)
                
                # 绘制五边形边
                vertices = [v1, v2, v3, v4, v5]
                for j in range(len(vertices)):
                    start_vertex = vertices[j]
                    end_vertex = vertices[(j + 1) % len(vertices)]
                    debug.draw_line(start_vertex, end_vertex, thickness=0.05, color=carla.Color(20, 20, 0), life_time=0.15)
                
                # 调试信息：打印第一个行人的五边形顶点
                if i == 0 and self.step_count % 10 == 0:
                    print(f"行人0五边形顶点: V1=({walker_x:.1f},{walker_y:.1f}), V2=({v2.x:.1f},{v2.y:.1f}), V3=({v3.x:.1f},{v3.y:.1f})")

    def update_walkers(self):
        """使用Social Force模型更新行人位置和速度"""
        car_tf = self.player.get_transform()
        car_vel = self.player.get_velocity()
        car_speed = math.sqrt(car_vel.x**2 + car_vel.y**2)
        
        for i, walker in enumerate(self.walkers):
            walker_tf = walker.get_transform()
            walker_x = walker_tf.location.x
            walker_y = walker_tf.location.y
            
            # 获取当前行人速度
            current_vx, current_vy = self.walker_velocities[i]
            
            # 使用Social Force模型计算新的速度
            # 注意：这里使用CARLA坐标，目的地是(90, 80)
            new_vx, new_vy = walker_logic_SF(
                car_v=car_speed,
                car_x_position=car_tf.location.x,
                car_y_position=car_tf.location.y,
                walker_x_position=walker_x,
                walker_y_position=walker_y,
                walker_v_x_past=current_vx,
                walker_v_y_past=current_vy,
                destination_y=self.walker_destination_y  # 目的地Y=80
            )
            
            # 更新存储的速度
            self.walker_velocities[i] = [new_vx, new_vy]
            
            # 应用速度到CARLA行人
            walker_control = carla.WalkerControl()
            walker_control.direction = carla.Vector3D(x=new_vx, y=new_vy, z=0.0)
            walker_control.speed = math.sqrt(new_vx**2 + new_vy**2)
            walker.apply_control(walker_control)

    def tick(self, clock):
        """每帧更新"""
        self.hud.tick(self, clock)
        
        # 更新行人位置和速度
        self.update_walkers()
        
        # 检查控制模式切换条件
        car_tf = self.player.get_transform()
        car_x = car_tf.location.x
        car_y = car_tf.location.y
        car_yaw = math.radians(car_tf.rotation.yaw)
        
        # 控制模式切换逻辑
        if car_x > 80.0 and self.control_mode == "CBF":
            self.control_mode = "STRAIGHT"
            print(f"控制模式切换到直行控制 (X坐标: {car_x:.2f})")
        elif car_x > 100.0 and self.control_mode == "STRAIGHT":
            self.control_mode = "LEFT_TURN"
            self.start_left_turn(car_x, car_y, car_yaw)
            print(f"控制模式切换到左转控制 (X坐标: {car_x:.2f})")
        
        # 根据控制模式应用相应的控制
        if self.control_mode == "CBF":
            # 应用CBF控制
            state = self.to_controller_state()
            u = cbf_controller_multi_pedestrian(state, self.predictor, cp_alpha=0.75, gamma=0.1, d_safe=1.0)
            self.apply_control(u)
        elif self.control_mode == "STRAIGHT":
            # 应用直行控制
            self.apply_straight_control()
        elif self.control_mode == "LEFT_TURN":
            # 应用左转控制
            self.apply_left_turn_control()
        else:
            # 应用手动控制
            self.apply_manual_control()
        
        # 绘制调试信息
        self.draw_debug_info()
        
        # 更新步数
        self.step_count += 1
        if self.step_count >= self.max_steps:
            print(f"仿真完成! 共运行了 {self.step_count} 步")
            return False
        
        return True

    def render(self, display):
        """渲染"""
        self.camera_manager.render(display)
        self.hud.render(display)
        
        # 捕获帧用于录制
        self.capture_frame(display)

    def destroy(self):
        """清理资源"""
        # 如果正在录制，停止录制并保存视频
        if self.recording:
            self.stop_recording()
        
        if self.camera_manager is not None:
            self.camera_manager.sensor.destroy()
        if self.player is not None:
            self.player.destroy()
        for walker in self.walkers:
            walker.destroy()


# ==============================================================================
# -- HUD -----------------------------------------------------------------------
# ==============================================================================

class HUD(object):
    def __init__(self, width, height):
        self.dim = (width, height)
        font = pygame.font.Font(pygame.font.get_default_font(), 20)
        font_name = 'courier' if os.name == 'nt' else 'mono'
        fonts = [x for x in pygame.font.get_fonts() if font_name in x]
        default_font = 'ubuntumono'
        mono = default_font if default_font in fonts else fonts[0]
        mono = pygame.font.match_font(mono)
        self._font_mono = pygame.font.Font(mono, 12 if os.name == 'nt' else 14)
        self._notifications = FadingText(font, (width, 40), (0, height - 40))
        self.server_fps = 0
        self.frame = 0
        self.simulation_time = 0
        self._show_info = True
        self._info_text = []
        self._server_clock = pygame.time.Clock()

    def on_world_tick(self, timestamp):
        self._server_clock.tick()
        self.server_fps = self._server_clock.get_fps()
        self.frame = timestamp.frame
        self.simulation_time = timestamp.elapsed_seconds

    def tick(self, world, clock):
        self._notifications.tick(world, clock)
        if not self._show_info:
            return
        
        if world.player is not None:
            t = world.player.get_transform()
            v = world.player.get_velocity()
            speed_kmh = 3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)
            
            self._info_text = [
                'Server:  % 16.0f FPS' % self.server_fps,
                'Client:  % 16.0f FPS' % clock.get_fps(),
                '',
                'Vehicle: % 20s' % get_actor_display_name(world.player, truncate=20),
                'Map:     % 20s' % world.world.get_map().name.split('/')[-1],
                'Simulation time: % 12s' % datetime.timedelta(seconds=int(self.simulation_time)),
                '',
                'Speed:   % 15.0f km/h' % speed_kmh,
                'Location:% 20s' % ('(% 5.1f, % 5.1f)' % (t.location.x, t.location.y)),
                'Height:  % 18.0f m' % t.location.z,
                '',
                f'Step: {world.step_count}/{world.max_steps}',
                f'Walkers: {len(world.walkers)}',
                # f'Control Mode: {world.control_mode}',
                ''
            ]
            
            # 添加行人信息
            if len(world.walkers) > 0:
                self._info_text.append('Pedestrians:')
                for i, walker in enumerate(world.walkers):
                    walker_tf = walker.get_transform()
                    walker_vel = walker.get_velocity()
                    walker_speed = math.sqrt(walker_vel.x**2 + walker_vel.y**2)
                    self._info_text.extend([
                        f'  Ped {i}:',
                        f'    Pos: ({walker_tf.location.x:5.1f}, {walker_tf.location.y:5.1f}, {walker_tf.location.z:5.1f})',
                        f'    Vel: ({walker_vel.x:5.1f}, {walker_vel.y:5.1f})',
                        f'    Speed: {walker_speed:5.1f} m/s',
                        ''
                    ])
                

    def notification(self, text, seconds=2.0):
        self._notifications.set_text(text, seconds=seconds)

    def render(self, display):
        if self._show_info:
            info_surface = pygame.Surface((220, self.dim[1]))
            info_surface.set_alpha(100)
            display.blit(info_surface, (0, 0))
            v_offset = 4
            for item in self._info_text:
                if v_offset + 18 > self.dim[1]:
                    break
                if item:  # At this point has to be a str.
                    surface = self._font_mono.render(item, True, (255, 255, 255))
                    display.blit(surface, (8, v_offset))
                v_offset += 18
        self._notifications.render(display)


# ==============================================================================
# -- FadingText ----------------------------------------------------------------
# ==============================================================================

class FadingText(object):
    def __init__(self, font, dim, pos):
        self.font = font
        self.dim = dim
        self.pos = pos
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)

    def set_text(self, text, color=(255, 255, 255), seconds=2.0):
        text_texture = self.font.render(text, True, color)
        self.surface = pygame.Surface(self.dim)
        self.seconds_left = seconds
        self.surface.fill((0, 0, 0, 0))
        self.surface.blit(text_texture, (10, 11))

    def tick(self, _, clock):
        delta_seconds = 1e-3 * clock.get_time()
        self.seconds_left = max(0.0, self.seconds_left - delta_seconds)
        self.surface.set_alpha(500.0 * self.seconds_left)

    def render(self, display):
        display.blit(self.surface, self.pos)


# ==============================================================================
# -- CameraManager -------------------------------------------------------------
# ==============================================================================

class CameraManager(object):
    def __init__(self, parent_actor, hud, gamma_correction):
        self.sensor = None
        self.surface = None
        self._parent = parent_actor
        self.hud = hud
        self.recording = False
        bound_x = 0.5 + self._parent.bounding_box.extent.x
        bound_y = 0.5 + self._parent.bounding_box.extent.y
        bound_z = 0.5 + self._parent.bounding_box.extent.z
        Attachment = carla.AttachmentType

        # self._camera_transforms = [
        #     (carla.Transform(carla.Location(x=-2.0*bound_x, y=+0.0*bound_y, z=2.0*bound_z), carla.Rotation(pitch=8.0)), Attachment.SpringArmGhost),
        #     (carla.Transform(carla.Location(x=+0.8*bound_x, y=+0.0*bound_y, z=1.3*bound_z)), Attachment.Rigid),
        #     (carla.Transform(carla.Location(x=+1.9*bound_x, y=+1.0*bound_y, z=1.2*bound_z)), Attachment.SpringArmGhost),
        #     (carla.Transform(carla.Location(x=-2.8*bound_x, y=+0.0*bound_y, z=4.6*bound_z), carla.Rotation(pitch=6.0)), Attachment.SpringArmGhost),
        #     (carla.Transform(carla.Location(x=-1.0, y=-1.0*bound_y, z=0.4*bound_z)), Attachment.Rigid)
        # ]
        self._camera_transforms = [
            # (carla.Transform(carla.Location(x=-2.5, z=0.0), carla.Rotation(pitch=-8.0)), Attachment.SpringArmGhost),
            (carla.Transform(carla.Location(x=1.6, z=1.7)), Attachment.Rigid),
                        # 新增俯视相机 - 既向下看又向前看，观察汽车和前方道路
            (carla.Transform(carla.Location(x=-1.0, y=0.0, z=20.0), carla.Rotation(pitch=30.0, yaw=0.0)), Attachment.SpringArmGhost),
            # (carla.Transform(carla.Location(x=2.5, y=0.5, z=0.0), carla.Rotation(pitch=-8.0)), Attachment.SpringArmGhost),
            (carla.Transform(carla.Location(x=-4.0, z=2.0), carla.Rotation(pitch=6.0)), Attachment.SpringArmGhost),
            (carla.Transform(carla.Location(x=0, y=-2.5, z=-0.0), carla.Rotation(yaw=90.0)), Attachment.Rigid)]


        self.transform_index = 0
        self.sensors = [
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB', {}],
            ['sensor.camera.depth', cc.Raw, 'Camera Depth (Raw)', {}],
            ['sensor.camera.depth', cc.Depth, 'Camera Depth (Gray Scale)', {}],
            ['sensor.camera.depth', cc.LogarithmicDepth, 'Camera Depth (Logarithmic Gray Scale)', {}],
            ['sensor.camera.semantic_segmentation', cc.Raw, 'Camera Semantic Segmentation (Raw)', {}],
            ['sensor.camera.semantic_segmentation', cc.CityScapesPalette, 'Camera Semantic Segmentation (CityScapes Palette)', {}],
        ]

        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()
        for item in self.sensors:
            bp = bp_library.find(item[0])
            if item[0].startswith('sensor.camera'):
                bp.set_attribute('image_size_x', str(hud.dim[0]))
                bp.set_attribute('image_size_y', str(hud.dim[1]))
                if bp.has_attribute('gamma'):
                    bp.set_attribute('gamma', str(gamma_correction))
                for attr_name, attr_value in item[3].items():
                    bp.set_attribute(attr_name, attr_value)
            item.append(bp)
        self.index = None

    def set_sensor(self, index, notify=True, force_respawn=False):
        index = index % len(self.sensors)
        needs_respawn = True if self.index is None else \
            (force_respawn or (self.sensors[index][2] != self.sensors[self.index][2]))
        if needs_respawn:
            if self.sensor is not None:
                self.sensor.destroy()
                self.surface = None
            self.sensor = self._parent.get_world().spawn_actor(
                self.sensors[index][-1],
                self._camera_transforms[self.transform_index][0],
                attach_to=self._parent,
                attachment_type=self._camera_transforms[self.transform_index][1])
            # We need to pass the lambda a weak reference to self to avoid
            # circular reference.
            weak_self = weakref.ref(self)
            self.sensor.listen(lambda image: CameraManager._parse_image(weak_self, image))
        if notify:
            self.hud.notification(self.sensors[index][2])
        self.index = index

    def render(self, display):
        if self.surface is not None:
            display.blit(self.surface, (0, 0))

    @staticmethod
    def _parse_image(weak_self, image):
        self = weak_self()
        if not self:
            return
        image.convert(self.sensors[self.index][1])
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))


# ==============================================================================
# -- game_loop() ---------------------------------------------------------------
# ==============================================================================

def game_loop(args):
    pygame.init()
    pygame.font.init()
    sim_world = None
    world = None

    client = carla.Client(args.host, args.port)
    client.set_timeout(2000.0)
    sim_world = client.get_world()

    # 保存并切换到同步设置
    original_settings = sim_world.get_settings()
    settings = sim_world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = dt
    sim_world.apply_settings(settings)

    traffic_manager = client.get_trafficmanager()
    traffic_manager.set_synchronous_mode(True)

    display = pygame.display.set_mode(
        (args.width, args.height),
        pygame.HWSURFACE | pygame.DOUBLEBUF)
    display.fill((0,0,0))
    pygame.display.flip()

    hud = HUD(args.width, args.height)
    world = World(sim_world, hud, args)

    sim_world.tick()

    clock = pygame.time.Clock()
    running = True
    while running:
        sim_world.tick()
        clock.tick_busy_loop(60)  # 保持60FPS，但视频输出20FPS
        
        # 获取当前按键状态
        keys = pygame.key.get_pressed()
        
        # 处理键盘输入
        world.handle_keyboard_input(keys)
        
        # 处理pygame事件
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYUP:
                if event.key == K_ESCAPE:
                    running = False
                elif event.key == K_F1:
                    hud._show_info = not hud._show_info
                elif event.key == K_TAB:
                    world.camera_manager.transform_index = (world.camera_manager.transform_index + 1) % len(world.camera_manager._camera_transforms)
                    world.camera_manager.set_sensor(world.camera_manager.index, notify=False, force_respawn=True)
                elif event.key == K_r:
                    # R键开始/停止录制
                    if world.recording:
                        world.stop_recording()
                    else:
                        world.start_recording()

        # 更新世界
        if not world.tick(clock):
            running = False

        world.render(display)
        pygame.display.flip()

    # 恢复设置并清理
    if sim_world is not None and original_settings is not None:
        sim_world.apply_settings(original_settings)
    if world is not None:
        world.destroy()
    pygame.quit()


# ==============================================================================
# -- main() --------------------------------------------------------------------
# ==============================================================================

def main():
    argparser = argparse.ArgumentParser(description='CARLA Episode Demo with CBF Controller')
    argparser.add_argument('--host', metavar='H', default='127.0.0.1', help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument('-p', '--port', metavar='P', default=2000, type=int, help='TCP port to listen to (default: 2000)')
    argparser.add_argument('--res', metavar='WIDTHxHEIGHT', default='1920x1080', help='window resolution (default: 1920x1080)')
    argparser.add_argument('--gamma', default=2.2, type=float, help='Gamma correction of the camera (default: 2.2)')
    argparser.add_argument('--sync', action='store_true', help='Activate synchronous mode execution')
    argparser.add_argument('--steps', default=400, type=int, help='Maximum simulation steps (default: 600)')
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
    
    argparser.add_argument('--record', type=str2bool, default=True, help='Whether to record and save MP4 video (default: True)')
    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split('x')]

    log_level = logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    try:
        game_loop(args)
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':
    main()
