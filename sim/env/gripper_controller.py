"""
Binary Gripper Controller
이진 분류 방식의 그리퍼 제어 - Open(1) 또는 Close(0)만 사용
0/1 명령을 받아 역정규화된 실제 그리퍼 값으로 보간
"""

import numpy as np


class GripperController:
    """
    그리퍼를 Open(1) 또는 Close(0) 두 상태로 제어
    명령 변화(0→1 or 1→0)를 감지하고 0~1 범위에서 선형 보간
    
    Parameters:
        open_duration (int): Open 명령(0→1)이 완료되는 스텝 수 (기본: 50)
        close_duration (int): Close 명령(1→0)이 완료되는 스텝 수 (기본: 50)
    """
    
    def __init__(self, open_duration=50, close_duration=50):
        self.open_duration = open_duration
        self.close_duration = close_duration
        
        # 좌팔 상태
        self.left_state = {
            'is_transitioning': False,
            'transition_type': None,  # 'OPENING' or 'CLOSING'
            'transition_start_step': 0,
            'current_value': 1.0,  # 초기: 열려있음 (1.0)
            'target_value': 1.0
        }
        
        # 우팔 상태
        self.right_state = {
            'is_transitioning': False,
            'transition_type': None,
            'transition_start_step': 0,
            'current_value': 1.0,  # 초기: 열려있음 (1.0)
            'target_value': 1.0
        }
    
    def _detect_command_change(self, previous_command, current_command):
        """
        명령 변화 감지 (0↔1)
        
        Returns:
            str or None: 'OPENING', 'CLOSING', or None
        """
        # 임계값으로 이진화
        prev_binary = 1 if previous_command > 0.5 else 0
        curr_binary = 1 if current_command > 0.5 else 0
        
        if prev_binary == curr_binary:
            # 명령 변화 없음
            return None
        
        if prev_binary == 0 and curr_binary == 1:
            # 0 → 1: OPEN
            return 'OPENING'
        
        if prev_binary == 1 and curr_binary == 0:
            # 1 → 0: CLOSE
            return 'CLOSING'
        
        return None
    
    def _get_interpolated_position(self, state_dict, step):
        """
        0~1 범위에서 선형 보간
        
        Args:
            state_dict (dict): 좌팔 또는 우팔 상태
            step (int): 현재 스텝
        
        Returns:
            float: 보간된 값 (0.0 ~ 1.0)
        """
        if not state_dict['is_transitioning']:
            return state_dict['target_value']
        
        # 경과 스텝 계산
        elapsed_steps = step - state_dict['transition_start_step']
        
        if state_dict['transition_type'] == 'OPENING':
            # 0.0 → 1.0으로 이동
            duration = self.open_duration
            progress = min(elapsed_steps / duration, 1.0)
            gripper_value = 0.0 + (1.0 - 0.0) * progress
            
            if progress >= 1.0:
                # 전환 완료
                state_dict['is_transitioning'] = False
                state_dict['current_value'] = 1.0
                state_dict['target_value'] = 1.0
                return 1.0
            
            return gripper_value
        
        elif state_dict['transition_type'] == 'CLOSING':
            # 1.0 → 0.0으로 이동
            duration = self.close_duration
            progress = min(elapsed_steps / duration, 1.0)
            gripper_value = 1.0 - (1.0 - 0.0) * progress
            
            if progress >= 1.0:
                # 전환 완료
                state_dict['is_transitioning'] = False
                state_dict['current_value'] = 0.0
                state_dict['target_value'] = 0.0
                return 0.0
            
            return gripper_value
    
    def process_gripper(self, side, previous_command, current_command, step):
        """
        그리퍼 제어 처리: 0~1 범위에서 보간
        
        Args:
            side (str): 'left' 또는 'right'
            previous_command (float): 이전 스텝의 명령값 (0.0~1.0)
            current_command (float): 현재 스텝의 명령값 (0.0~1.0)
            step (int): 현재 스텝
        
        Returns:
            float: 보간된 그리퍼 값 (0.0~1.0)
        """
        # 상태 선택
        state_dict = self.left_state if side == 'left' else self.right_state
        
        # 명령 변화 감지
        transition_type = self._detect_command_change(previous_command, current_command)
        
        if transition_type:
            # 새로운 전환 시작
            state_dict['is_transitioning'] = True
            state_dict['transition_type'] = transition_type
            state_dict['transition_start_step'] = step
            state_dict['target_value'] = 1.0 if transition_type == 'OPENING' else 0.0
        
        # 위치 계산 및 반환
        return self._get_interpolated_position(state_dict, step)
    
    def get_current_binary_state(self, side):
        """
        현재 이진 상태 반환 (0.0 또는 1.0)
        
        Args:
            side (str): 'left' 또는 'right'
        
        Returns:
            float: 0.0 (닫힌) 또는 1.0 (열린)
        """
        state_dict = self.left_state if side == 'left' else self.right_state
        # target_value가 1.0이면 열린 상태(1.0), 0.0이면 닫힌 상태(0.0)
        return state_dict['target_value']
    
    def reset(self):
        """
        컨트롤러 상태 초기화 (새 에피소드 시작 시)
        """
        self.left_state = {
            'is_transitioning': False,
            'transition_type': None,
            'transition_start_step': 0,
            'current_value': 1.0,  # 초기: 열려있음
            'target_value': 1.0
        }
        
        self.right_state = {
            'is_transitioning': False,
            'transition_type': None,
            'transition_start_step': 0,
            'current_value': 1.0,  # 초기: 열려있음
            'target_value': 1.0
        }
