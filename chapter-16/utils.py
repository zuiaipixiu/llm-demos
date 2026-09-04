import re
import threading
import time
from datetime import datetime, timedelta

def format_output(result):
    if isinstance(result, dict) and 'output' in result:
        # 分割输出内容为段落
        paragraphs = result['output'].split('\n\n')
        formatted_output = []
        
        for para in paragraphs:
            # 处理标题（以**开头的行）
            if para.startswith('**'):
                formatted_output.append(f"\n\033[1;36m{para}\033[0m")  # 青色加粗
            # 处理列表项
            elif para.strip().startswith('-'):
                items = para.split('\n')
                formatted_items = [f"  • {item.strip('- ')}" for item in items]
                formatted_output.append('\n'.join(formatted_items))
            # 处理数字列表
            elif re.match(r'^\d+\.', para.strip()):
                items = para.split('\n')
                formatted_items = [f"  {item.strip()}" for item in items]
                formatted_output.append('\n'.join(formatted_items))
            else:
                formatted_output.append(para)
        
        return '\n'.join(formatted_output)
    return str(result)

class TaskTimer:
    def __init__(self):
        self.timers = {}
        self.timer_counter = 0

    def create_timer(self, task: str, target_time: datetime) -> int:
        """创建一个定时器，返回定时器ID"""
        timer_id = self.timer_counter
        self.timer_counter += 1
        
        def timer_callback():
            current_time = datetime.now()
            if current_time >= target_time:
                print(f"\n\033[1;33m[提醒] {task}\033[0m")  # 黄色加粗
                print(f"\033[1;33m[时间] {target_time.strftime('%Y-%m-%d %H:%M:%S')}\033[0m")
                print("\033[1;33m" + "="*50 + "\033[0m")
                if timer_id in self.timers:
                    del self.timers[timer_id]
            else:
                # 如果还没到时间，重新调度
                remaining_seconds = (target_time - current_time).total_seconds()
                if remaining_seconds > 0:
                    self.timers[timer_id] = threading.Timer(remaining_seconds, timer_callback)
                    self.timers[timer_id].start()

        # 计算初始延迟
        current_time = datetime.now()
        initial_delay = (target_time - current_time).total_seconds()
        
        if initial_delay > 0:
            self.timers[timer_id] = threading.Timer(initial_delay, timer_callback)
            self.timers[timer_id].start()
            return timer_id
        else:
            print(f"\n\033[1;31m[错误] 目标时间 {target_time} 已经过去\033[0m")
            return -1

    def cancel_timer(self, timer_id: int) -> bool:
        """取消定时器"""
        if timer_id in self.timers:
            self.timers[timer_id].cancel()
            del self.timers[timer_id]
            return True
        return False

    def get_active_timers(self) -> dict:
        """获取所有活动的定时器"""
        return self.timers

# 创建全局定时器实例
task_timer = TaskTimer()

def parse_relative_time(time_str: str) -> datetime:
    """解析时间字符串，支持相对时间和绝对时间格式
    
    支持的格式：
    1. 相对时间：
       - "2分钟后"
       - "1小时后"
       - "3天后"
    2. 绝对时间：
       - "2024-03-20 15:00:00"
       - "15:00:00" (使用当前日期)
    """
    now = datetime.now()
    
    # 移除所有空格
    time_str = time_str.strip()
    
    # 处理相对时间
    if "后" in time_str:
        # 移除所有空格
        time_str = time_str.replace(" ", "")
        
        # 处理"之后"的情况
        if "之后" in time_str:
            time_str = time_str.replace("之后", "")
        
        # 处理分钟
        if "分钟" in time_str:
            minutes = int(time_str.replace("分钟", ""))
            return now + timedelta(minutes=minutes)
        
        # 处理小时
        elif "小时" in time_str:
            hours = int(time_str.replace("小时", ""))
            return now + timedelta(hours=hours)
        
        # 处理天
        elif "天" in time_str:
            days = int(time_str.replace("天", ""))
            return now + timedelta(days=days)
    
    # 处理绝对时间
    try:
        # 尝试解析完整的时间格式 (YYYY-MM-DD HH:MM:SS)
        if len(time_str) > 8:  # 包含日期
            return datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
        else:  # 只有时间
            # 使用当前日期
            time_obj = datetime.strptime(time_str, "%H:%M:%S").time()
            return datetime.combine(now.date(), time_obj)
    except ValueError as e:
        raise ValueError(f"无法解析时间字符串: {time_str}, 错误: {str(e)}")