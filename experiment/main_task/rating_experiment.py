import json
from pathlib import Path
from datetime import datetime
import tkinter as tk
from tkinter import ttk, messagebox

import pandas as pd
from psychopy import visual, core, event

import config

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = SCRIPT_DIR / 'rating_data'
FULLSCREEN = True
DEBUG_WINDOW_SIZE = (1440, 900)
TRIALS_PER_SU = 10
RANDOM_SEED = None
ITI_SECONDS = 0.3
RESPONSE_KEYS = ['f', 'j']
RESPONSE_MAP = {'f': 0, 'j': 1}
RESPONSE_LABELS = {0: '不存在', 1: '存在'}


class RatingSubjectDialog:
    def __init__(self):
        self.info = None

    def show(self):
        root = tk.Tk()
        root.title('独立判断被试信息')
        root.geometry('560x320')
        root.resizable(False, False)
        root.update_idletasks()
        x = (root.winfo_screenwidth() // 2) - 280
        y = (root.winfo_screenheight() // 2) - 160
        root.geometry(f'560x320+{x}+{y}')

        fields = {}

        tk.Label(root, text='被试ID:', font=('Arial', 12)).grid(row=0, column=0, padx=20, pady=18, sticky='e')
        fields['id'] = tk.Entry(root, font=('Arial', 12), width=24)
        fields['id'].grid(row=0, column=1, padx=20, pady=18)

        tk.Label(root, text='性别:', font=('Arial', 12)).grid(row=1, column=0, padx=20, pady=18, sticky='e')
        fields['gender'] = ttk.Combobox(root, font=('Arial', 12), width=21, state='readonly')
        fields['gender']['values'] = ('男', '女', '其他')
        fields['gender'].current(0)
        fields['gender'].grid(row=1, column=1, padx=20, pady=18)

        tk.Label(root, text='年龄:', font=('Arial', 12)).grid(row=2, column=0, padx=20, pady=18, sticky='e')
        fields['age'] = tk.Entry(root, font=('Arial', 12), width=24)
        fields['age'].grid(row=2, column=1, padx=20, pady=18)

        def on_confirm():
            subject_id = fields['id'].get().strip()
            if not subject_id:
                messagebox.showerror('错误', '请输入被试ID')
                return

            age_text = fields['age'].get().strip()
            try:
                age = int(age_text)
                if age < 0 or age > 120:
                    raise ValueError
            except ValueError:
                messagebox.showerror('错误', '请输入有效年龄（0-120）')
                return

            self.info = {
                'SubjectID': subject_id,
                'Gender': fields['gender'].get(),
                'Age': age,
            }
            root.destroy()

        tk.Button(root, text='开始任务', font=('Arial', 12), width=12, command=on_confirm).grid(
            row=3, column=0, columnspan=2, pady=28
        )

        root.mainloop()
        return self.info


class CongruentRatingExperiment:
    def __init__(self, subject_info, stimuli_file=None):
        self.subject_info = subject_info
        self.subject_id = subject_info['SubjectID']
        self.stimuli_file = self._resolve_stimuli_file(stimuli_file)
        self.stimuli_df = self._load_stimuli(self.stimuli_file)
        self.output_dir = OUTPUT_ROOT / f"sub-{self.subject_id}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.trial_data = []
        self.win = None
        self.actual_width = None
        self.actual_height = None
        self.coord_scale_x = None
        self.coord_scale_y = None
        self.instruction_text = None
        self.prompt_text = None
        self.scale_text = None
        self.progress_text = None
        self.ball = None
        self.catcher_left_parts = []
        self.catcher_right_parts = []

    def _resolve_stimuli_file(self, stimuli_file):
        if stimuli_file is not None:
            path = Path(stimuli_file)
            if not path.is_absolute():
                path = SCRIPT_DIR / path
            if not path.exists():
                raise FileNotFoundError(f'找不到刺激文件: {path}')
            return path

        candidates = [
            SCRIPT_DIR / 'stimuli' / 'stimulus_config.csv',
            SCRIPT_DIR / 'stimulus_config.csv',
        ]
        for path in candidates:
            if path.exists():
                return path
        raise FileNotFoundError('找不到 stimulus_config.csv，请先确认刺激文件存在。')

    def _load_stimuli(self, stimuli_path):
        df = pd.read_csv(stimuli_path)
        df['HV_primary'] = df['HV_primary'].astype(str).str.strip().str.lower()
        df['SU_level'] = pd.to_numeric(df['SU_level'], errors='coerce').astype('Int64')
        df = df[(df['HV_primary'] == 'congruent') & (df['SU_level'].between(1, config.SU_LEVELS))].copy()
        if df.empty:
            raise ValueError('刺激文件中没有可用的 congruent 条件试次。')

        selected_parts = []
        for su_level in range(1, config.SU_LEVELS + 1):
            su_df = df[df['SU_level'] == su_level].copy()
            if su_df.empty:
                raise ValueError(f'congruent 条件下缺少 SU_level={su_level} 的刺激。')
            if TRIALS_PER_SU is not None:
                if len(su_df) < TRIALS_PER_SU:
                    raise ValueError(
                        f'SU_level={su_level} 只有 {len(su_df)} 个 congruent 刺激，不足 {TRIALS_PER_SU} 个。'
                    )
                su_df = su_df.sample(n=TRIALS_PER_SU, replace=False, random_state=RANDOM_SEED)
            selected_parts.append(su_df)

        trial_df = pd.concat(selected_parts, ignore_index=True)
        trial_df = trial_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
        trial_df['RatingTrialID'] = range(1, len(trial_df) + 1)
        return trial_df

    def setup_psychopy(self):
        window_kwargs = dict(
            screen=0,
            allowGUI=False,
            monitor='testMonitor',
            color=[0, 0, 0],
            colorSpace='rgb',
            units='pix',
        )

        if FULLSCREEN:
            self.win = visual.Window(fullscr=True, **window_kwargs)
        else:
            self.win = visual.Window(size=DEBUG_WINDOW_SIZE, fullscr=False, **window_kwargs)

        self.actual_width, self.actual_height = self.win.size
        config.update_screen_params(int(self.actual_width), int(self.actual_height))
        self.coord_scale_x = self.actual_width / config.DESIGN_WIDTH
        self.coord_scale_y = self.actual_height / config.DESIGN_HEIGHT

        text_height = int(28 * min(self.coord_scale_x, self.coord_scale_y))
        small_height = int(22 * min(self.coord_scale_x, self.coord_scale_y))

        self.instruction_text = visual.TextStim(
            self.win,
            text='',
            height=text_height,
            color='white',
            wrapWidth=int(1250 * self.coord_scale_x),
        )
        self.prompt_text = visual.TextStim(
            self.win,
            text='请判断：从小球起始位置到某一接球器是否存在直线轨迹\nF=否（不存在）    J=是（存在）',
            pos=(0, -self.actual_height / 2 + 100),
            height=small_height,
            color='white',
            wrapWidth=int(1400 * self.coord_scale_x),
        )
        self.scale_text = visual.TextStim(
            self.win,
            text='按键盘 F 表示否，J 表示是    ESC 退出',
            pos=(0, -self.actual_height / 2 + 55),
            height=int(20 * min(self.coord_scale_x, self.coord_scale_y)),
            color='white',
        )
        self.progress_text = visual.TextStim(
            self.win,
            text='',
            pos=(self.actual_width / 2 - 120, self.actual_height / 2 - 40),
            height=int(20 * min(self.coord_scale_x, self.coord_scale_y)),
            color='white',
        )
        self.ball = visual.Circle(
            self.win,
            radius=config.BALL_RADIUS,
            fillColor='red',
            lineColor='red',
        )

        left_center_x = config.CATCHER_LEFT_X - self.actual_width / 2
        right_center_x = config.CATCHER_RIGHT_X - self.actual_width / 2
        catcher_y = self.actual_height / 2 - config.CATCHER_Y
        wall_width = max(8, int(10 * min(self.coord_scale_x, self.coord_scale_y)))
        wall_height = max(50, int(80 * min(self.coord_scale_x, self.coord_scale_y)))

        catcher_left = visual.Rect(
            self.win,
            width=config.CATCHER_WIDTH,
            height=config.CATCHER_HEIGHT,
            pos=(left_center_x, catcher_y),
            fillColor='green',
            lineColor='green',
        )
        catcher_right = visual.Rect(
            self.win,
            width=config.CATCHER_WIDTH,
            height=config.CATCHER_HEIGHT,
            pos=(right_center_x, catcher_y),
            fillColor='blue',
            lineColor='blue',
        )

        self.catcher_left_parts = [
            catcher_left,
            visual.Rect(
                self.win,
                width=wall_width,
                height=wall_height,
                pos=(left_center_x - config.CATCHER_WIDTH / 2 + wall_width / 2, catcher_y + wall_height / 2),
                fillColor='green',
                lineColor='green',
            ),
            visual.Rect(
                self.win,
                width=wall_width,
                height=wall_height,
                pos=(left_center_x + config.CATCHER_WIDTH / 2 - wall_width / 2, catcher_y + wall_height / 2),
                fillColor='green',
                lineColor='green',
            ),
        ]
        self.catcher_right_parts = [
            catcher_right,
            visual.Rect(
                self.win,
                width=wall_width,
                height=wall_height,
                pos=(right_center_x - config.CATCHER_WIDTH / 2 + wall_width / 2, catcher_y + wall_height / 2),
                fillColor='blue',
                lineColor='blue',
            ),
            visual.Rect(
                self.win,
                width=wall_width,
                height=wall_height,
                pos=(right_center_x + config.CATCHER_WIDTH / 2 - wall_width / 2, catcher_y + wall_height / 2),
                fillColor='blue',
                lineColor='blue',
            ),
        ]

    def draw_planks(self, planks_value):
        if isinstance(planks_value, str):
            planks = json.loads(planks_value)
        else:
            planks = planks_value

        plank_stims = []
        for plank in planks:
            psychopy_x = float(plank['x']) * self.coord_scale_x - self.actual_width / 2
            psychopy_y = self.actual_height / 2 - float(plank['y']) * self.coord_scale_y
            plank_stims.append(
                visual.Rect(
                    self.win,
                    width=config.PLANK_WIDTH,
                    height=config.PLANK_HEIGHT,
                    pos=(psychopy_x, psychopy_y),
                    ori=-float(plank['angle']),
                    fillColor='white',
                    lineColor='white',
                )
            )
        return plank_stims

    def draw_trial_scene(self, trial_info):
        plank_stims = self.draw_planks(trial_info['Planks'])
        for stim in plank_stims:
            stim.draw()
        for stim in self.catcher_left_parts:
            stim.draw()
        for stim in self.catcher_right_parts:
            stim.draw()

        ball_start_x = float(trial_info.get('Ball_start_x', config.DESIGN_WIDTH / 2))
        ball_start_y = float(trial_info.get('Ball_start_y', 100))
        self.ball.pos = (
            ball_start_x * self.coord_scale_x - self.actual_width / 2,
            self.actual_height / 2 - ball_start_y * self.coord_scale_y,
        )
        self.ball.draw()

    def show_instructions(self):
        n_trials = len(self.stimuli_df)
        instructions = (
            f'欢迎参加独立判断任务\n\n'
            f'您将看到 {n_trials} 张静态板面图片。\n'
            f'请只根据图片本身，判断从小球起始位置到某一接球器是否存在一条直线轨迹，任务只收集您对材料本身的感知，不存在对错之分\n\n'
            f'按键规则：\n'
            f'F = 否（不存在）\n'
            f'J = 是（存在）\n\n'
            f'请使用键盘 F 或 J 进行判断。\n'
            f'准备好后按空格开始。'
        )
        self.instruction_text.text = instructions
        self.instruction_text.draw()
        self.win.flip()
        event.waitKeys(keyList=['space', 'escape'])

    def run_trial(self, trial_info):
        event.clearEvents()
        self.progress_text.text = f"{int(trial_info['RatingTrialID'])}/{len(self.stimuli_df)}"
        self.draw_trial_scene(trial_info)
        self.prompt_text.draw()
        self.scale_text.draw()
        self.progress_text.draw()
        self.win.flip()

        clock = core.Clock()
        keys = event.waitKeys(keyList=RESPONSE_KEYS + ['escape'], timeStamped=clock)
        key, rt = keys[0]
        if key == 'escape':
            self.quit_experiment()
        response_binary = RESPONSE_MAP[key]

        record = {
            'SubjectID': self.subject_info['SubjectID'],
            'Gender': self.subject_info['Gender'],
            'Age': self.subject_info['Age'],
            'SessionTime': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'StimuliFile': str(self.stimuli_file),
            'RatingTrialID': int(trial_info['RatingTrialID']),
            'BoardID': trial_info['BoardID'],
            'HV_primary': trial_info['HV_primary'],
            'SU_level': int(trial_info['SU_level']),
            'SU_raw': trial_info.get('SU_raw', None),
            'PhysOutcome': trial_info.get('PhysOutcome', ''),
            'Ball_start_side': trial_info.get('Ball_start_side', ''),
            'Ball_start_x': trial_info.get('Ball_start_x', None),
            'ResponseKey': key,
            'StraightTrajectoryExists': response_binary,
            'JudgmentLabel': RESPONSE_LABELS[response_binary],
            'RT_ms': round(float(rt) * 1000, 3),
        }
        self.trial_data.append(record)
        self.save_data()
        core.wait(ITI_SECONDS)

    def save_data(self):
        if not self.trial_data:
            return
        output_file = self.output_dir / 'ratings.csv'
        pd.DataFrame(self.trial_data).to_csv(output_file, index=False)

    def run(self):
        try:
            self.setup_psychopy()
            self.show_instructions()
            for _, trial_info in self.stimuli_df.iterrows():
                self.run_trial(trial_info)
            self.instruction_text.text = '判断完成！\n\n感谢您的参与！'
            self.instruction_text.draw()
            self.win.flip()
            core.wait(2.5)
        finally:
            self.save_data()
            self.cleanup()

    def cleanup(self):
        if self.win is not None:
            self.win.close()

    def quit_experiment(self):
        self.save_data()
        self.cleanup()
        core.quit()


def main():
    dialog = RatingSubjectDialog()
    subject_info = dialog.show()
    if subject_info is None:
        print('用户取消，程序结束。')
        return

    experiment = CongruentRatingExperiment(subject_info)
    experiment.run()


if __name__ == '__main__':
    main()
