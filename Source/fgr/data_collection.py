import os

from psychopy import gui, visual, core, data, logging
from psychopy.constants import (NOT_STARTED, STARTED, FINISHED)
from psychopy.hardware import keyboard
from pathlib import Path

from Source.streamer.data import Data


class Experiment:
    """This class is used to run a PsychoPy experiment for data collection."""

    def __init__(self):
        self.data = None

    def trigger(self, msg, verbose: bool = True):

        if self.data is not None:
            self.data.add_annotation(msg)
            if verbose:
                print(f'TRIGGER: {self.data.annotations[-1]}')
        elif verbose:
            print(f'TRIGGER: {msg}')

    # todo: refactor this method - break it into smaller methods to achieve better readability and maintainability
    def run(self,
            data_obj: Data,
            data_dir: str,
            n_repetitions: int = 10,
            img_secs: float = 5,
            fullscreen: bool = True,
            screen_num: int = 0) -> Path:
        """
        Run the experiment.

        Parameters
        ----------
        data_obj : Data
            Data object to send annotations to.
        data_dir : str
            Directory in which to save data.
        n_repetitions : int, optional
            Number of repetitions of each gesture during data collection. The default is 10.
        img_secs : float, optional
            Number of seconds to display each image. The default is 5.
        fullscreen : bool, optional
            Whether to display the experiment in fullscreen. The default is True.
        screen_num : int, optional
            Screen number to display the experiment on. The default is 0.
        """

        """ HARD-CODED EXPERIMENT PARAMETERS """
        img_dir = r'.../images'
        instruction_secs = 4
        relax_secs = 3
        """ END PARAMETERS """

        self.data = data_obj

        img_names = os.listdir(img_dir)

        # Ensure that relative paths start from the same directory as this script
        _thisDir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(_thisDir)
        # Store info about the experiment session
        exp_info = {'participant': '', 'session': '', 'position': ''}
        dlg = gui.DlgFromDict(dictionary=exp_info, sortKeys=False, title='fill in experiment info')
        if not dlg.OK:
            core.quit()  # user pressed cancel
        exp_info['date'] = data.getDateStr()  # add a simple timestamp
        exp_info['expName'] = 'fgr - real time'
        exp_info['psychopyVersion'] = '2022.1.3'

        # make sure the save dir exists and guve the future file a unique name according to the experiment info
        os.makedirs(data_dir, exist_ok=True)
        self.data.save_as = str(Path(data_dir, f"GR_pos{exp_info['position']}_{exp_info['participant'].rjust(3, '0')}_S{exp_info['session']}_BT.edf"))

        # start recording data
        self.data.start()

        # save a log file for detail verbose info
        logging.console.setLevel(logging.WARNING)  # this outputs to the screen, not a file

        frame_tolerance = 0.001  # how close to onset before 'same' frame

        # Start Code - component code to be run after the window creation

        # Set up the Window
        win = visual.Window(
            size=[1920, 1080],
            fullscr=fullscreen,
            screen=screen_num,
            winType='pyglet', allowGUI=False, allowStencil=False,
            monitor='testMonitor', color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb',
            blendMode='avg', useFBO=True,
            units='height')
        # store frame rate of monitor if we can measure it
        exp_info['frameRate'] = win.getActualFrameRate()

        # create a default keyboard (e.g. to check for escape)
        default_keyboard = keyboard.Keyboard(backend='iohub')

        # Initialize components for Routine "Welcome"
        welcome_clock = core.Clock()
        welcome_text = visual.TextStim(
            win=win, name='welcome_text',
            text='A series of images will be shown on screen.\n\n\n'
                 'Perform the gesture only when\n"Perform gesture"\nis written above the image.\n\n\n'
                 'Relax your arm between gestures.\n\n\n'
                 '(Press space when ready.)',
            font='Calibri Light',
            pos=(0, 0), height=0.04, wrapWidth=None, ori=0.0,
            color=[1.0000, 1.0000, 1.0000], colorSpace='rgb', opacity=None,
            languageStyle='RTL'
        )
        key_resp = keyboard.Keyboard()

        # Initialize components for Routine "TwoFingersInst"
        img_text = visual.TextStim(
            win=win, name='img_text',
            text="PLACEHOLDER",
            font='Calibri',
            pos=(0, 0.3), height=0.04, wrapWidth=None, ori=0.0,
            color='white', colorSpace='rgb', opacity=None,
            languageStyle='LTR'
        )

        # Initialize image routines
        imgs = {}
        clocks = {}
        for n, img_name in enumerate(img_names):
            clock = core.Clock()
            img = visual.ImageStim(
                win=win,
                name=img_name,
                image=os.path.join(img_dir, img_name),
                mask=None,
                anchor='center', ori=0.0, size=0.5, pos=(0, 0),
                color=(1, 1, 1), colorSpace='rgb', opacity=None,
                flipHoriz=False, flipVert=False,
                texRes=128.0, interpolate=True, depth=0.0
            )
            imgs[img_name] = img
            clocks[img_name] = clock

        # Create some handy timers
        routine_timer = core.CountdownTimer()  # to track time remaining of each (non-slip) routine

        # ------Prepare to start Routine "Welcome"-------
        continue_routine = True
        # update component parameters for each repeat
        # keep track of which components have finished
        key_resp.keys = []
        key_resp.rt = []
        _key_resp_allKeys = []
        welcome_components = [welcome_text, key_resp]
        for thisComponent in welcome_components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        welcome_clock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
        n_frame = -1

        # -------Run Routine "Welcome"-------
        while continue_routine:
            # get current time
            t = welcome_clock.getTime()
            t_flip = win.getFutureFlipTime(clock=welcome_clock)
            t_flip_global = win.getFutureFlipTime(clock=None)
            n_frame = n_frame + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame

            # *instructions_text* updates
            if welcome_text.status == NOT_STARTED and t_flip >= -frame_tolerance:
                # keep track of start time/frame for later
                welcome_text.frameNStart = n_frame  # exact frame index
                welcome_text.tStart = t  # local t and not account for scr refresh
                welcome_text.tStartRefresh = t_flip_global  # on global time
                win.timeOnFlip(welcome_text, 'tStartRefresh')  # time at next scr refresh
                welcome_text.setAutoDraw(True)

            # *key_resp* updates
            wait_on_flip = False
            if key_resp.status == NOT_STARTED and t_flip >= 0.0-frame_tolerance:
                # keep track of start time/frame for later
                key_resp.frameNStart = n_frame  # exact frame index
                key_resp.tStart = t  # local t and not account for scr refresh
                key_resp.tStartRefresh = t_flip_global  # on global time
                win.timeOnFlip(key_resp, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                key_resp.status = STARTED
                # keyboard checking is just starting
                wait_on_flip = True
                win.callOnFlip(key_resp.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_resp.status == STARTED and not wait_on_flip:
                these_keys = key_resp.getKeys(keyList=['space'], waitRelease=False)
                _key_resp_allKeys.extend(these_keys)
                if len(_key_resp_allKeys):
                    key_resp.keys = _key_resp_allKeys[-1].name  # just the last key pressed
                    key_resp.rt = _key_resp_allKeys[-1].rt
                    # a response ends the routine
                    continue_routine = False

            # check for quit (typically the Esc key)
            if default_keyboard.getKeys(keyList=["escape"]):
                core.quit()

            # check if all components have finished
            if not continue_routine:  # a component has requested a forced-end of Routine
                break
            continue_routine = False  # will revert to True if at least one component still running
            for thisComponent in welcome_components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continue_routine = True
                    break  # at least one component has not yet finished

            # refresh the screen
            if continue_routine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()

        # -------Ending Routine "Welcome"-------
        for thisComponent in welcome_components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # the Routine "Welcome" was not non-slip safe, so reset the non-slip timer
        routine_timer.reset()

        for img_name in img_names:
            trigger_name = os.path.splitext(img_name)[0]
            trials = data.TrialHandler(nReps=n_repetitions,
                                       method='random',
                                       extraInfo=exp_info,
                                       originPath=-1,
                                       trialList=[None],
                                       seed=None,
                                       name=trigger_name)
            for n_rep, trial in enumerate(trials):

                instruction_secs_rep = instruction_secs if n_rep == 0 else 0

                if trial is not None:
                    for paramName in trial:
                        exec('{} = trial[paramName]'.format(paramName))

                # ------Prepare to start Routine "ThreeFingers"-------
                continue_routine = True
                # routine_timer.addTime(5)
                # update component parameters for each repeat
                imgs[img_name].setImage(os.path.join(img_dir, img_name))

                # keep track of which components have finished
                components = [imgs[img_name], img_text]
                for component in components:
                    component.tStart = None
                    component.tStop = None
                    component.tStartRefresh = None
                    component.tStopRefresh = None
                    if hasattr(component, 'status'):
                        component.status = NOT_STARTED
                # reset timers
                _timeToFirstFrame = win.getFutureFlipTime(clock="now")
                clocks[img_name].reset(-_timeToFirstFrame)  # t0 is time of first possible flip\
                n_frame = -1

                # -------Run Routine-------
                while continue_routine:
                    # get current time
                    t = clocks[img_name].getTime()
                    t_flip = win.getFutureFlipTime(clock=clocks[img_name])
                    t_flip_global = win.getFutureFlipTime(clock=None)
                    # update/draw components on each frame

                    # image updates
                    if imgs[img_name].status == NOT_STARTED:
                        # keep track of start time/frame for later
                        imgs[img_name].tStart = t  # local t and not account for scr refresh
                        imgs[img_name].tStartRefresh = t_flip_global  # on global time
                        win.timeOnFlip(imgs[img_name], 'tStartRefresh')  # time at next scr refresh
                        imgs[img_name].setAutoDraw(True)
                    if imgs[img_name].status == STARTED:
                        if t_flip_global > imgs[img_name].tStartRefresh + instruction_secs_rep + img_secs - frame_tolerance:
                            # keep track of stop time/frame for later
                            imgs[img_name].tStop = t  # not accounting for scr refresh
                            # win.timeOnFlip(imgs[img_name], 'tStopRefresh')  # time at next scr refresh
                            imgs[img_name].setAutoDraw(False)

                    # check for quit (typically the Esc key)
                    if default_keyboard.getKeys(keyList=["escape"]):
                        core.quit()

                    # text updates must come AFTER image updates in order to be placed on top
                    if img_text.status == NOT_STARTED and t_flip >= -frame_tolerance:
                        # keep track of start time/frame for later
                        img_text.tStart = t  # local t and not account for scr refresh
                        img_text.tStartRefresh = t_flip_global  # on global time
                        win.timeOnFlip(img_text, 'tStartRefresh')  # time at next scr refresh
                        img_text.setText("When \"Perform gesture\" appears, perform this gesture:")
                        img_text.setColor((1, 1, 1))
                        img_text.setAutoDraw(True)
                    if img_text.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if t_flip_global > img_text.tStartRefresh + instruction_secs_rep + img_secs + relax_secs - frame_tolerance:
                            # keep track of stop time/frame for later
                            img_text.tStop = t  # not accounting for scr refresh
                            img_text.frameNStop = n_frame  # exact frame index
                            win.timeOnFlip(img_text, 'tStopRefresh')  # time at next scr refresh
                            img_text.setAutoDraw(False)
                        elif t_flip_global > img_text.tStartRefresh + instruction_secs_rep + img_secs - frame_tolerance:
                            txt = "Relax arm."
                            if img_text.text != txt:
                                img_text.setText(txt)
                                img_text.setColor((0.9, 0, 0))
                                self.trigger(f"Release_{trigger_name}_{n_rep:02d}")
                        elif t_flip_global > img_text.tStartRefresh + instruction_secs_rep + img_secs - frame_tolerance - 0.05:
                            img_text.setText("")
                        elif t_flip_global > img_text.tStartRefresh + instruction_secs_rep - frame_tolerance:
                            txt = "Perform gesture:"
                            if img_text.text != txt:
                                img_text.setText(txt)
                                self.trigger(f"Start_{trigger_name}_{n_rep:02d}")
                                img_text.setColor((0.2, 1, 0.2))
                        elif t_flip_global > img_text.tStartRefresh + instruction_secs_rep - frame_tolerance - 0.05:
                            img_text.setText("")
                        else:
                            pass

                    # check if all components have finished
                    if not continue_routine:  # a component has requested a forced-end of Routine
                        break
                    continue_routine = False  # will revert to True if at least one component still running
                    for component in components:
                        if hasattr(component, "status") and component.status != FINISHED:
                            continue_routine = True
                            break  # at least one component has not yet finished

                    # refresh the screen
                    if continue_routine:  # don't flip if this routine is over or we'll get a blank screen
                        win.flip()

                # -------Ending Routine-------
                for component in components:
                    if hasattr(component, "setAutoDraw"):
                        component.setAutoDraw(False)
                trials.addData(f'{imgs[img_name]}.started', imgs[img_name].tStartRefresh)
                trials.addData(f'{imgs[img_name]}.stopped', imgs[img_name].tStopRefresh)\

                # the Routine "ThreeFingers" was not non-slip safe, so reset the non-slip timer
                routine_timer.reset()

        # Flip one final time so any remaining win.callOnFlip() and win.timeOnFlip() tasks get executed before quitting
        win.flip()

        self.data.stop()
        logging.flush()
        # make sure everything is closed down
        win.close()

        # return the location of the saved data
        return self.data.save_as
