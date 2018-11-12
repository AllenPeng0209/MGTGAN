import numpy as np
import glob
import datetime
import math
import random
import os
import shutil
import matplotlib.pyplot as plt
import pretty_midi
from pypianoroll import Multitrack, Track
import librosa.display
from utils import *
import convert_clean
import write_midi

ROOT_PATH = '/Users/allenpeng/Documents/Learning/Multi_Domian&Track_Music_Transfer/my_datasets/'
test_ratio = 0.1
LAST_BAR_MODE = 'remove'


music_gerne_selected = ['country','shuffle','funk','bossanova','rock']
#music_gerne_selected = ['bossanova']




def set_piano_roll_to_instrument(piano_roll, instrument, velocity=100, tempo=120.0, beat_resolution=16):
    # Calculate time per pixel
    tpp = 60.0 / tempo / float(beat_resolution)
    threshold = 60.0 / tempo / 4
    phrase_end_time = 60.0 / tempo * 4 * piano_roll.shape[0]
    # Create piano_roll_search that captures note onsets and offsets
    piano_roll = piano_roll.reshape((piano_roll.shape[0] * piano_roll.shape[1], piano_roll.shape[2]))
    piano_roll_diff = np.concatenate((np.zeros((1, 128), dtype=int), piano_roll, np.zeros((1, 128), dtype=int)))
    piano_roll_search = np.diff(piano_roll_diff.astype(int), axis=0)
    # Iterate through all possible(128) pitches
    
    for note_num in range(128):
        # Search for notes
        start_idx = (piano_roll_search[:, note_num] > 0).nonzero()
        start_time = list(tpp * (start_idx[0].astype(float)))
        # print('start_time:', start_time)
        # print(len(start_time))
        end_idx = (piano_roll_search[:, note_num] < 0).nonzero()
        end_time = list(tpp * (end_idx[0].astype(float)))
        # print('end_time:', end_time)
        # print(len(end_time))
        duration = [pair[1] - pair[0] for pair in zip(start_time, end_time)]
        # print('duration each note:', duration)
        # print(len(duration))
        
        temp_start_time = [i for i in start_time]
        temp_end_time = [i for i in end_time]
        
        for i in range(len(start_time)):
            # print(start_time)
            if start_time[i] in temp_start_time and i != len(start_time) - 1:
                # print('i and start_time:', i, start_time[i])
                t = []
                current_idx = temp_start_time.index(start_time[i])
                for j in range(current_idx + 1, len(temp_start_time)):
                    # print(j, temp_start_time[j])
                    if temp_start_time[j] < start_time[i] + threshold and temp_end_time[j] <= start_time[i] + threshold:
                        # print('popped start time:', temp_start_time[j])
                        t.append(j)
                # print('popped temp_start_time:', t)
                for _ in t:
                    temp_start_time.pop(t[0])
                    temp_end_time.pop(t[0])
# print('popped temp_start_time:', temp_start_time)

    start_time = temp_start_time
    # print('After checking, start_time:', start_time)
    # print(len(start_time))
    end_time = temp_end_time
    # print('After checking, end_time:', end_time)
    # print(len(end_time))
    duration = [pair[1] - pair[0] for pair in zip(start_time, end_time)]
    # print('After checking, duration each note:', duration)
    # print(len(duration))
    
    if len(end_time) < len(start_time):
        d = len(start_time) - len(end_time)
        start_time = start_time[:-d]
    # Iterate through all the searched notes
    for idx in range(len(start_time)):
        if duration[idx] >= threshold:
            # Create an Note object with corresponding note number, start time and end time
            note = pretty_midi.Note(velocity=velocity, pitch=note_num, start=start_time[idx], end=end_time[idx])
            # Add the note to the Instrument object
            instrument.notes.append(note)
        else:
            if start_time[idx] + threshold <= phrase_end_time:
                # Create an Note object with corresponding note number, start time and end time
                note = pretty_midi.Note(velocity=velocity, pitch=note_num, start=start_time[idx],
                                        end=start_time[idx] + threshold)
            else:
                # Create an Note object with corresponding note number, start time and end time
                note = pretty_midi.Note(velocity=velocity, pitch=note_num, start=start_time[idx],
                                        end=phrase_end_time)
            # Add the note to the Instrument object
            instrument.notes.append(note)
# Sort the notes by their start time
    instrument.notes.sort(key=lambda note: note.start)
# print(max([i.end for i in instrument.notes]))
# print('tpp, threshold, phrases_end_time:', tpp, threshold, phrase_end_time)



def get_bar_piano_roll(piano_roll):
    if int(piano_roll.shape[0] % 64) is not 0:
        if LAST_BAR_MODE == 'fill':
            piano_roll = np.concatenate((piano_roll, np.zeros((64 - piano_roll.shape[0] % 64, 128))), axis=0)
        elif LAST_BAR_MODE == 'remove':
            piano_roll = np.delete(piano_roll,  np.s_[-int(piano_roll.shape[0] % 64):], axis=0)
    piano_roll = piano_roll.reshape(-1, 64, 128, 3)
    return piano_roll



def to_binary(bars, threshold=0.0):
    """Turn velocity value into boolean"""
    track_is_max = tf.equal(bars, tf.reduce_max(bars, axis=-1, keep_dims=True))
    track_pass_threshold = (bars > threshold)
    out_track = tf.logical_and(track_is_max, track_pass_threshold)
    return out_track





def save_midis(bars, file_path, tempo=80.0):
    padded_bars = np.concatenate((np.zeros((bars.shape[0], bars.shape[1], 24, bars.shape[3])), bars,
                                  np.zeros((bars.shape[0], bars.shape[1], 20, bars.shape[3]))), axis=2)
    pause = np.zeros((bars.shape[0], 64, 128, bars.shape[3]))
    images_with_pause = padded_bars
    images_with_pause = images_with_pause.reshape(-1, 64, padded_bars.shape[2], padded_bars.shape[3])

    images_with_pause_list = []
    for ch_idx in range(padded_bars.shape[3]):
      images_with_pause_list.append(images_with_pause[:, :, :, ch_idx].reshape(images_with_pause.shape[0],
                                                                               images_with_pause.shape[1],
                                                                               images_with_pause.shape[2]))
    print("succesful 1" ,  images_with_pause.shape)
    write_midi.write_piano_rolls_to_midi(images_with_pause_list, filename=file_path,
                              tempo=tempo, beat_resolution=4)

    print("succesful 2" )


def train_pre_process_midi():
    print('start to preprocess train midi')
    for music_gerne in music_gerne_selected:
        print("start working on" ,music_gerne)
        
        
        
        """build path"""
        if not os.path.exists(os.path.join(ROOT_PATH, music_gerne+'/'+music_gerne+'_train')):
            os.makedirs(os.path.join(ROOT_PATH, music_gerne+'/'+music_gerne+'_train'))
        if not os.path.exists(os.path.join(ROOT_PATH, music_gerne+'/'+music_gerne+'_train/origin_midi')):
            os.makedirs(os.path.join(ROOT_PATH, music_gerne+'/'+music_gerne+'_train/origin_midi'))

        
        """1. divide the original set into train and train sets"""
        l = [f for f in os.listdir(os.path.join(ROOT_PATH, music_gerne +'/'+music_gerne+'_midi'))]
        print(l)
     

        for i in range(len(l)):
            shutil.move(os.path.join(ROOT_PATH, music_gerne +'/'+music_gerne+'_midi', l[i]),
                        os.path.join(ROOT_PATH, music_gerne+'/'+music_gerne+'_train'+'/origin_midi', l[i]))
        
        """2. convert_clean.py"""
        convert_clean.main(music_gerne=music_gerne, mode='train')
        
        """3. choose the clean midi from original sets"""
        if not os.path.exists(os.path.join(ROOT_PATH, music_gerne+'/'+music_gerne+'_train/cleaner_midi')):
            os.makedirs(os.path.join(ROOT_PATH, music_gerne+'/'+music_gerne+'_train/cleaner_midi'))
        if not os.path.exists(os.path.join(ROOT_PATH, music_gerne+'/'+music_gerne+'_train/cleaner')):
            os.makedirs(os.path.join(ROOT_PATH, music_gerne+'/'+music_gerne+'_train/cleaner'))
        l = [f for f in os.listdir(os.path.join(ROOT_PATH, music_gerne+'/'+music_gerne+'_train/cleaner'))]
        print(l)
        print(len(l))
        for i in l:
            shutil.copy(os.path.join(ROOT_PATH, music_gerne+'/'+music_gerne+'_train'+'/origin_midi', os.path.splitext(i)[0] + '.mid'),
                        os.path.join(ROOT_PATH, music_gerne+'/'+music_gerne+'_train/cleaner_midi', os.path.splitext(i)[0] + '.mid'))
        
        """4. merge and crop"""
        if not os.path.exists(os.path.join(ROOT_PATH, music_gerne+'/'+music_gerne+'_train/cleaner_midi_gen')):
            os.makedirs(os.path.join(ROOT_PATH, music_gerne+'/'+music_gerne+'_train/cleaner_midi_gen'))
        if not os.path.exists(os.path.join(ROOT_PATH, music_gerne+'/'+music_gerne+'_train/cleaner_npy')):
            os.makedirs(os.path.join(ROOT_PATH, music_gerne+'/'+music_gerne+'_train/cleaner_npy'))
        l = [f for f in os.listdir(os.path.join(ROOT_PATH, music_gerne+'/'+music_gerne+'_train/cleaner_midi'))]
        print(l)
        count = 0
        for i in range(len(l)):
            try:
                multitrack = Multitrack(beat_resolution=4, name=os.path.splitext(l[i])[0])
                x = pretty_midi.PrettyMIDI(os.path.join(ROOT_PATH, music_gerne+'/'+music_gerne+'_train/cleaner_midi', l[i]))
                multitrack.parse_pretty_midi(x)
                stacked = multitrack.get_stacked_pianorolls()
                print("stacked_shape:",stacked.shape)
                pr = get_bar_piano_roll(stacked)
                print("pr_shape:",pr.shape)
                pr_clip = pr[:, :, 24:108]
                print("pr_clip_shape:",pr_clip.shape)
                if int(pr_clip.shape[0] % 4) != 0:
                    pr_clip = np.delete(pr_clip, np.s_[-int(pr_clip.shape[0] % 4):], axis=0)
                pr_re = pr_clip.reshape(-1, 64, 84, 3)
                
                print("pr_re_shape:",pr_re.shape)
                print(os.path.join(ROOT_PATH, music_gerne+'/'+music_gerne+'_train/cleaner_midi_gen', os.path.splitext(l[i])[0] +'.mid'))
                save_midis(pr_re, os.path.join(ROOT_PATH, music_gerne+'/'+music_gerne+'_train/cleaner_midi_gen', os.path.splitext(l[i])[0] +'.mid'))
        
                np.save(os.path.join(ROOT_PATH, music_gerne+'/'+music_gerne+'_train/cleaner_npy', os.path.splitext(l[i])[0] + '.npy'), pr_re)
            except:
                count += 1
                print('Wrong', l[i])
                continue
        print('total fails:',count)

        """5. concatenate into a big binary numpy array file"""
        l = [f for f in os.listdir(os.path.join(ROOT_PATH, music_gerne+'/'+music_gerne+'_train/cleaner_npy'))]
        print(l)
        train = np.load(os.path.join(ROOT_PATH, music_gerne+'/'+music_gerne+'_train/cleaner_npy', l[0]))
        print(train.shape, np.max(train))
        for i in range(1, len(l)):
            print(i, l[i])
            t = np.load(os.path.join(ROOT_PATH, music_gerne+'/'+music_gerne+'_train/cleaner_npy', l[i]))
            train = np.concatenate((train, t), axis=0)
            print(train.shape)
            np.save(os.path.join(ROOT_PATH, music_gerne+'/'+music_gerne+'_train'+'/'+music_gerne+'_train.npy'), (train > 0.0))
    
        """6. separate numpy array file into single phrases"""
        if not os.path.exists(os.path.join(ROOT_PATH, music_gerne+'/'+music_gerne+'_train/phrase_train')):
            os.makedirs(os.path.join(ROOT_PATH, music_gerne+'/'+music_gerne+'_train/phrase_train'))
        x = np.load(os.path.join(ROOT_PATH, music_gerne+'/'+music_gerne+'_train'+'/'+music_gerne+'_train.npy'))
        print(x.shape)
        count = 0
        for i in range(x.shape[0]):
            if np.max(x[i]):
                count += 1
                np.save(os.path.join(ROOT_PATH, music_gerne+'/'+music_gerne+'_train/phrase_train'+'/'+music_gerne+'_train_{}.npy'.format(i+1)), x[i])
                print(x[i].shape)
        # if count == 11216:
        #     break
    print(count)

def test_pre_process_midi():
    print('start to preprocess test midi')
    for music_gerne in music_gerne_selected:
        print("start working on" ,music_gerne)
        
        """build path"""
        if not os.path.exists(os.path.join(ROOT_PATH, music_gerne+'/'+music_gerne+'_test')):
            os.makedirs(os.path.join(ROOT_PATH, music_gerne+'/'+music_gerne+'_test/'))
        if not os.path.exists(os.path.join(ROOT_PATH, music_gerne+'/'+music_gerne+'_test/origin_midi')):
            os.makedirs(os.path.join(ROOT_PATH, music_gerne+'/'+music_gerne+'_test/origin_midi'))

        
        """1. divide the original set into train and test sets"""
        l = [f for f in os.listdir(os.path.join(ROOT_PATH, music_gerne +'/'+music_gerne+'_midi'))]
        print(l)
        idx = np.random.choice(len(l), int(test_ratio * len(l)), replace=False)
        print(len(idx))
        for i in idx:
            shutil.move(os.path.join(ROOT_PATH, music_gerne +'/'+music_gerne+'_midi', l[i]),
                         os.path.join(ROOT_PATH, music_gerne+'/'+music_gerne+'_test'+'/origin_midi', l[i]))

        """2. convert_clean.py"""
        convert_clean.main(music_gerne=music_gerne, mode='test')

        """3. choose the clean midi from original sets"""
        if not os.path.exists(os.path.join(ROOT_PATH, music_gerne+'/'+music_gerne+'_test/cleaner_midi')):
             os.makedirs(os.path.join(ROOT_PATH, music_gerne+'/'+music_gerne+'_test/cleaner_midi'))
        if not os.path.exists(os.path.join(ROOT_PATH, music_gerne+'/'+music_gerne+'_test/cleaner')):
            os.makedirs(os.path.join(ROOT_PATH, music_gerne+'/'+music_gerne+'_test/cleaner'))
        l = [f for f in os.listdir(os.path.join(ROOT_PATH, music_gerne+'/'+music_gerne+'_test/cleaner'))]
        print(l)
        print(len(l))
        for i in l:
            shutil.copy(os.path.join(ROOT_PATH, music_gerne+'/'+music_gerne+'_test'+'/origin_midi', os.path.splitext(i)[0] + '.mid'),
                        os.path.join(ROOT_PATH, music_gerne+'/'+music_gerne+'_test/cleaner_midi', os.path.splitext(i)[0] + '.mid'))

        """4. merge and crop"""
        if not os.path.exists(os.path.join(ROOT_PATH, music_gerne+'/'+music_gerne+'_test/cleaner_midi_gen')):
            os.makedirs(os.path.join(ROOT_PATH, music_gerne+'/'+music_gerne+'_test/cleaner_midi_gen'))
        if not os.path.exists(os.path.join(ROOT_PATH, music_gerne+'/'+music_gerne+'_test/cleaner_npy')):
            os.makedirs(os.path.join(ROOT_PATH, music_gerne+'/'+music_gerne+'_test/cleaner_npy'))
        l = [f for f in os.listdir(os.path.join(ROOT_PATH,music_gerne+'/'+music_gerne+'_test/cleaner_midi'))]
        print(l)
        count = 0
        for i in range(len(l)):
            try:
                multitrack = Multitrack(beat_resolution=4, name=os.path.splitext(l[i])[0])
                x = pretty_midi.PrettyMIDI(os.path.join(ROOT_PATH, music_gerne+'/'+music_gerne+'_test/cleaner_midi', l[i]))
                multitrack.parse_pretty_midi(x)
                stacked = multitrack.get_stacked_pianorolls()
                print("stacked_shape:",stacked.shape)
                pr = get_bar_piano_roll(stacked)
                print("pr_shape:",pr.shape)
                pr_clip = pr[:, :, 24:108]
                print("pr_clip_shape:",pr_clip.shape)
                if int(pr_clip.shape[0] % 4) != 0:
                    pr_clip = np.delete(pr_clip, np.s_[-int(pr_clip.shape[0] % 4):], axis=0)
                pr_re = pr_clip.reshape(-1, 64, 84, 3)
                
                print("pr_re_shape:",pr_re.shape)
                print(os.path.join(ROOT_PATH, music_gerne+'/'+music_gerne+'_test/cleaner_midi_gen', os.path.splitext(l[i])[0] +'.mid'))
                save_midis(pr_re, os.path.join(ROOT_PATH, music_gerne+'/'+music_gerne+'_test/cleaner_midi_gen', os.path.splitext(l[i])[0] +'.mid'))

                np.save(os.path.join(ROOT_PATH, music_gerne+'/'+music_gerne+'_test/cleaner_npy', os.path.splitext(l[i])[0] + '.npy'), pr_re)
            except:
                count += 1
                print('Wrong', l[i])
                continue
        print('total fails:',count)

        """5. concatenate into a big binary numpy array file"""
        l = [f for f in os.listdir(os.path.join(ROOT_PATH, music_gerne+'/'+music_gerne+'_test/cleaner_npy'))]
        print(l)
        train = np.load(os.path.join(ROOT_PATH, music_gerne+'/'+music_gerne+'_test/cleaner_npy', l[0]))
        print(train.shape, np.max(train))
        for i in range(1, len(l)):
            print(i, l[i])
            t = np.load(os.path.join(ROOT_PATH, music_gerne+'/'+music_gerne+'_test/cleaner_npy', l[i]))
            train = np.concatenate((train, t), axis=0)
        print(train.shape)
        np.save(os.path.join(ROOT_PATH, music_gerne+'/'+music_gerne+'_test'+'/'+music_gerne+'_test.npy'), (train > 0.0))

        """6. separate numpy array file into single phrases"""
        if not os.path.exists(os.path.join(ROOT_PATH, music_gerne+'/'+music_gerne+'_test/phrase_test')):
             os.makedirs(os.path.join(ROOT_PATH, music_gerne+'/'+music_gerne+'_test/phrase_test'))
        x = np.load(os.path.join(ROOT_PATH, music_gerne+'/'+music_gerne+'_test'+'/'+music_gerne+'_test.npy'))
        print(x.shape)
        count = 0
        for i in range(x.shape[0]):
            if np.max(x[i]):
                count += 1
                np.save(os.path.join(ROOT_PATH, music_gerne+'/'+music_gerne+'_test/phrase_test'+'/'+music_gerne+'_test_{}.npy'.format(i+1)), x[i])
                print(x[i].shape)
            # if count == 11216:
            #     break
        print(count)

        """some other codes"""
        # filepaths = []
        # msd_id_list = []
        # for dirpath, _, filenames in os.walk(os.path.join(ROOT_PATH, 'MIDI/Sinfonie Data')):
        #     for filename in filenames:
        #         if filename.endswith('.mid'):
        #             msd_id_list.append(filename)
        #             filepaths.append(os.path.join(dirpath, filename))
        # print(filepaths)
        # print(msd_id_list)
        # for i in range(len(filepaths)):
        #     shutil.copy(filepaths[i], os.path.join(ROOT_PATH, 'MIDI/classic/classic_midi/{}'.format(msd_id_list[i])))

        # x1 = np.load(os.path.join(ROOT_PATH, 'MIDI/classic/classic_train/classic_train_piano_1.npy'))
        # x2 = np.load(os.path.join(ROOT_PATH, 'MIDI/classic/classic_train/classic_train_piano_2.npy'))
        # x3 = np.load(os.path.join(ROOT_PATH, 'MIDI/classic/classic_train/classic_train_piano_3.npy'))
        # x4 = np.load(os.path.join(ROOT_PATH, 'MIDI/classic/classic_train/classic_train_piano_4.npy'))
        # x5 = np.load(os.path.join(ROOT_PATH, 'MIDI/classic/classic_train/classic_train_piano_5.npy'))
        # x = np.concatenate((x1, x2, x3, x4, x5), axis=0)
        # print(x.shape)
        # np.save(os.path.join(ROOT_PATH, 'MIDI/classic/classic_train/classic_train_piano.npy'), x)


        # multitrack = Multitrack(beat_resolution=4, name='YMCA')
        # x = pretty_midi.PrettyMIDI(os.path.join(ROOT_PATH, 'MIDI/famous_songs/P2C/origin/YMCA.mid'))
        # multitrack.parse_pretty_midi(x)
        #
        # category_list = {'Piano': [], 'Drums': []}
        # program_dict = {'Piano': 0, 'Drums': 0}
        #
        # for idx, track in enumerate(multitrack.tracks):
        #     if track.is_drum:
        #         category_list['Drums'].append(idx)
        #     else:
        #         category_list['Piano'].append(idx)
        # tracks = []
        # merged = multitrack[category_list['Piano']].get_merged_pianoroll()
        #
        # # merged = multitrack.get_merged_pianoroll()
        # print(merged.shape)
        #
        # pr = get_bar_piano_roll(merged)
        # print(pr.shape)
        # pr_clip = pr[:, :, 24:108]
        # print(pr_clip.shape)
        # if int(pr_clip.shape[0] % 4) != 0:
        #     pr_clip = np.delete(pr_clip, np.s_[-int(pr_clip.shape[0] % 4):], axis=0)
        # pr_re = pr_clip.reshape(-1, 64, 84, 1)
        # print(pr_re.shape)
        # save_midis(pr_re, os.path.join(ROOT_PATH, 'MIDI/famous_songs/P2C/merged_midi/YMCA.mid'), 127)
        # np.save(os.path.join(ROOT_PATH, 'MIDI/famous_songs/P2C/merged_npy/YMCA.npy'), (pr_re > 0.0))

if __name__ == "__main__":
   
    test_pre_process_midi()
    train_pre_process_midi()
