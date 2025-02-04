
import os
import glob
import pyedflib
import h5py
import numpy as np
import pandas as pd
import datetime
from scipy.signal import resample
import matplotlib.pyplot as plt
from tqdm import tqdm
import multiprocessing
from loguru import logger
import argparse
import warnings
from scipy.signal import butter, filtfilt
import mne


class EDFToHDF5Converter:
    def __init__(self, root_dir, target_dir, resample_rate=512, num_threads=1, num_files=-1):
        self.resample_rate = resample_rate 
        self.root_dir = root_dir
        self.target_dir = target_dir
        self.num_threads = num_threads
        self.num_files = num_files
        self.file_locations = self.get_files() 
        # self.scorers = ['ES','LS','MS']
        self.flow_events = {'Central Apnea': 1, 'Mixed Apnea': 2, 'Obstructive Apnea': 3, 'Hypopnea': 4, 'RERA': 5}
        self.plm_events = {'P-Marker': 1, 'I-Marker': 2, 'LM Body position': 3, 'LM Resp': 4}
        self.arousal_events = {'Autonomic arousals': 1, 'Respiratory Arousal': 2}
        self.sleep_stages = {'Wake': 0, 'Rem': 1, 'N1': 2, 'N2': 3, 'N3': 4, 'Artifact': 5}


    def get_files(self):
        # Search for all '.edf' files within each subdirectory of the root directory
        file_paths = glob.glob(os.path.join(self.root_dir, '**/*.[eE][dD][fF]'), recursive=True)
        file_names = [os.path.basename(path) for path in file_paths]
        return file_paths, file_names
    
    def extract_start_time(self, file_path):
        with open(file_path, 'r') as file:
            lines = [next(file) for _ in range(5)]
            
        for line in lines:
            if line.startswith("Start Time:"):
                return line.split(": ", 1)[1].strip()
        return "Start Time not found"

    def create_signal_from_events(self, df, total_seconds, event_type = None):
        valid_types = {'flow', 'plm', 'arousal','stages'}
        if event_type not in valid_types:
            raise ValueError("event_type must be 'flow', 'plm', or 'arousal'")
        
        total_samples = int(total_seconds * self.resample_rate)
        # make initial array of zeros with length total_sec
        event_array = np.zeros(total_samples)

        # go through each event and mark the corresponding seconds in the array
        for _, row in df.iterrows():
            event_start = int(row['sec_from_start'] * self.resample_rate)
            event_stop = int(row['sec_from_start'] * self.resample_rate + row['dur'] * self.resample_rate)
            if event_type == 'flow':
                event_code = self.flow_events.get(row['event_type'], 0)
            elif event_type == 'plm':
                event_code = self.plm_events.get(row['event_type'], 0)
            elif event_type == 'arousal':
                event_code = self.arousal_events.get(row['event_type'], 0)
            elif event_type == 'stages':
                event_code = self.sleep_stages.get(row['event_type'])
            
            event_array[event_start:event_stop] = event_code
        
        return event_array

    def make_event_dataframe(self, folder, event_type = None):
        valid_types = {'flow', 'plm', 'arousal','stages'}
        if event_type not in valid_types:
            raise ValueError("event_type must be 'flow', 'plm', or 'arousal'")
        if event_type == 'flow':
            flow_file = os.path.join(self.root_dir,folder,'Flow Events.txt')
            df = pd.read_csv(flow_file, delimiter=';',skiprows=5, names=['start-stop', 'duration', 'event_type'])
        elif event_type == 'plm':
            flow_file = os.path.join(self.root_dir,folder,'PLM Events.txt')
            df = pd.read_csv(flow_file, delimiter=';',skiprows=5, names=['start-stop', 'duration', 'event_type'])
        elif event_type == 'arousal':
            flow_file = os.path.join(self.root_dir,folder,'Autonomic arousals.txt')
            dfAutonomic = pd.read_csv(flow_file, delimiter=';',skiprows=5, names=['start-stop', 'duration', 'event_type'])
            flow_file = os.path.join(self.root_dir,folder,'Classification arousals.txt')
            dfClassification = pd.read_csv(flow_file, delimiter=';',skiprows=5, names=['start-stop', 'duration', 'event_type'])
            df = pd.concat([dfAutonomic,dfClassification], ignore_index=True)
        elif event_type == 'stages':
            flow_file1 = os.path.join(self.root_dir,folder,'Flow Events.txt')
            start_time = self.extract_start_time(file_path = flow_file1)
            start_time = datetime.datetime.strptime(start_time, "%m/%d/%Y %I:%M:%S %p")
            flow_file = os.path.join(self.root_dir,folder,'Sleep profile.txt')
            df = pd.read_csv(flow_file, delimiter=';',skiprows=7, names=['start', 'event_type']) 
            df['start'] = pd.to_datetime(df['start'], format='%H:%M:%S,%f').dt.time
            df['sec_from_start'] = df['start'].apply(lambda x: (datetime.datetime.combine(datetime.date(1,1,1),x) - datetime.datetime.combine(datetime.date(1,1,1),start_time.time())).total_seconds())
            df['dur'] = 30
            df['event_type'] = df['event_type'].str.strip()
            if df['sec_from_start'].iloc[0] < 0:
                df['dur'][0] = 30 + df['sec_from_start'].iloc[0]
                df['sec_from_start'].iloc[0] = 0
            df.loc[df.sec_from_start < 0, 'sec_from_start'] += 24*60*60
            return df

        # df = pd.read_csv(flow_file, delimiter=';',skiprows=5, names=['start-stop', 'duration', 'event_type'])
        if len(df.values) != 0:
            df[['start', 'stop']] = df['start-stop'].str.split('-', expand=True)

            start_time = self.extract_start_time(file_path = flow_file)
            start_time = datetime.datetime.strptime(start_time, "%m/%d/%Y %I:%M:%S %p")

            df['start'] = pd.to_datetime(df['start'], format='%H:%M:%S,%f').dt.time
            df['stop'] = pd.to_datetime(df['stop'], format='%H:%M:%S,%f').dt.time
            df['sec_from_start'] = df['start'].apply(lambda x: (datetime.datetime.combine(datetime.date(1,1,1),x) - datetime.datetime.combine(datetime.date(1,1,1),start_time.time())).total_seconds())
            df['sec_from_stop'] = df['stop'].apply(lambda x: (datetime.datetime.combine(datetime.date(1,1,1),x) - datetime.datetime.combine(datetime.date(1,1,1),start_time.time())).total_seconds())

            df['duration'] = pd.to_numeric(df['duration'])

            df = df[['start', 'stop', 'duration', 'event_type', 'sec_from_start','sec_from_stop']]

            df.loc[df.sec_from_start < 0, 'sec_from_start'] += 24*60*60
            df.loc[df.sec_from_stop < 0, 'sec_from_stop'] += 24*60*60
            df['dur'] = df['sec_from_stop'] - df['sec_from_start']
        else:
            df  = pd.DataFrame(columns = ['start', 'stop', 'duration', 'event_type', 'sec_from_start','sec_from_stop', 'dur'])

        return df

    def convert_events(self, folder,total_seconds,event_type):
        df_events = self.make_event_dataframe(folder, event_type=event_type)
        event_array = self.create_signal_from_events(df=df_events, total_seconds=total_seconds, event_type=event_type)
        return event_array

    def read_edf_old(self, file_path):
        logger.info('reading edf')
        with pyedflib.EdfReader(file_path) as edf:
            signals = [edf.readSignal(i) for i in range(edf.signals_in_file)]
            sample_rates = np.array([edf.getSampleFrequency(i) for i in range(edf.signals_in_file)])
            channel_names = np.array([edf.getLabel(i) for i in range(edf.signals_in_file)])
            # annotations = []#edf.readAnnotations()
        return signals, sample_rates, channel_names
    def read_edf(self, file_path):
        logger.info('reading edf')
        raw = mne.io.read_raw_edf(file_path, preload=True)
        signals = [raw.get_data(picks=[ch_name])[0] for ch_name in raw.ch_names]
        sample_rates = np.array([raw.info['sfreq'] for _ in raw.ch_names])
        channel_names = np.array(raw.ch_names)
        return signals, sample_rates, channel_names

    def resample_signals_old(self, signals, sample_rates):
        # Vectorization and broadcasting could be applied within the resample function itself
        logger.info('resampling signals')
        resampled_signals = [resample(signal, int(len(signal) * self.resample_rate / rate))
                             for signal, rate in zip(signals, sample_rates)]
        standardized_signals = [(signal - np.mean(signal)) / np.std(signal) for signal in resampled_signals]
        # add signal names as input and filter the spo2 preprocessing as scaled 0 to 1
        return np.stack(standardized_signals)# Using np.stack for proper array dimensions


    def safe_standardize(self, signal):
        mean = np.mean(signal)
        std = np.std(signal)
        
        if std == 0:
            standardized_signal = (signal - mean)
        else:
            standardized_signal = (signal - mean) / std
        
        return standardized_signal
        
    def filter_signal(self, signal, sample_rate):
        print("Filtering signal")
        nyquist_freq = sample_rate / 2
        cutoff = min(self.resample_rate / 2, nyquist_freq)
        normalized_cutoff = cutoff / nyquist_freq
        b, a = butter(4, normalized_cutoff, btype='low', analog=False)
        filtered_signal = filtfilt(b, a, signal)
        return filtered_signal

    def resample_signals(self, signals, sample_rates):
        logger.info('resampling signals')
        resampled_signals = []
        for signal, rate in zip(signals, sample_rates):
            # Calculate the duration of the signal
            duration = len(signal) / rate
            
            # Original time points
            original_time_points = np.linspace(0, duration, num=len(signal), endpoint=False)
            
            # New sample rate and new time points
            new_sample_count = int(duration * self.resample_rate)
            new_time_points = np.linspace(0, duration, num=new_sample_count, endpoint=False)

            #filter signal
            if rate > self.resample_rate:
                signal = self.filter_signal(signal, rate)
            
            # Linear interpolation
            resampled_signal = np.interp(new_time_points, original_time_points, signal)
            
            # Standardize the resampled signal (optional, based on your need)
            # standardized_signal = (resampled_signal - np.mean(resampled_signal)) / np.std(resampled_signal)
            standardized_signal = self.safe_standardize(resampled_signal)
            
            if np.isnan(standardized_signal).any():
                logger.info('Found NaN in the resampled signal.')
                # Handle the NaN case here (e.g., skip or fix the signal)
                continue

            resampled_signals.append(standardized_signal)

        
        return np.stack(resampled_signals)  # Stack for a consistent output format

    def save_to_hdf5(self, signals, channel_names, annotation_signals, annotation_names, file_path):
        logger.info('saving hdf5')
        samples_per_chunk = 5 * 60 * self.resample_rate
        with h5py.File(file_path, 'w') as hdf:
            for signal, name in zip(signals, channel_names):
                dataset_name = self._get_unique_name(hdf, name)
                hdf.create_dataset(dataset_name, data=signal,
                                   dtype='float16', chunks=(samples_per_chunk,), compression="gzip")

            for annot_signal, annot_name in zip(annotation_signals, annotation_names):
                hdf.create_dataset(annot_name, data=annot_signal)

    def _get_unique_name(self, hdf, base_name):
        # Helper method to ensure dataset names are unique
        i = 1
        unique_name = base_name
        while unique_name in hdf:
            unique_name = f"{base_name}_{i}"
            i += 1
        return unique_name

    def get_annotations(self, total_seconds, folder):
        # folder_path = 'C:/Users/45223/OneDrive - Danmarks Tekniske Universitet/PhD/Oliver_triple_scored/triple_scored_studies1/CSA009/ES'
        flow_events = self.convert_events(folder = folder, total_seconds = total_seconds, event_type = 'flow')
        plm_events = self.convert_events( folder = folder, total_seconds = total_seconds, event_type = 'plm')
        arousal_events = self.convert_events( folder = folder, total_seconds = total_seconds, event_type = 'arousal')
        sleep_stages = self.convert_events( folder = folder, total_seconds = total_seconds, event_type = 'stages')

        return flow_events, plm_events, arousal_events, sleep_stages
    
    def convert(self, edf_path, hdf5_path):
        signals, sample_rates, channel_names = self.read_edf(edf_path)
        resampled_signals = self.resample_signals(signals, sample_rates)
        total_duration_seconds = len(signals[0])/sample_rates[0]
        event_signals = []
        event_signal_names = []
        self.save_to_hdf5(resampled_signals, channel_names,event_signals,event_signal_names, hdf5_path)

    def convert_multiprocessing(self, args):
        edf_files = args

        for edf_file in tqdm(edf_files, desc="Converting EDF files"):

            if edf_file.endswith(".edf"):
                replace_str = ".edf"
            elif edf_file.endswith(".EDF"):
                replace_str = ".EDF"
            hdf5_file = os.path.join(self.target_dir, edf_file.split('/')[-1].replace(replace_str, '.hdf5'))

            if os.path.exists(hdf5_file):
                logger.info(f"File already processed: {hdf5_file}")
                continue
            try:
                self.convert(edf_file, hdf5_file)
            except Exception as e:
                warnings.warn(f"Warning: Could not process the file {edf_file}. Error: {str(e)}")
                continue
        return [1]

    def convert_all(self):
        edf_files, edf_names = self.get_files() 
        # folders = self.get_folders()
        for edf_file in tqdm(edf_files, desc="Converting EDF files"):
            # edf_files = [os.path.join(folder, f) for f in os.listdir(os.path.join(self.root_dir,folder)) if f.endswith('.edf')]
            # edf_file = os.path.join(self.root_dir,edf_files[0])
            if edf_file.endswith(".edf"):
                replace_str = ".edf"
            elif edf_file.endswith(".EDF"):
                replace_str = ".EDF"
            hdf5_file = os.path.join(self.target_dir,edf_file.split('/')[-1].replace(replace_str, '.hdf5'))
            # logger.info(edf_file)
            # logger.info(hdf5_file)

            try:
                self.convert(edf_file, hdf5_file)
            except Exception as e:
                warnings.warn(f"Warning: Could not process the file {edf_file}. Error: {str(e)}")
                continue

    def convert_all_multiprocessing(self):
        edf_files, edf_names = self.get_files() 

        if self.num_files != -1:
            edf_files = edf_files[:self.num_files]

        edf_files_chunks = np.array_split(edf_files, self.num_threads)
        tasks = [(edf_files_chunk) for edf_files_chunk in edf_files_chunks]
        with multiprocessing.Pool(self.num_threads) as pool:
            preprocessed_results = [y for x in pool.imap_unordered(self.convert_multiprocessing, tasks) for y in x]

    def convert_with_annot(self, edf_path, hdf5_path, folder):
        signals, sample_rates, channel_names = self.read_edf(edf_path)
        resampled_signals = self.resample_signals(signals, sample_rates)
        total_duration_seconds = len(signals[0])/sample_rates[0]
        event_signals = []
        event_signal_names = []
        for scorer in self.scorers:
            scorer_folder = os.path.join(folder, scorer) 
            flow_events, plm_events, arousal_events, sleep_stages = self.get_annotations(total_seconds = total_duration_seconds, folder = scorer_folder)
            event_signals.extend([flow_events, plm_events, arousal_events, sleep_stages])
            event_signal_names.extend(['flow_events'+scorer, 'plm_events'+scorer, 'arousal_events'+scorer, 'sleep_stages'+scorer])
        self.save_to_hdf5(resampled_signals, channel_names,event_signals,event_signal_names, hdf5_path)

    def convert_all_with_annot(self):
        edf_files, edf_names = self.get_files() 
        folders = self.get_folders()
        for folder in tqdm(folders, desc="Converting EDF files"):
            edf_files = [os.path.join(folder, f) for f in os.listdir(os.path.join(self.root_dir,folder)) if f.lower().endswith('.edf')]
            edf_file = os.path.join(self.root_dir,edf_files[0])
            # header = self.extract_end_time(edf_file)
            # hdf5_file = os.path.join(self.root_dir, edf_file.replace('.edf', '.hdf5'))
            if edf_file.endswith(".edf"):
                replace_str = ".edf"
            elif edf_file.endswith(".EDF"):
                replace_str = ".EDF"
            hdf5_file = os.path.join(self.target_dir,edf_file.split('\\')[-1].replace(replace_str, '.hdf5'))

            # logger.info(hdf5_file)
            self.convert_with_annot(edf_file, hdf5_file, folder)
    def plot_results(self, resampled_signals, channel_names):
        print("plotting resampled_signals")
        num_signals = len(resampled_signals)
        fig, axs = plt.subplots(num_signals, 1, figsize=(15, 3*num_signals), sharex=True)
        samples_to_plot = 10 * self.resample_rate
        sample_to_start = 10 * self.resample_rate
        for i, (signal, name) in enumerate(zip(resampled_signals, channel_names)):
            signal_chunk = signal[sample_to_start:sample_to_start+samples_to_plot]
            axs[i].plot(signal_chunk)
            axs[i].set_title(name)
            axs[i].set_ylabel('Amplitude')
        
        axs[-1].set_xlabel('Samples')
        plt.tight_layout()
        plt.show()

    def plot_first_results(self, resampled_signals, channel_names):
        print("plotting resampled_signals")
        num_signals = len(resampled_signals)
        fig = plt.figure(figsize=(15, 3))
        samples_to_plot = 10 * self.resample_rate
        sample_to_start = 10 * self.resample_rate
        for i, (signal, name) in enumerate(zip(resampled_signals, channel_names)):
            signal_chunk = signal[sample_to_start:sample_to_start+samples_to_plot]
            plt.plot(signal_chunk)
            plt.title(name)
            plt.ylabel('Amplitude')
            break
        
        plt.xlabel('Samples')
        plt.tight_layout()
        plt.show()

    def process_and_plot_single_file(self, edf_path):
        signals, sample_rates, channel_names = self.read_edf(edf_path)
        resampled_signals = self.resample_signals(signals, sample_rates)
        #self.plot_results(resampled_signals, channel_names)
        self.plot_first_results(resampled_signals, channel_names)
    

def main():
    parser = argparse.ArgumentParser(description="Process data and create hdf5")
    parser.add_argument('--root_dir', type=str, required=True, help='Path to edf')
    parser.add_argument('--target_dir', type=str, required=True, help='Path to save hdf5')
    parser.add_argument("--num_threads", type=int, default=4, help="Number of threads for parallel processing")
    parser.add_argument("--num_files", type=int, default=-1, help="Number of files to process. If -1, process all")
    parser.add_argument("--resample_rate", type=int, default=256, help="Target sampling rate for hdf5 file.")
    args = parser.parse_args()

    os.makedirs(args.target_dir, exist_ok=True)

    converter = EDFToHDF5Converter(root_dir=args.root_dir,
                                target_dir=args.target_dir, 
                                num_threads=args.num_threads, 
                                num_files=args.num_files,
                                resample_rate=args.resample_rate)

    converter.convert_all_multiprocessing()


if __name__ == "__main__":
    main()



