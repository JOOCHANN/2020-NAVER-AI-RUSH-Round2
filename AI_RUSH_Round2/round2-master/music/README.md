# Music Classification

## Tasks

The main purpose of the tasks are to classify music pieces, while taking advantage of the full length audio files of music transformed to mel-spectrograms, and some meta data including artists, lyrics, etc.
Inputs are given as image-like data having two main dimensions of freqeuncy and time.
Outputs are labels designed for each question.

Q1, Q3 are single class classification tasks, where Q2 is a multi-tag classification task.
Evaluations for Q1, Q3 are done with top-1 accuracy, while Q2 will be evaluated with micro-averaged f1-scores.

### Q1 (rush4): Station Classification
* 4 station classes: 외로울때, 파티할때, 집중할때, 휴식할때
* 1 station class for each track

### Q2 (rush5): Multi-tagging Mood
* 100 tags for mood
* 1~5 tags for each track

### Q3 (rush6): Japanese Music Genre Classification
* 4 genre classes: 가요곡/엔카, R&B, 레게, J-ROCK
* 1 genre class for each track

## How to run
* fix `train.sh` and run
* run nsml command directly
```
nsml run -d rush4-1 -v -a "--config_file q1/config.yaml"
```


## Dataset Structure

### Q1
* train
  * train_data
    * meta (given in q1/meta)
      * album_meta.json
      * artist_meta.json
      * lyric_meta.json
      * track_meta.json
    * mel_spectrogram
      * [track_id].npy
  * train_label
  ```
  {'track_index': {data_idx: track_id}, 
   'station_name': {data_idx: label}}
  ```
* test
  * test_data
    * meta
      * album_meta.json
      * artist_meta.json
      * lyric_meta.json
      * track_meta.json
      * q1_test.json
      ```
      {'track_index': {data_idx: track_id}}
      ```
    * mel_spectrogram
      * [track_id].npy

### Q2
* train
  * train_data
    * meta (given in q2/meta)
      * album_meta.json
      * artist_meta.json
      * lyric_meta.json
      * track_meta.json
    * mel_spectrogram
      * [track_id].npy
  * train_label
  ```
  {'track_index': {data_idx: track_id}, 
   'mood_tag': {data_idx: label_list}}
  ```
* test
  * test_data
    * meta
      * album_meta.json
      * artist_meta.json
      * lyric_meta.json
      * track_meta.json
      * q2_test.json
    ```
    {'track_index': {data_idx: track_id}}
    ```
    * mel_spectrogram
      * [track_id].npy

### Q3
* train
  * train_data
    * meta (empty)
    * mel_spectrogram
      * [track_id].npy
  * train_label (given as q3/meta/track_meta.json)
  ```
  {'track_index': {data_idx: track_id}, 
   'track_title': {data_idx: title},
   'artist_name_list': {data_idx: list of artists},
   'album_name': {data_idx: album},
   'genre': {data_idx: label}
  ```
* test
  * test_data
    * meta
      * q3_test.json
      ```
      {'track_index': {data_idx: track_id}}
      ```
    * mel_spectrogram
      * [track_id].npy

## Baseline
* 2D CNN w/ 5 layers
* First 1000 frames of mel-spectrograms for each track input


## Mel-spectrogram parameters
- sample rate: 16000
- number of mel bins: 128
- shape of each data point: [1, 128, n_frames]
- each time frame length: 32ms
