#!/bin/bash
for nationality in */ ; do
  #echo $nationality
  for id in $nationality* ; do
    #echo "$id"
    for path in $id/* ; do
    echo "$path"
      COUNTER=0
      for file in $path/*.wav ; do
      #echo "/$file"
      #echo "$COUNTER"
        yes | ffmpeg -i $file -af "anlmdn" -acodec pcm_s16le -ar 16000 $path/${COUNTER}_no_noise.wav
        yes | ffmpeg -i $path/${COUNTER}_no_noise.wav -af "loudnorm" -acodec pcm_s16le -ar 16000 $path/${COUNTER}_normalized.wav
        yes | ffmpeg -i $path/${COUNTER}_normalized.wav -af silenceremove=1:0:-50dB -acodec pcm_s16le -ar 16000 $path/${COUNTER}.wav
        rm $file
        rm $path/${COUNTER}_no_noise.wav
        rm $path/${COUNTER}_normalized.wav
        COUNTER=$((COUNTER + 1))
      done
    done
  done
done
