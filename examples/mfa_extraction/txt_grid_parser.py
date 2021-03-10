# -*- coding: utf-8 -*-
# Copyright 2020 TensorFlowTTS Team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Create training file and durations from textgrids."""

import os
from dataclasses import dataclass
from pathlib import Path

import click
import numpy as np
import textgrid
import yaml
from tqdm import tqdm

import logging
import sys


logging.basicConfig(
    level=logging.DEBUG,
    stream=sys.stdout,
    format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
)


@dataclass
class TxtGridParser:
    sample_rate: int
    multi_speaker: bool
    txt_grid_path: str
    hop_size: int
    output_durations_path: str
    dataset_path: str
    training_file: str = "train.txt"
    phones_mapper = {"sil": "SIL", "sp": "SIL", "spn": "SIL", "": "END"}
    """ '' -> is last token in every cases i encounter so u can change it for END but there is a safety check
        so it'll fail always when empty string isn't last char in ur dataset just chang it to silence then
    """
    sil_phones = set(phones_mapper.keys())

    def parse(self):
        speakers = (
            [
                i
                for i in os.listdir(self.txt_grid_path)
                if os.path.isdir(os.path.join(self.txt_grid_path, i))
            ]
            if self.multi_speaker
            else []
        )
        data = []

        if speakers:
            for speaker in speakers:
                file_list = os.listdir(os.path.join(self.txt_grid_path, speaker))
                self.parse_text_grid(file_list, data, speaker)
        else:
            file_list = os.listdir(self.txt_grid_path)
            self.parse_text_grid(file_list, data, "")

        with open(os.path.join(self.dataset_path, self.training_file), "w") as f:
            f.writelines(data)
    
    def parse_punc_intervals(self, text_grid, txtFile):
        words = text_grid[0]
        punc_phones = ["!","?"]
        hasPunc = False
        puncIntervals=[]
        endingPunc='NA'
        with open(txtFile,'r') as f:
          sentence =f.read().rstrip().lower()
          if '!' in sentence or '?' in  sentence:
            hasPunc = True
            if sentence.endswith("?") or sentence.endswith("?\"") or sentence.endswith("?'"):
                endingPunc='?'
            elif sentence.endswith("!") or sentence.endswith("!\"") or sentence.endswith("!'"):
                endingPunc='!'
          split=sentence.replace('-', ' ').split(" ")
          split[:]=[x for x in split if x]
          wordInSentenceIdx=-1
          for idx, wordInterval in enumerate(words.intervals):
            if wordInterval.mark=="" or wordInterval.mark=='<unk>':
              continue
            else:
              wordInSentenceIdx+=1
            if wordInSentenceIdx>=0 and len(split)>wordInSentenceIdx:
              wordInSentence=split[wordInSentenceIdx]
              #print(wordInSentence)
            else:
              continue
            trimmedWord = wordInSentence.rstrip()
            lastChar=trimmedWord[-1]
            if (lastChar == "'" or lastChar == '"') and len(trimmedWord)>1:
                lastChar=trimmedWord[-2]
            if lastChar in punc_phones and len(words.intervals)>idx+1:
              nextInterval=words.intervals[idx+1]
              if nextInterval.mark=="":
                puncIntervals.append({'punc':lastChar, 'interval':nextInterval}) 
        return {'puncIntervals': puncIntervals, 'hasPunc':hasPunc, 'sentence':sentence, 'endingPunc':endingPunc}

    def phon_in_punc(self, interval, puncIntervals):
      for idx, puncInterval in enumerate(puncIntervals):
        if (puncInterval['interval'].minTime<=interval.minTime \
        and puncInterval['interval'].maxTime>=interval.maxTime \
        and interval.mark != '') or \
        (puncInterval['interval'].minTime==interval.minTime \
        and puncInterval['interval'].maxTime==interval.maxTime \
        and interval.mark == '') :
          return {'addPuncPhon': True, 'phon': puncInterval['punc']}
      return {'addPuncPhon':False, 'phon':{}}

    def parse_text_grid(self, file_list: list, data: list, speaker_name: str):
        logging.info(
            f"\n Parse: {len(file_list)} files, speaker name: {speaker_name} \n"
        )
        for f_name in tqdm(file_list):
            text_grid = textgrid.TextGrid.fromFile(
                os.path.join(self.txt_grid_path, speaker_name, f_name)
            )
            pha = text_grid[1]
            durations = []
            phs = []
            txtFile= '{}/{}/{}.txt'.format(self.dataset_path,speaker_name,f_name.split(".")[0])
            parsedPuncs = self.parse_punc_intervals(text_grid, txtFile)
            puncIntervals = parsedPuncs['puncIntervals']
            hasPunc = parsedPuncs['hasPunc']
            puncMarkCreated = False
            for iterator, interval in enumerate(pha.intervals):
                mark = interval.mark
                if mark in self.sil_phones:
                    punc = self.phon_in_punc(interval, puncIntervals)
                    if punc['addPuncPhon']:
                      mark = punc['phon']
                      puncMarkCreated = True
                    else:
                      mark = self.phones_mapper[mark]
                      if mark == "END":
                          assert iterator == pha.intervals.__len__() - 1
                          # check if empty ph is always last example in your dataset if not fix it

                dur = interval.duration() * (self.sample_rate / self.hop_size)
                durations.append(round(dur))
                phs.append(mark)

            if parsedPuncs['endingPunc'] != 'NA' and phs[len(phs)-2] == 'SIL' and phs[len(phs)-1] == 'END':
                phs[len(phs)-2] = parsedPuncs['endingPunc']
            full_ph = " ".join(phs)
            
            if hasPunc and not puncMarkCreated:
                logging.info(
                    f"\n Punc mark not created for: {txtFile}\n {parsedPuncs['sentence']} \n {full_ph} \n"
                )

            assert full_ph.split(" ").__len__() == durations.__len__()  # safety check

            base_name = f_name.split(".TextGrid")[0]
            np.save(
                os.path.join(self.output_durations_path, f"{base_name}-durations.npy"),
                np.array(durations).astype(np.int32),
                allow_pickle=False,
            )
            data.append(f"{speaker_name}/{base_name}|{full_ph}|{speaker_name}\n")


@click.command()
@click.option(
    "--yaml_path", default="examples/fastspeech2_libritts/conf/fastspeech2libritts.yaml"
)
@click.option("--dataset_path", default="dataset", type=str, help="Dataset directory")
@click.option("--text_grid_path", default="mfa/parsed", type=str)
@click.option("--output_durations_path", default="dataset/durations")
@click.option("--sample_rate", default=24000, type=int)
@click.option("--multi_speakers", default=1, type=int, help="Use multi-speaker version")
@click.option("--train_file", default="train.txt")
def main(
    yaml_path: str,
    dataset_path: str,
    text_grid_path: str,
    output_durations_path: str,
    sample_rate: int,
    multi_speakers: int,
    train_file: str,
):

    with open(yaml_path) as file:
        attrs = yaml.load(file)
        hop_size = attrs["hop_size"]

    Path(output_durations_path).mkdir(parents=True, exist_ok=True)

    txt_grid_parser = TxtGridParser(
        sample_rate=sample_rate,
        multi_speaker=bool(multi_speakers),
        txt_grid_path=text_grid_path,
        hop_size=hop_size,
        output_durations_path=output_durations_path,
        training_file=train_file,
        dataset_path=dataset_path,
    )
    txt_grid_parser.parse()


if __name__ == "__main__":
    main()
