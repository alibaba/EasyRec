#!/bin/bash

ps aux | grep "easy_rec.python.train_eval" | grep -v grep |  awk '{ print $2}' | while read line_str; do echo "kill $line_str"; kill -9 $line_str; done
