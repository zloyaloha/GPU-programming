#!/usr/bin/env bash

cd ../build

cmake ..
make

./lab2/lab2 < test.txt
python3 ../lab2/out.py