#!/usr/bin/env bash
clang++ -Wall -Wextra -O2 -std=c++17 lightning.cc -lsfml-system -lsfml-graphics -lsfml-window -fopenmp
