#!/bin/bash

kill `ps ax | grep trainer | grep python | awk '{print $1}'`
