#!/usr/bin/env bash

echo '12af' | awk '{print strtonum("0x"$0)}'