#!/usr/bin/env bash
set -e

ports=(50051 50053 50055 50057)
#ports=(50051 50053 50055 50057 50059 50061 50063 50065 50067 50069 50071 50073 50075 50077 50079 50081 50083 50085 50087 50089 50091 50093 50095 50097 50099 50101 50103 50105 50107 50109 50111 50113 50115 50117 50119 50121 50123 50125 50127 50129 50131 50133 50135 50137 50139 50141 50143 50145 50147 50149)
for port in "${ports[@]}"; do
    lsof -i:"$port" -t | xargs kill || true
done

# python-is-python-3
python -c 'import sys;assert sys.version_info.major==3'

for port in "${ports[@]}"; do
    bind_address="127.0.0.1"
    screen -Sdm "exp-port-$port" python main.py "$bind_address:$port" "$port"
done

sleep 0.5s

screen -ls
