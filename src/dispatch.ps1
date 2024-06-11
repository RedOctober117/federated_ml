# $num = Read-Host 'number of stations?'
# $stations = Get-Content -Path .\station_ids_greater_30.txt

# for (($i = 0); ($i -lt $num); ($i++)) {
#     start-process pwsh -ArgumentList "-NoExit", "-Command", "& {..\Scripts\Activate.ps1}; python client.py $(Get-Random $stations)"
# }

start-process pwsh -ArgumentList "-NoExit", "-Command", "& {..\Scripts\Activate.ps1}; python client.py 414088"
start-process pwsh -ArgumentList "-NoExit", "-Command", "& {..\Scripts\Activate.ps1}; python client.py 474204"
start-process pwsh -ArgumentList "-NoExit", "-Command", "& {..\Scripts\Activate.ps1}; python client.py 489543"


start-process pwsh -ArgumentList "-NoExit", "-Command", "& {..\Scripts\Activate.ps1}; python server.py"