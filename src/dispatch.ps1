$num = Read-Host 'number of stations?'
$stations = Get-Random -Shuffle (Get-Content -Path .\station_ids.txt) 

for ($i -eq 0; $i -lt $num; $i++) {
    start-process pwsh -ArgumentList "-Command", "& {..\Scripts\Activate.ps1}; python client.py $stations"
}