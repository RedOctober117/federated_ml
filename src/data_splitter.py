path = 'normalized_ev_data_reduced.csv'

station_list = []

with open('station_ids.txt', 'r') as file:
  for line in file.readlines():
    station_list.append(line.strip('\n').strip())

station_dict = {}

with open('station_id_occurances.txt', 'r') as file:
  for line in file.readlines():
    clean = line.strip('\n').strip()
    if clean not in station_dict:
      station_dict[clean] = 1
    else:
      station_dict[clean] += 1

for k, v in station_dict.items():
  if v > 30:
    print(k)