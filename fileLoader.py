def getDataClean():
    #jack add your paths into this list.
    possible_data = ["Users/quinnsmith/Desktop/guitar_data/clean","/Volumes/PortableSSD/guitar_data/clean"]

    for data_path in possible_data:
        try:
            with open(data_path) as f:
                return data_path
        except FileNotFoundError:
            continue
    raise FileNotFoundError("No valid data path found in possible_data list.")

def getDataDistorted():
    #jack add your paths into this list.
    possible_data = ["Users/quinnsmith/Desktop/guitar_data/dist","/Volumes/PortableSSD/guitar_data/dist"]

    for data_path in possible_data:
        try:
            with open(data_path) as f:
                return data_path
        except FileNotFoundError:
            continue
    raise FileNotFoundError("No valid data path found in possible_data list.")