import os
#DONT USE THIS ONE
def getDataClean():
    #jack add your paths into this list.
    possible_data = ["Users/quinnsmith/Desktop/guitar_data/clean","/Volumes/PortableSSD/guitar_data/clean"]
    for data_path in possible_data:
           if os.path.isdir(data_path):
               return data_path
    raise FileNotFoundError("No valid clean data path found in possible_data list.")
# DONT USE THIS ONE
def getDataDistorted():
    #jack add your paths into this list.
    possible_data = ["Users/quinnsmith/Desktop/guitar_data/dist","/Volumes/PortableSSD/guitar_data/dist"]

    for data_path in possible_data:
        if os.path.isdir(data_path):
            return data_path
    raise FileNotFoundError("No valid data path found in possible_data list.")

def getData(type):
    # for classification all files go in 'dist'
    if type not in ['clean','dist','split']:
        raise ValueError("type must be 'clean','dist', or 'split'")
    
    possible_data = [f"/Users/quinnsmith/Desktop/guitar_data/{type}",
                     f"/Volumes/PortableSSD/guitar_data/{type}"]
    
    for data_path in possible_data:
       if os.path.isdir(data_path):
              return data_path
    raise FileNotFoundError(f"No valid {type} data path found in possible_data list.")

