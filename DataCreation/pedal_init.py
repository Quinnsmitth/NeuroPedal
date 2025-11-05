from pedalboard import load_plugin
plugin_path = "/Library/Audio/Plug-Ins/VST3/Audiority/Dr Drive.vst3"

#Initalizing Plug In to best fit the sound of. TS-9
def initialize_plugin(plugin):
    plugin.parameters['hq'].value = True #Use highquality Over Sampling
    plugin.parameters['od_power'].value = True # Make sure the circut is active
    plugin.parameters['od_attack'].value = "A1" # Best  Tube-Style for clipping
    plugin.parameters['input_gain'].value = 0.0 # Unity Gain - no prepedal boosting
    plugin.parameters['output_gain'].value = 0.0 # Unity Gain - no postpedal boosting
    plugin.parameters['mix'].value = 100.0 # Full Effected Signal
    plugin.parameters['od_level'].value = 0.5 # Use to set output level
    plugin.parameters['noise_gate'].value = 0.0 # Noise Gate Off
    plugin.parameters['bypass'].value = False  # Bypass Off - Pedal is Active
    return plugin

plugin = initialize_plugin(load_plugin(plugin_path))
print(plugin.parameters)