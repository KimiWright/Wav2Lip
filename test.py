import torch

checkpoint_path_wav2lip = "/home/ksw38/RVL/color_syncnet/Wav2Lip/lipsync_expert.pth"
checkpoint_path_mine = "/home/ksw38/RVL/color_syncnet/Wav2Lip/checkpoints/checkpoint_step000510000.pth"

checkpoint_wav2lip = torch.load(checkpoint_path_wav2lip, map_location='cpu')
state_dict_wav2lip = checkpoint_wav2lip['state_dict']

checkpoint_mine = torch.load(checkpoint_path_mine, map_location='cpu')
state_dict_mine = checkpoint_mine['state_dict']

# for k in full_state_dict.keys():
#     print(k)

keys_mine = list(state_dict_mine.keys())
keys_wav2lip = list(state_dict_wav2lip.keys())

mismatched = 0
for i in range(len(keys_mine)):
    print(f"{keys_mine[i]}   |   {keys_wav2lip[i]}")
    if keys_mine[i] != keys_wav2lip[i]:
        print("\tMISMATCH")
        mismatched += 1
    # keys_mine_val = keys_mine[i].split('.')
    # keys_wav2lip_val = keys_wav2lip[i].split('.')
    # for j in range(1,5):
    #     if keys_mine_val[-j] != keys_wav2lip_val[-j]:
    #         print(f'\tMISMATCH at -{j}')
    #         mismatched += 1
print(f"\n{mismatched} keys were mismatched")
print(len(keys_mine))
print(len(keys_wav2lip))