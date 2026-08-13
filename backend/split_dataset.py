import splitfolders

# ၁။ မူရင်း Split မလုပ်ရသေးသော Dataset Folder လမ်းကြောင်း
input_folder = "data/hyperpigmentation" 

# ၂။ Split လုပ်ပြီး ခွဲထွက်လာမည့် Dataset Folder လမ်းကြောင်း
output_folder = "data/hyperpigmentation_split"

# Train 80%, Valid 10%, Test 10% အချိုးဖြင့် ခွဲပေးမည်
splitfolders.ratio(
    input_folder, 
    output=output_folder, 
    seed=1337, 
    ratio=(0.8, 0.1, 0.1), # 80% Train, 10% Validation, 10% Test
    group_prefix=None
)

print("Dataset ကို train, val, test အဖြစ် အောင်မြင်စွာ ခွဲခြားပြီးပါပြီ!")