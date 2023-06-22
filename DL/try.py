progress_bar = ["Step 1", "Step 2", "Step 3", "Step 4"]

startnum = 2

for index, item in enumerate(progress_bar[startnum:], start= st):
    if index  <= len(progress_bar):
       print(f"Progress: {index}/{len(progress_bar)} - {item}")
