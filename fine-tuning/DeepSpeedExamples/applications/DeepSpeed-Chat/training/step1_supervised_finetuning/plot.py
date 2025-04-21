import pandas as pd
import os
import matplotlib.pyplot as plt

def scrape(script_file, world_size):
  with open(script_file) as f:
    script = f.readlines()
    
  losses = {}
  loss_per_step = {}
  
  num_micro_batches = None
  for line in script:
    if "Total Micro Batches" in line:
      num_micro_batches = int(line.split(",")[1].replace("Total Micro Batches","").replace(" ",""))
  
  for line in script:
    if "Step:" in line and "Rank:" in line and "loss = " in line:
      line_splitted = line.replace(" ","").split(",")
      epoch = int(line_splitted[0].split(":")[-1].replace(" ",""))
      step = int(line_splitted[1].replace("Step:",""))
      rank = line_splitted[2].replace("Rank:","")
      loss = float(line_splitted[3].replace("loss=",""))
      step = epoch * num_micro_batches + step
      if step not in loss_per_step:
        loss_per_step[step] = {}
      loss_per_step[step][rank] = loss
      if len(loss_per_step[step]) == world_size:
        avg_loss = sum([val for val in loss_per_step[step].values()]) / world_size
        losses[step] = avg_loss
  losses_df = pd.DataFrame(data=losses.values(), columns=["loss"], index=losses.keys()).sort_index()
  
  return losses_df
        
def plot(df, output_dir):
  plot = df.plot(kind='line', 
                 ylabel="Train Loss Error",
                 xlabel="Step",
                 title="Fine-tuning LLaMA3.1-8B",
                 grid=True,
                 legend=False)
  plt.savefig(f"{output_dir}/plot.pdf")
  

def main(script_file, world_size, output_dir):
  plot(scrape(script_file, 
              world_size), 
       output_dir)

if __name__ == '__main__':
  script_file = "/work/09308/zhengmk/finetune_llama3.1_DL_assignment/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/training_scripts/llama3.1/rtx-58759.out"
  world_size = 8
  output_dir = "/work/09308/zhengmk/finetune_llama3.1_DL_assignment/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/training_scripts/llama3.1/output_step1_llama3.1_8b"
  
  main(script_file=script_file,
       world_size=world_size,
       output_dir=output_dir)