from trl import SFTTrainer
from datasets import load_dataset
from random import randrange
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from transformers import TrainingArguments, TrainerCallback, TrainerState, TrainerControl
import os
import torch
from peft import AutoPeftModelForCausalLM, set_peft_model_state_dict
from transformers import AutoTokenizer
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from fire import Fire

# Load dataset from the hub
# dataset =  load_dataset('json',data_files="/lustre06/project/6007993/DataBank/enC4_Clean/fin-llama-dataset/data.jsonl",
#                          split="train", cache_dir="/lustre07/scratch/gagan30/arocr/cache")

# cluster_id = 0
def train(cluster_id):
    dataset = load_dataset("json",
                        data_files=f"/lustre07/scratch/gagan30/arocr/meta-llama/moe/dataset/WizardLM_evol_instruct_70k/cluster_{cluster_id}.json",
                        cache_dir="/lustre07/scratch/gagan30/arocr/cache",
                        split="train")

    print(f"dataset size: {len(dataset)}")
    print(dataset[randrange(len(dataset))])


    def format_instruction(sample):
        return f"""### Instruction:
        {sample['instruction']}
        ### Output:
        {sample['output']}
        """


    use_flash_attention = False
    # COMMENT IN TO USE FLASH ATTENTION
    # replace attention with flash attention
    if torch.cuda.get_device_capability()[0] >= 8:
        from flash_attn_patch import replace_llama_attn_with_flash_attn
        replace_llama_attn_with_flash_attn()
        print("Using flash attention")
        # replace_attn_with_flash_attn()
        use_flash_attention = True


    # Hugging Face model id
    # model_id = "NousResearch/Llama-2-7b-hf" # non-gated
    model_id = "/lustre07/scratch/gagan30/arocr/meta-llama/models/Llama-2-7b-hf"  # gated


    # BitsAndBytesConfig int-4 config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_id,
                                                 quantization_config=bnb_config,
                                                use_cache=False,
                                                device_map="auto")
    # model = model.to_bettertransformer()
    model.config.pretraining_tp = 1

    # Validate that the model is using flash attention, by comparing doc strings
    # if use_flash_attention:
    #     from flash_attn_patch import replace_llama_attn_with_flash_attn
    #     replace_llama_attn_with_flash_attn()


    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # LoRA config based on QLoRA paper
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj","v_proj","k_proj","o_proj","gate_proj","down_proj","up_proj"]
    )

    # prepare model for training
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)
    # model = model.to("cuda")

    model.print_trainable_parameters()

    args = TrainingArguments(
        output_dir=f"/lustre07/scratch/gagan30/arocr/meta-llama/moe_models/llama-2-13b-cluster-{cluster_id}",
        num_train_epochs=10,
        per_device_train_batch_size=6 if use_flash_attention else 4,
        gradient_accumulation_steps=2,
        gradient_checkpointing=True,
        optim="paged_adamw_32bit",
        logging_steps=10,
        save_strategy="epoch",
        learning_rate=2e-4,
        bf16=True,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="constant",
        save_total_limit=2,
        save_steps=1000,
        # disable_tqdm=True # disable tqdm since with packing values are in correct
    )


    max_seq_length = 256  # max sequence length for model and packing of the dataset


    class SavePeftModelCallback(TrainerCallback):
        def on_save(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            **kwargs,
        ):
            checkpoint_folder = os.path.join(
                args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")

            kwargs["model"].save_pretrained(checkpoint_folder)

            pytorch_model_path = os.path.join(
                checkpoint_folder, "pytorch_model.bin")
            torch.save({}, pytorch_model_path)
            return control


    class LoadBestPeftModelCallback(TrainerCallback):
        def on_train_end(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            **kwargs,
        ):
            print(
                f"Loading best peft model from {state.best_model_checkpoint} (score: {state.best_metric}).")
            best_model_path = os.path.join(
                state.best_model_checkpoint, "adapter_model.bin")
            adapters_weights = torch.load(best_model_path)
            model = kwargs["model"]
            set_peft_model_state_dict(model, adapters_weights)
            return control


    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        packing=True,
        args=args,
        callbacks=[SavePeftModelCallback(), LoadBestPeftModelCallback()],
        formatting_func=format_instruction
    )

    trainer.train()
    tokenizer.save_pretrained(args.output_dir)

    # save model
    trainer.save_model()

if __name__ == "__main__":
    Fire(train)
