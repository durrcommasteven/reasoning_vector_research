from residual_stream_collection_tools import *

model = load_llama8br1()

prompt_list = read_prompts(output_type=None)

run_experiments(
    model,
    prompts=prompt_list,
    save_root="./reasoning_resid_data",
    max_new_tokens = 500,
    template_modes = ["base", "immediate_answer", "reasoning_boosted"]
)