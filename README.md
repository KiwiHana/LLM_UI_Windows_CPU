# LLM_UI_Windows_CPU
Run LLM UI Application on Windows 11 CPU 


conda create -n llm python=3.9

conda activate llm

pip install --pre --upgrade bigdl-llm[all]

pip install gradio mdtex2html

python LLM_demo_v1.0.py




LLM_demo_v1.0.py

theme3.json

checkpoint
-	bigdl_llm_llama2_13b_q4_0.bin
-	bigdl_llm_starcoder_q4_0.bin
-	ggml-chatglm2-6b-q4_0.bin
