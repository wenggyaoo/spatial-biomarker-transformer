from langchain.schema import HumanMessage, SystemMessage


# class langchain_api_demo:
#     try:
#         from api import chat
#     except ImportError:
#         print("ERROR: Could not import 'chat' from api.py.")
#         print("Please make sure api.py is in the same directory and is configured correctly.")
#         exit()
#     except Exception as e:
#         print(f"An error occurred while initializing the API: {e}")
#         exit()
#
#     length = 32
#
#     system_prompt = f"""You are an expert bioinformatician specializing in spatial omics. Your task if to map a list of biomarkers that are used in some
#     spatial-omics studies to a biologically context-rich vector of length {length}, which will encompass information related to the nature of this biomarker,
#     the cell-type of cells that it marks and feel free to add more. These informations will be further tackeled using deep-learning models so please
#     make sure that the embedding yielded is context-rich and self-explainable and consistent. respond only with the vector
#     you embedded the biomarker into and nothing else."""
#
#     user_prompt = f"The biomarker to be embedded is CD45'"
#
#     res = chat.invoke([
#         SystemMessage(content=system_prompt),
#         HumanMessage(content=user_prompt)
#     ])
#
#     print(res.content)
#
# from config import Config
# from esm_embedder import *
# import torch
#
# class gpt_embedding_demo:
#     embedder = BiomarkerEmbedder(Config())
#     print(embedder.forward(['CD45'], torch.tensor([42.0])))

# from amino_acid_seq_loader import *
#
# class gpt_amino_acid_seq_demo:
#     print(get_human_protein_sequence("CD45"))

import esm
class esm_embedding_demo:
    model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()