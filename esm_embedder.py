import torch
import torch.nn as nn
import esm
import os
import ast
import pickle
import numpy as np
from tqdm import tqdm
import csv
import pandas as pd
from typing import Dict, List, Optional
from torch.nn import Parameter
import json
import re
import warnings
from Bio import Entrez


class BiomarkerEmbedder(nn.Module):
    """
    Encodes biomarker intensity profiles into cell embeddings using various methods.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = config.device
        self.method = getattr(config, 'embedding_method', 'learnable')
        self.biomarker_info = self._load_biomarker_info()
        self.biomarker_mapping = self._load_biomarker_mapping()
        self.valid_biomarkers = list(self.biomarker_info.keys())
        self.biomarker_embeddings = None  # Will be a dict or a ParameterDict, defined in the init methods
        self.biomarker_to_idx = {}
        self.idx_to_biomarker = {}
        self.embedding_dim = None
        self.gene_summaries = {}
        self._initialize_method()

    # ... (all other methods like _load_biomarker_info, _init_esm, etc., remain exactly the same) ...
    def _load_biomarker_info(self):
        """ Returns a dict of biomarker name to sequence """
        biomarker_info = {}
        filepath = self.config.biomarker_sequence_file

        try:
            df = pd.read_csv(filepath)

            for _, row in df.iterrows():
                biomarker_name = str(row.iloc[0]).strip()
                amino_acid_seq = row.iloc[1]

                if pd.isna(amino_acid_seq):
                    biomarker_info[biomarker_name] = {
                        'type': 'legit_no_sequence',
                        'sequence': None
                    }
                elif str(amino_acid_seq).upper() == 'N/A':
                    biomarker_info[biomarker_name] = {
                        'type': 'not_legit',
                        'sequence': None
                    }
                else:
                    biomarker_info[biomarker_name] = {
                        'type': 'has_sequence',
                        'sequence': str(amino_acid_seq).strip()
                    }

            print(f"Loaded {len(biomarker_info)} biomarkers from {filepath}")

        except FileNotFoundError:
            print(f"ERROR: The file '{filepath}' was not found.")
            return {}
        except Exception as e:
            print(f"ERROR loading biomarker info: {e}")
            return {}

        return biomarker_info

    def _load_biomarker_mapping(self):
        """ Returns a biomarker name mapping dict. """
        try:
            json_path = self.config.biomarker_name_mapping_file

            if os.path.exists(json_path):
                with open(json_path, 'r', encoding='utf-8') as f:
                    mapping = json.load(f)
                print(f"Loaded biomarker mapping from {json_path}")
                return mapping
            else:
                print(f"No JSON mapping file found at {json_path}")
                print("Proceeding without EMPTY filtering")
                return {}
        except Exception as e:
            print(f"Error loading biomarker mapping: {e}")
            return {}
        
    def _sanitize_name(self, name):
        """Sanitize parameter name by replacing invalid characters"""
        # Replace periods and other invalid characters with underscores
        return str(name).replace('.', '_').replace(' ', '_').replace('-', '_')

    def _initialize_method(self):
        """Initialize method-specific components."""
        if self.method == 'onehot':
            self._init_onehot()
        elif self.method == 'learnable':
            self._init_learnable()
        elif self.method == 'esm':
            self._init_esm()
        # elif self.method == 'genept':
        #     self._init_genept()
        else:
            raise ValueError(f"Unsupported embedding method: {self.method}")

    def _init_onehot(self):
        """Initialize one-hot embeddings."""
        # Dimension will be set based on vocabulary size
        self.biomarker_embeddings = {}  # One-hot embeddings will not be learnable

    def _init_learnable(self):
        """Initialize learnable embeddings."""
        self.embedding_dim = getattr(self.config, 'learnable_embedding_dim', 512)
        self.biomarker_embeddings = nn.ParameterDict()  # Learnable embeddings will be created later

    def _init_esm(self):
        """Initialize ESM model for protein embeddings."""
        print("Initializing ESM model...")
        self.biomarker_embeddings = {}  # Combination of fixed and learnable embeddings
        self.embedding_dim = 1280
        try:
            self.esm_model, self.esm_alphabet = esm.pretrained.esm2_t33_650M_UR50D()
            self.esm_model.eval()
            self.esm_model = self.esm_model.to(self.device)
            self.esm_batch_converter = self.esm_alphabet.get_batch_converter()
            print("ESM model loaded successfully")
        except Exception as e:
            print(f"Failed to load ESM model: {e}")
            raise

    # def _init_genept(self):
    #     """Initialize GenePT-specific components."""
    #     self.embedding_dim = 512  # OpenAI text-embedding-ada-002 dimension
    #     self.genept_mode = getattr(self.config, 'genept_mode', 'summary')

    def build_biomarker_vocab(self, all_biomarkers: list = None):
        """Build vocabulary and compute embeddings using the specified method."""
        print(f"Building biomarker vocabulary using {self.method} method...")

        # ALWAYS use the pre-filtered valid biomarkers (EMPTY ones already removed)
        if all_biomarkers is None:
            all_biomarkers = self.valid_biomarkers
        else:
            # If biomarkers are provided externally, still filter them
            filtered_biomarkers = []
            for biomarker in all_biomarkers:
                if biomarker in self.biomarker_mapping:
                    if self.biomarker_mapping[biomarker] != "EMPTY":
                        filtered_biomarkers.append(biomarker)
                else:
                    filtered_biomarkers.append(biomarker)
            all_biomarkers = filtered_biomarkers

        print(f"Processing {len(all_biomarkers)} valid biomarkers (EMPTY ones filtered out)")

        # Build vocabulary mapping
        for i, biomarker_name in enumerate(all_biomarkers):
            self.biomarker_to_idx[biomarker_name] = i
            self.idx_to_biomarker[i] = biomarker_name

        # Method-specific embedding computation
        if self.method == 'onehot':
            self._build_onehot_embeddings(all_biomarkers)
        elif self.method == 'learnable':
            self._build_learnable_embeddings(all_biomarkers)
        elif self.method == 'esm':
            self._build_esm_embeddings(all_biomarkers)
        # elif self.method == 'genept':
        #     self._build_genept_embeddings(all_biomarkers)
        return

    def _build_onehot_embeddings(self, all_biomarkers: list):
        """Build one-hot embeddings."""
        print("Building one-hot embeddings...")
        self.embedding_dim = len(all_biomarkers)
        for i, biomarker_name in enumerate(all_biomarkers):
            # One hot embeddings are not trainable
            embedding = torch.zeros(self.embedding_dim, requires_grad=False)
            embedding[i] = 1.0
            safe_name = self._sanitize_name(biomarker_name)
            self.biomarker_embeddings[safe_name] = embedding

    def _build_learnable_embeddings(self, all_biomarkers: list):
        """Build learnable embeddings."""
        print("Building learnable embeddings...")
        for biomarker_name in all_biomarkers:
<<<<<<< HEAD
            embedding = self._create_learnable_fallback_embedding(biomarker_name)
            safe_name = self._sanitize_name(biomarker_name)
            self.register_parameter(f"emb:{safe_name}", embedding)
=======
            # Apply the same mapping as forward() so vocab keys are canonical names.
            # DEBUG: without this, names like 'CD49f' map to 'CD49' at lookup time but would be stored under 'CD49f', causing a KeyError.
            canonical_name = self.biomarker_mapping.get(biomarker_name, biomarker_name)
            if canonical_name == "EMPTY":
                continue
            safe_name = self._sanitize_name(canonical_name)
            if safe_name in self.biomarker_embeddings:
                continue  # multiple originals (CD49f, CD49d, …) share one canonical
            embedding = self._create_learnable_fallback_embedding(canonical_name)
            self.register_parameter(f"emb__{safe_name}", embedding)
>>>>>>> 6aef426 (try implementations)
            self.biomarker_embeddings[safe_name] = embedding

    def _create_learnable_fallback_embedding(self, biomarker_name: str) -> torch.Tensor:
        """Create a learnable embedding as fallback when NCBI fails."""
        # Create a deterministic but unique embedding based on biomarker name
        # This ensures reproducibility while giving each biomarker a unique vector

        # Use hash of biomarker name as seed for reproducibility
        import hashlib
        hash_object = hashlib.md5(biomarker_name.encode())
        seed = int(hash_object.hexdigest(), 16) % (2 ** 32)

        # Create random state with this seed
        rng = np.random.RandomState(seed)

        # Generate embedding with correct dimension for current method
        embedding = rng.normal(0, 0.1, self.embedding_dim)
        embedding = embedding / np.linalg.norm(embedding)  # Normalize
        return Parameter(torch.tensor(embedding, dtype=torch.float32, requires_grad=True).to(self.device))

    # def _fetch_ncbi_gene_summary(self, gene_name: str) -> Optional[str]:
    #     """
    #     Fetch gene summary from NCBI gene database.
    #     Returns None if not found, gene summary string if found.
    #     """
    #     try:
    #         # Configure email for NCBI
    #         Entrez.email = getattr(self.config, 'ncbi_email', "aguo0521@connect.hku.hk")

    #         # Search for gene
    #         search_handle = Entrez.esearch(db="gene", term=f"{gene_name}[Gene Name] AND Homo sapiens[Organism]",
    #                                        retmax=1)
    #         search_results = Entrez.read(search_handle)
    #         search_handle.close()

    #         if not search_results["IdList"]:
    #             print(f"No NCBI entry found for {gene_name}")
    #             return None

    #         gene_id = search_results["IdList"][0]

    #         # Fetch gene summary using XML format for better parsing
    #         fetch_handle = Entrez.efetch(db="gene", id=gene_id, rettype="xml", retmode="text")
    #         gene_info = fetch_handle.read()
    #         fetch_handle.close()

    #         # Parse XML to extract summary
    #         import xml.etree.ElementTree as ET

    #         try:
    #             root = ET.fromstring(gene_info)

    #             # Look for gene summary in the XML structure
    #             for elem in root.iter():
    #                 if elem.tag == 'Entrezgene_summary' and elem.text:
    #                     summary = elem.text.strip()
    #                     if summary:
    #                         return summary

    #             # If no summary found, try to get gene name and description
    #             gene_desc = None
    #             for elem in root.iter():
    #                 if elem.tag == 'Gene-ref_desc' and elem.text:
    #                     gene_desc = elem.text.strip()
    #                     break

    #             if gene_desc:
    #                 return gene_desc

    #             print(f"No summary found in NCBI data for {gene_name}")
    #             return None

    #         except ET.ParseError as e:
    #             print(f"Error parsing XML for {gene_name}: {e}")
    #             return None

    #     except Exception as e:
    #         print(f"Error fetching NCBI summary for {gene_name}: {e}")
    #         return None

    # def _build_genept_embeddings(self, all_biomarkers: list):
    #     """Build embeddings using GenePT approach with NCBI summaries and fallbacks."""
    #     print("Building GenePT embeddings...")

    #     # Statistics tracking
    #     ncbi_success = 0
    #     learnable_fallback = 0

    #     print("Processing biomarkers...")

    #     for biomarker_name in tqdm(all_biomarkers, desc="Processing biomarkers"):
    #         # Try to fetch NCBI description
    #         summary = self._fetch_ncbi_gene_summary(biomarker_name)

    #         if summary is not None:
    #             # NCBI description successfully retrieved
    #             text_input = self._prepare_genept_text_input(biomarker_name, summary)

    #             try:
    #                 # Use OpenAI embeddings API
    #                 from openai import OpenAI
    #                 client = OpenAI()  # Make sure API key is set in environment

    #                 response = client.embeddings.create(
    #                     input=text_input,
    #                     model="text-embedding-ada-002"
    #                 )

    #                 embedding_vector = response.data[0].embedding
    #                 embedding = Parameter(torch.tensor(embedding_vector, dtype=torch.float32))
    #                 self.biomarker_embeddings[biomarker_name] = embedding
    #                 ncbi_success += 1

    #             except Exception as e:
    #                 print(f"Error getting OpenAI embedding for {biomarker_name}: {e}")
    #                 # Fall back to learnable embedding
    #                 embedding = self._create_learnable_fallback_embedding(biomarker_name)
    #                 self.biomarker_embeddings[biomarker_name] = embedding
    #                 learnable_fallback += 1

    #         else:
    #             # NCBI description not retrieved, use learnable fallback
    #             print(f"Using learnable fallback for {biomarker_name}")
    #             embedding = self._create_learnable_fallback_embedding(biomarker_name)
    #             self.biomarker_embeddings[biomarker_name] = embedding
    #             learnable_fallback += 1

    #     print(f"\nGenePT Embedding Statistics:")
    #     print(f"  NCBI descriptions successfully used: {ncbi_success}")
    #     print(f"  Learnable fallback embeddings: {learnable_fallback}")
    #     print(f"  Total processed: {ncbi_success + learnable_fallback}")

    # def _prepare_genept_text_input(self, biomarker_name: str, gene_summary: str) -> str:
    #     """Prepare text input for GenePT embedding based on mode."""
    #     if self.genept_mode == 'name_only':
    #         return biomarker_name
    #     elif self.genept_mode == 'summary':
    #         return f"Gene Name {biomarker_name} Summary {gene_summary}"
    #     elif self.genept_mode == 'full':
    #         return f"{biomarker_name} Official Full Name {biomarker_name} Summary {gene_summary}"
    #     else:
    #         return f"Gene Name {biomarker_name} Summary {gene_summary}"

    def _build_esm_embeddings(self, all_biomarkers: list):
        """Build embeddings using ESM model."""
        print("Building ESM embeddings...")

        for biomarker_name in tqdm(all_biomarkers, desc="Computing ESM Embeddings"):
            
            # Check biomarker legitimacy
            if biomarker_name in self.biomarker_info:
                biomarker_info = self.biomarker_info[biomarker_name]

                # Fallback to learnable embedding if no sequence
                if biomarker_info['type'] == 'legit_no_sequence':
                    emb = self._create_learnable_fallback_embedding(biomarker_name)
                    safe_name = self._sanitize_name(biomarker_name)
                    self.register_parameter(f"emb:{safe_name}", emb)

                elif biomarker_info['type'] == 'has_sequence':
                    sequence = biomarker_info['sequence']
                    raw_emb = self._compute_esm_embedding(biomarker_name, sequence)
                    emb = raw_emb.detach()
                    emb.requires_grad = False
            else:
                # Fallback to learnable embedding if not legit
                emb = self._create_learnable_fallback_embedding(biomarker_name)
                safe_name = self._sanitize_name(biomarker_name)
                self.register_parameter(f"emb:{safe_name}", emb)

            safe_name = self._sanitize_name(biomarker_name)
            self.biomarker_embeddings[safe_name] = emb
        
        self.esm_model.cpu()
        del self.esm_model
        torch.cuda.empty_cache()
        return

    def _compute_esm_embedding(self, biomarker_name: str, sequence: str) -> torch.Tensor:
        """Compute ESM embedding for a protein sequence."""
        try:
            data = [(biomarker_name, sequence)]
            batch_labels, batch_strs, batch_tokens = self.esm_batch_converter(data)
            batch_tokens = batch_tokens.to(self.device)

            with torch.no_grad():
                results = self.esm_model(batch_tokens, repr_layers=[33], return_contacts=False)

            token_representations = results["representations"][33]
            sequence_representations = token_representations[0, 1:-1].mean(0)

            return sequence_representations.to(self.device)

        except Exception as e:
            print(f"Error computing ESM embedding for {biomarker_name}: {e}")
            return self._create_learnable_fallback_embedding(biomarker_name)

    def forward(self, biomarker_names: list, intensities: torch.Tensor) -> torch.Tensor:
        """Forward pass to create cell embeddings.
        
        biomarker_names: List of biomarker names (strings)
        intensities: 2D torch array of shape (num_cells, num_biomarkers)
        """
        valid_biomarkers = []
        valid_biomarker_indices = []
        for i, name in enumerate(biomarker_names):
            if name in self.biomarker_mapping:
                name = self.biomarker_mapping[name]
                if name == "EMPTY":
                    continue
            if self._sanitize_name(name) in self.biomarker_embeddings:
                valid_biomarkers.append(name)
                valid_biomarker_indices.append(i)

        if len(valid_biomarkers) < len(biomarker_names):
            warnings.warn(f"{len(biomarker_names) - len(valid_biomarkers)} biomarkers not found in embeddings and will be ignored.")

        if not valid_biomarkers:
            raise ValueError("No valid biomarkers found in the input.")
        valid_intensities = intensities[:, valid_biomarker_indices]
        valid_intensities = torch.tensor(valid_intensities, dtype=torch.float32, device=self.device)

        # if self.method == 'genept':
        #     cell_mode = getattr(self.config, 'genept_cell_mode', 'weighted')
        #     return self.create_genept_cell_embeddings(valid_biomarkers, valid_intensities, mode=cell_mode)

        biomarker_embeddings = torch.stack([
            self.biomarker_embeddings[self._sanitize_name(name)].to(self.device)
            for name in valid_biomarkers])
        cell_profile = torch.matmul(valid_intensities, biomarker_embeddings)
        return cell_profile

    def get_embedding_dim(self):
        """Returns the dimension of the embeddings."""
        return self.embedding_dim

    # def create_genept_cell_embeddings(self, biomarker_names: list, intensities: torch.Tensor,
    #                                   mode: str = 'weighted') -> torch.Tensor:
    #     """
    #     Create cell embeddings using GenePT method with different aggregation modes
    #     """
    #     if len(biomarker_names) == 0:
    #         return torch.zeros(self.embedding_dim, device=self.device)

    #     valid_embeddings = []
    #     valid_intensities = []

    #     for i, name in enumerate(biomarker_names):
    #         if name in self.biomarker_embeddings:
    #             valid_embeddings.append(self.biomarker_embeddings[name].to(self.device))
    #             valid_intensities.append(intensities[i])

    #     if not valid_embeddings:
    #         return torch.zeros(self.embedding_dim, device=self.device)

    #     embeddings = torch.stack(valid_embeddings)
    #     intensities_tensor = torch.stack(valid_intensities)

    #     # Also apply the fix here for safety, though less likely to be an issue
    #     intensity_sum = intensities_tensor.sum()

    #     if mode == 'weighted':
    #         normalized_intensities = intensities_tensor / (intensity_sum + 1e-9)
    #         if normalized_intensities.ndim == 1:
    #             normalized_intensities = normalized_intensities.unsqueeze(1)
    #         cell_embedding = torch.sum(embeddings * normalized_intensities, dim=0)
    #     elif mode == 'mean':
    #         cell_embedding = torch.mean(embeddings, dim=0)
    #     elif mode == 'max':
    #         cell_embedding = torch.max(embeddings, dim=0)[0]
    #     else:
    #         normalized_intensities = intensities_tensor / (intensity_sum + 1e-9)
    #         if normalized_intensities.ndim == 1:
    #             normalized_intensities = normalized_intensities.unsqueeze(1)
    #         cell_embedding = torch.sum(embeddings * normalized_intensities, dim=0)

    #     return cell_embedding

    def get_single_biomarker_embedding(self, biomarker_name: str) -> torch.Tensor:
        """Get embedding for a single biomarker by name."""
        if biomarker_name in self.biomarker_mapping:
            biomarker_name = self.biomarker_mapping[biomarker_name]
            if biomarker_name == "EMPTY":
                raise ValueError(f"Invalid biomarker name: {biomarker_name}")
        safe_name = self._sanitize_name(biomarker_name)
        if safe_name in self.biomarker_embeddings:
            return self.biomarker_embeddings[safe_name].to(self.device)
        else:
            raise ValueError(f"Invalid biomarker name: {biomarker_name}")

    def get_batched_embeddings(self, biomarker_names: list[str]) -> torch.Tensor:
        """Get embeddings for a list of biomarker names."""
        mapped_names = [self.biomarker_mapping.get(name, name) for name in biomarker_names]
        embeddings = torch.stack([
            self.biomarker_embeddings[self._sanitize_name(name)] for name in mapped_names
        ], dim=0).to(self.device)  # (N, d_embed)

        return embeddings