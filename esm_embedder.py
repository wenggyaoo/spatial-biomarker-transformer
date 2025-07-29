import torch
import torch.nn as nn
import esm
import os
import ast
import pickle
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Optional
import openai
from langchain.chat_models import AzureChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import HumanMessage, SystemMessage
import json
import re


class BiomarkerEmbedder(nn.Module):
    """
    Encodes biomarker intensity profiles into cell embeddings using various methods.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = config.device
        self.method = getattr(config, 'embedding_method', 'gpt')

        # Common attributes across all methods
        self.biomarker_embeddings = {}
        self.biomarker_to_idx = {}
        self.idx_to_biomarker = {}
        self.embedding_dim = None

        # Method-specific initialization
        self._initialize_method()

    def _initialize_method(self):
        """Initialize method-specific components."""
        if self.method == 'esm':
            self._init_esm()
        elif self.method == 'gpt':
            self._init_gpt()
        elif self.method == 'random':
            self._init_random()
        elif self.method == 'onehot':
            self._init_onehot()
        else:
            raise ValueError(f"Unsupported embedding method: {self.method}")

    def _init_esm(self):
        """Initialize ESM-specific components."""
        print("Loading ESM-2 model...")
        self.model, self.alphabet = esm.pretrained.esm2_t12_35M_UR50D()
        self.model = self.model.to(self.device)
        self.model.eval()
        self.repr_layer = self.model.num_layers
        self.embedding_dim = self.model.embed_dim

    def _init_gpt(self):
        """Initialize GPT API-specific components."""
        self.embedding_dim = getattr(self.config, 'gpt_embedding_dim', 1536)

    def _init_random(self):
        """Initialize random embedding components."""
        self.embedding_dim = getattr(self.config, 'random_embedding_dim', 512)
        self.random_seed = getattr(self.config, 'random_seed', 42)

    def _init_onehot(self):
        """Initialize one-hot embedding components."""
        self.embedding_dim = None

    def build_biomarker_vocab(self, all_biomarkers: list):
        """Build vocabulary and compute embeddings using the specified method."""
        print(f"Building biomarker vocabulary using {self.method} method...")

        # Check for cached embeddings
        cache_path = os.path.join(
            self.config.model_save_path,
            f'biomarker_embeddings_{self.method}.pt'
        )

        if os.path.exists(cache_path) and not getattr(self.config, 'recompute_embeddings', False):
            print(f"Loading cached embeddings from {cache_path}")
            self.load_embeddings(cache_path)
            return

        # Build vocabulary mapping
        for i, biomarker_name in enumerate(all_biomarkers):
            self.biomarker_to_idx[biomarker_name] = i
            self.idx_to_biomarker[i] = biomarker_name

        # Method-specific embedding computation
        if self.method == 'esm':
            self._build_esm_embeddings(all_biomarkers)
        elif self.method == 'gpt':
            self._build_gpt_embeddings(all_biomarkers)
        elif self.method == 'random':
            self._build_random_embeddings(all_biomarkers)
        elif self.method == 'onehot':
            self._build_onehot_embeddings(all_biomarkers)

        # Save computed embeddings
        self.save_embeddings(cache_path)

    def _build_esm_embeddings(self, all_biomarkers: list):
        """Build embeddings using ESM model."""
        biomarker_sequences = self._load_biomarker_sequences()

        with torch.no_grad():
            for biomarker_name in tqdm(all_biomarkers, desc="Computing ESM Embeddings"):
                if biomarker_name not in biomarker_sequences:
                    print(f"Warning: Biomarker '{biomarker_name}' not found in sequence mapping.")
                    # Use fallback for missing sequences
                    embedding = self._generate_fallback_embedding(biomarker_name)
                    self.biomarker_embeddings[biomarker_name] = embedding
                    continue

                sequence = biomarker_sequences[biomarker_name]
                tokens = self.alphabet.encode(sequence)
                tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(self.device)

                results = self.model(tokens, repr_layers=[self.repr_layer], return_contacts=False)
                embedding = results["representations"][self.repr_layer].squeeze(0)[1:-1].mean(0)

                self.biomarker_embeddings[biomarker_name] = embedding.detach().cpu()

    def _build_gpt_embeddings(self, all_biomarkers: list):
        """Build embeddings using Azure OpenAI API via LangChain."""
        system_prompt = f"""You are an expert bioinformatician specializing in spatial omics. Your task if to map a list of biomarkers that are used in some 
        spatial-omics studies to a biologically context-rich vector of length {self.embedding_dim}, which will encompass information related to the nature of this biomarker,
        the cell-type of cells that it marks and feel free to add more. These informations will be further tackeled using deep-learning models so please
        make sure that the embedding yielded is context-rich and self-explainable and consistent. Your response MUST be a single, 
        valid JSON array of floating-point numbers and nothing else. Example format: [0.123, -0.456, 0.789, ...]"""

        try:
            from api import chat
        except ImportError:
            print("ERROR: Could not import 'chat' from api.py.")
            print("Please make sure api.py is in the same directory and is configured correctly.")
            exit()
        except Exception as e:
            print(f"An error occurred while initializing the API: {e}")
            exit()

        for biomarker_name in tqdm(all_biomarkers, desc="Computing Azure OpenAI Embeddings"):
            user_prompt = f"The biomarker to be embedded is {biomarker_name}. Make the output exactly {self.embedding_dim} numbers long.'"
            try:
                res = chat.invoke([
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt)
                ])
                # --- MODIFICATION 2: Use json.loads for parsing ---
                # This is the standard, safe, and robust way to parse JSON strings.
                try:
                    # --- MODIFICATION 3: Adjust the error handling for JSON ---
                    vector_list = json.loads(res.content)

                    # Optional: Add a check to ensure the list is not empty and has the correct type
                    if not isinstance(vector_list, list) or not all(isinstance(n, (int, float)) for n in vector_list):
                        raise ValueError("Parsed data is not a list of numbers.")

                except json.JSONDecodeError as e:
                    print(f"Error: Failed to decode JSON for biomarker '{biomarker_name}'. Response was: {res.content}")
                    # Fallback to random embedding on parsing failure
                    embedding = torch.randn(self.embedding_dim)
                    self.biomarker_embeddings[biomarker_name] = embedding
                    print(f"Using random fallback for {biomarker_name}")
                    continue  # Skip to the next biomarker
                except ValueError as e:
                    print(f"Error: Parsed JSON is not in the expected format for '{biomarker_name}'. {e}")
                    embedding = torch.randn(self.embedding_dim)
                    self.biomarker_embeddings[biomarker_name] = embedding
                    print(f"Using random fallback for {biomarker_name}")
                    continue

                embedding = torch.tensor(vector_list, dtype=torch.float32)
                print(embedding)

                # Optional: Add a dimension check
                if embedding.shape[0] != self.embedding_dim:
                    print(
                        f"Warning: Embedding for '{biomarker_name}' has incorrect dimension {embedding.shape[0]}. Expected {self.embedding_dim}.")
                    # Handle dimension mismatch, e.g., by using fallback
                    embedding = torch.randn(self.embedding_dim)
                    print(f"Using random fallback for {biomarker_name}")

                self.biomarker_embeddings[biomarker_name] = embedding

            except Exception as e:
                print(f"Error getting Azure OpenAI embedding for {biomarker_name}: {e}")
                # Fallback to random embedding on API call failure
                embedding = torch.randn(self.embedding_dim)
                self.biomarker_embeddings[biomarker_name] = embedding
                print(f"Using random fallback for {biomarker_name}")

    def _build_random_embeddings(self, all_biomarkers: list):
        """Build embeddings using random initialization."""
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)

        for biomarker_name in tqdm(all_biomarkers, desc="Computing Random Embeddings"):
            embedding = self._generate_fallback_embedding(biomarker_name)
            self.biomarker_embeddings[biomarker_name] = embedding

    def _build_onehot_embeddings(self, all_biomarkers: list):
        """Build one-hot embeddings using identity matrix."""
        vocab_size = len(all_biomarkers)
        self.embedding_dim = vocab_size

        identity_matrix = torch.eye(vocab_size, dtype=torch.float32)

        for biomarker_name in tqdm(all_biomarkers, desc="Creating One-Hot Embeddings"):
            idx = self.biomarker_to_idx[biomarker_name]
            embedding = identity_matrix[idx]
            self.biomarker_embeddings[biomarker_name] = embedding

        print(f"Created one-hot embeddings: {vocab_size} biomarkers, {vocab_size}-dimensional vectors")

    def forward(self, biomarker_names: list, intensities: torch.Tensor):
        """
        Computes the cell profile vector for a single cell.
        Works the same across all methods.
        """
        # Retrieve pre-computed embeddings and move to device
        embeddings = torch.stack([
            self.biomarker_embeddings[name].to(self.device)
            for name in biomarker_names
        ])

        # Ensure intensities are broadcastable
        if intensities.ndim == 1:
            intensities = intensities.unsqueeze(1)

        # Weighted sum of embeddings
        cell_profile = torch.sum(embeddings * intensities, dim=0)
        return cell_profile

    def _load_biomarker_sequences(self):
        """Load biomarker amino acid sequences (for ESM method)."""
        # Extended sequence database
        sequences = {
            "CD45": "MKLVFLVLLTLLVLSGSSGKEDPKEGHCKEEDEEDPSGFAPVEMDTDDVDEQRKPWPVVWLNGKPAAAIGTREVARNSELVDFIQAFCDAIVTYGFRDAGAAINVFADLFDDCNTQFDTQHLSMDVEIGFVGMNSLQCMGFIPIGRLVPEGSIIMSGATVLWFAGVGLNLLAAICAHRPKCDPNSTPEEPRLYTAASAAPPPAPLPLEAPPQRRNSDVYVGEEDHLSCTFQGSKPNQCTFCTHRDHSQALVQFTRSNSSLQQIGQFTDNVSGLLASLDGKTPGNSVITIRGVGALEKVAYNYSITEDIKSLNSVRVIGVRVKLQNCLCEKYDTTTRIISELNTSDSGMMNLWPNQQTACRMHMPLLVDQQTLFASFGQHIFNHQKMGLVLPVDMLQRDAESGEQKDLKLKEQVQSEFLLFMNVFGTEVRRCNLTPYPCNLGMFKHHCCVHNDCLKYNSKNLQQEEVILNGESLVDGQQYQLVTVAGGRFLQFKEEFDEDLKLNSNVFEQFAGGVSTQLSTTHQVQGNQIQVCQECGGELLQEFKTQQASASNEENLYNIDLPLCYPETQDTIQVDLVFSVGKPQDKLNLTLWQNQDVTEVDPAQRSMTTDMTGAEHPLTRYFVTFGFDDLRYQVQMNNSGPAKYDYPSAYVFEKAKAEMHRKVTAMHKPVDEGLVAYQRGRSVVDLNKGNFDIEESLLRDKLQTPGGTLLSKTDVLLSFGPQIQSNYEVGLNVTEAKDGNEAAEKGKNAEEAAENSPEKQQRQKMQGTIFPRTFQFDHPATTNVDTIAKEKLTVKRLNKYVDSSNVGKDLVLNTVTKTNQASQTVDKVYVGPNFIFLPEKSDNNKMNFSQEEHFHISYKPTLPNSDFSTYLNSYRFEHHLDMLPDEPMAKYGDKGVQRFIYRTGEKETSGDTGKYYLMLPDFKDYFEQFNYKFTDYQKRVQQADPRGPYEKQLDDHFLFDYYDEAEEIVKLAAHEAKPLLDQFNLLNSKLVEGKLRNAQKGIPGKMRSILMDGGTCQIIAADRMPNVNLLLGNSQLRPGDTYCYDMNGGVLKAFGQKPSRDGYEEPKKTMGNTEEQASFNNEAETNVGEEQLTCQPFQRRAQAQDLRGKYAYQGTQLASDRNTTANLDLVSKYLTPFEVLNKDSSLDPLGKLVLRLEKNGQIVTEEHIVVCEGQLNNIDNGIKLFPGDYKAERAEDAQLLRGSYKPRPVTYKSEEETKKDKVVFPEVQSYEFKYQVLPPRLKRSQLHKGYFSITYKRDPFKGNNTRIQFPLGLAAVTNQMPGKFVGSQPPRGNLPGHGMDAKDHVFSTDPRFGFTYDVQKTVQIGDYPDDFVDAYSMGQRLALHAIIFEHAGHRGFLCNGPQDTQFTVLGSNVSVEGSLFAVVPEGSQTARRGEQGYNMGPPNQIKYQIGVGFDHAGEFPPEVQTEAKQFGQRRMLDQFFTTAYDLVFKLDIRKSRKSSLALYKPSEEKTKLALQHKGREALNIQDGQTAAKRRMKTVLHQVAFYLGHAKPPAVIAAQFENYKLEQFDKPSKTNMKVLNRPLTLLYGWLPNGKMWVTDTSSVVLAEGDTNSLRTLVPDNTSVGFWDLQYDQFEQNNLLSFRDVPPLPKDVHKIGQRATEFKYLHSQLLLDAFLQTQDDQKLFVEAHEKQGKIIPKFASMAQGVLTVSGELKKAIAERGTFPRNVTPYFATPPDYQDVGEQGLHLRRSAQSLYLLLSKQGWPPLLNQMYVKRRGSDLLQPVEFDTYFRIQRFSDQASMEAKDGTLMFKQTNVHVQSLFLDLLDNNQSNFTLSQNQEESDYQYKQFTQQVQAMGLKEIQTLQKTSKYKNLLIPLDQGPKEWQIGVVVGSPTVNSQVEQQVQQPVHAEQSYTFSLKQSGFLQQLVKKTKQVHIQQNQEVSEYDKAQVWQKGTFKGLDLTEAEGQGPQGEGSFGVLQTNNGKKGSGPYDILQAQESQMYFEALKGALHEQYPGTGQTYGVGTYNQEKYFYVEAIQNTFPDGQRTITVGGEYFDQISNIQRSAGKQAKMEFNKAVEGVKKGGKTVYQILSYGDYDGDKDGSVADFNAYHTLQKLLCEDGEQGPQGQGSFGVLQTNNGKKGSGPYDILQAQESQMYFEALKGALHEQYPGTGQTYGVGTYNQEKYFYVEAIQNTFPDGQRTITVGGEYFDQISNIQRSAGKQAKMEFNKAVEGVKKGGKTVYQILSYGDYDGDKDGSVADFNAYHTLQKLL",
            "CD3": "MLKCWLCLGLALGSVLGPAQQTDTQMKEELEQVNLPGVVRQITLKSWEDGETSRCNLTGEPLLHELLSQEAYTQVHVRDMSGLYRCNGTDITVLDGEEDSGNLTFDLRPGQCYTITLYKNSDLDLDLERRSDYSFCSLLRDQFTGPEEVRTFPMETVTYAEQEDTMHQSNMQEYTGSKLDMAYWPQLEDGSLQLLNKQGHQPYKYQLKYLGSREDGQLTSSNLESREEVQMYVLQVPQSTLSGGKKHEEELRQGDLDSKDELDDSFDLVHQEDSGKRGSIKSRFDEDGEGLLRTFQQVKYPLSYGFKADTDMPMKHLDFHLQITFRQLTEEGEFRYNAFQDRQRKSHGAFSRLLYGFYKEQRQGQFVNAILFSVTGEQEFQVEGSKEQEVAFPSQNNPEGKTIVPEGLNTRSLTREEDGDYYKGTKQKQFNTYQGSLLVNKNSDDVVIYDQDTYHGGDYFLLCGSRTKQLLRDGDVSELATLITKKNRRSLQPRFESVREILQADTKVLGEEIQPYGFGVHDPEGKYMSRFDVSDRSKRTFRPSYDLLNNSRQVQQDVVFTVFGGTQREYYLRHGHTLNQEFQRVEYEFQNVPEYFRQRYSGDQAKLRQGDEQLNQARKHLLHDNPEVFNTQVGDSKRFRLDDVLFRPRSPQEKFYVDKQIDVDYVSYNLREDKYPQHFRPAEPYGDAEGDGLRGKYSVSNYTYQEPEGGFNRHFDRDSLNKVKDNLNQYKYKFPNVTSPMFYGKRQYMDIQSDDFHLCFLSEIPYQLEIQRRLLKDYDRVQAERSEENLNLFYAYKYKKQQGDLKATYTVKQARRNLNQYGKKYALVGHNLDEPQKPAVQETMRRIVQAAEFAQNRSPSASDFRYDRASMHFDLQLPVTYEQLLGDRYAQSRNNTNLIFLLDYVQYTQGGFLGGAKYYKPTLIKVTKQKLETSSSNFTDKQMPIPQAPADFKQVRTLLQEEPHLMVEPNALGGDFYRLQQDQHHFTHLIQDFQKPEEELPDLPSQLYFDVAYQKDDLRYSGEPKYDGAIVRMGRQLKDFLLQGQQKNYFTDMVSRSGEAKQFDYFSYFRVAGAAGFYFKVFTDNVQRHREFLRRTRQYGQMKVAFPKVVAQADPVEQLNKLKKLFEFNFRRNSDTMMDATNTVGVAHVDHTDDLGLQGLHFHLDYRKSKSQAERTVEIINAYRTDLGEKVMIRGLRFKFSYGQWTKRDGRLRRMTAIAEQRQLFVAKAAHLADVAAEQRRRMAASMARGQHLQDFQQLLLGYDMQRGEQMDLMDDEYQLRQRMLLNYFQFNQKHQDKAANLQIALFKNLDDQEDQDQHDQLQLLQQHQEQFDIMRHQNEQKEHLRLTHLSLQEQFSKEELKEVEGLGLQQDAMLLHESLLQFQEQRVMEEGTGVQRLQRHFRGEQYSMQIHEKLELQGDYVLNQDFKMRLEHKFVESQNQDSTMYRQLLPELQQQLYHTQPELQKLHQQHQNLLEQQRQLLQMKKQPQLALQALLAELQQEKEADNELQGDMTAAQLLQRHQENLLKLQDLAQLLQEHQLQALQLGPRPEGLQALPPAEDYHLRLQDLMQHRLGQEVPDVLLQLQKEQRQELQQLEQKAGLKEEHSRELQQAQEALHRQLQELNQAEELLQGLQKLREQQELQLGLAKLEQSLAEQEEQLQKLQEKVDQAQQQHQQKQEYEEQVEQKLAKLERQLRELMQEEEQLQGLQKLREQQELQLGLAKLEQSLAEQEEQLQKLQEKVDQAQQQHQQ",
            "Ki-67": "MSLKSKQEQHVDQISNVACSQKPGNGTQTSSSPVPPQPQVAPPPSQSSQQQQDSKKHQVQKFTDVKEKQDVSNIQKLTSHPGQGGNTQTSSSPVPPPPPPRQVQQQQDSKKHQVQKFTDVKEKQDVSNIQKLTSHPGQGGNTQTSSSPVPPQPQVAPPPSQSSQQQQDSKKHQVQKFTDVKEKQDVSNIQKLTSHP",
            "Pan-CK": "MSRQSSGGYGGSSYGSGGGSRGGYGGGSYGSGGGSRGGYGGGSYGSGGGSRGGYGGGSYGSGGGSRGGYGGGSYGSGGGSRGGYGGGSYGSGGGSRGGYGGGSYGSGGGSR",
            "DAPI": "MQKQHLDKLMERGQSDDDDKKDDIAKRAEEAKEKTLPALFHGVTAELEKRKVLSEMQINPQHSIVVAKRYLDNTTMVSKFKSKLTKAQVNHAKIVKQPHTLLKFKPQLQKCVAMKNGFTAHGFSGGQSQGGSGGSSVASDLAEQTLYTMNPKIDTKEYQTLLQETDLQRFGLAFNDLDFYSGMETVEHQFVRALAQKYQELTEQQMLLQDKSGGQSQGTQTPAPVKRQVPFLDPSLSRVFPAAPSNDAPPPLPPAPQPQRSQSPQPGPPQCPQYPQGQTSCPQPYPQGQPGQTSCPQPYPQGQPGQTSCPQPYPQGQPGQTSCPQPYPQGQ",
        }
        return sequences

    def get_embedding_dim(self):
        """Returns the dimension of the embeddings."""
        return self.embedding_dim

    def save_embeddings(self, path):
        """Save computed biomarker embeddings and vocabulary."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data_to_save = {
            'method': self.method,
            'embeddings': self.biomarker_embeddings,
            'biomarker_to_idx': self.biomarker_to_idx,
            'idx_to_biomarker': self.idx_to_biomarker,
            'embedding_dim': self.embedding_dim
        }

        torch.save(data_to_save, path)
        print(f"Biomarker embeddings ({self.method}) saved to {path}")

    def load_embeddings(self, path):
        """Load pre-computed biomarker embeddings and vocabulary."""
        data = torch.load(path, map_location='cpu')

        if data.get('method') != self.method:
            print(f"Warning: Loading embeddings from {data.get('method')} method for {self.method} method")

        self.biomarker_embeddings = data['embeddings']
        self.biomarker_to_idx = data['biomarker_to_idx']
        self.idx_to_biomarker = data['idx_to_biomarker']
        self.embedding_dim = data['embedding_dim']

        print(f"Successfully loaded {len(self.biomarker_embeddings)} cached embeddings ({self.method}).")