from Bio import Entrez
from Bio import SeqIO


def get_human_protein_sequence(protein_name):
    """
    Fetches the amino acid sequence for a given human protein name from NCBI.

    This function is pre-configured with the following fixed values:
    - Organism: Homo sapiens
    - User Email: aguo0521@connect.hku.hk

    Args:
        protein_name (str): The name or symbol of the protein to search for (e.g., "CD45", "BRCA1").

    Returns:
        Bio.SeqRecord.SeqRecord: A SeqRecord object from Biopython containing the
                                 protein's data, or None if the protein is not found.
    """
    # Hardcoded values as per your request
    email = "aguo0521@connect.hku.hk"
    organism = "Homo sapiens"

    # Set the email for all Entrez requests
    Entrez.email = email

    # Construct a precise search query to find a specific human protein.
    # It searches for the protein name within the "Protein Name" field,
    # filters by organism, and prioritizes high-quality RefSeq entries.
    search_query = f'({protein_name}[Protein Name]) AND "{organism}"[Organism] AND refseq[filter]'

    print(f"Searching NCBI for human protein: '{protein_name}'...")

    try:
        # Step 1: Search for the protein's unique ID using the constructed query.
        # We ask for only the top result (retmax="1").
        handle = Entrez.esearch(db="protein", term=search_query, retmax="1")
        record = Entrez.read(handle)
        handle.close()

        id_list = record["IdList"]
        if not id_list:
            print(f"-> Search complete. No RefSeq result was found for '{protein_name}' in Homo sapiens.")
            print(
                "-> Tip: The name might be an alias or gene name. Check the official protein name on NCBI or UniProt.")
            return None

        protein_id = id_list[0]
        print(f"-> Found a matching protein with ID: {protein_id}")

        # Step 2: Use the found ID to fetch the full protein record in FASTA format.
        print(f"-> Fetching the full sequence for ID {protein_id}...")
        fetch_handle = Entrez.efetch(db="protein", id=protein_id, rettype="fasta", retmode="text")

        # Parse the FASTA record into a Biopython SeqRecord object
        seq_record = SeqIO.read(fetch_handle, "fasta")
        fetch_handle.close()

        return seq_record

    except Exception as e:
        print(f"An unexpected error occurred during the NCBI search: {e}")
        return None


# --- Example of How to Use the Function ---

# Define the name of the protein you want to find.
# You can change this value to any other human protein.
target_protein_name = "CD45"

# Call the function. Notice you only need to provide the protein name.
protein_record = get_human_protein_sequence(target_protein_name)

# If the function successfully returns a protein record, print its details.
if protein_record:
    print("\n--- Protein Sequence Retrieved Successfully ---")
    print(f"Accession ID: {protein_record.id}")
    print(f"Full Description: {protein_record.description}")
    print(f"Sequence Length: {len(protein_record.seq)} amino acids")
    print(f"Amino Acid Sequence:\n{protein_record.seq}")

    # You can also automatically save the sequence to a FASTA file
    # We clean the ID to make a valid filename
    clean_id = protein_record.id.split('|')[1]
    file_name = f"{clean_id}_{target_protein_name}.fasta"

    with open(file_name, "w") as output_file:
        SeqIO.write(protein_record, output_file, "fasta")
    print(f"\nSequence has been saved to the file: '{file_name}'")