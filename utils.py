
from tqdm.auto import tqdm
from dataclasses import dataclass
import os
import torch 
from torch.nn.utils.rnn import pad_sequence 
from transformers import Trainer

tqdm.pandas()
@dataclass
class ByteDatasetConfig:
    """
    Configuration class for ByteDataset.
    
    Attributes:
        data_dir (str): Directory where the data files are stored.
        window_size (int): Size of the text window to process in each dataset item, with a default of 8192.
        stx_token (bytes): Start of Text token as a byte, defaulting to the STX ASCII control character.
        etx_token (bytes): End of Text token as a byte, defaulting to the ETX ASCII control character.
    """

    filepath: str
    window_size: int = 8092
    stx_token: bytes = b'\x02' # STX ASCII control character (Start of TeXt)
    etx_token: bytes = b'\x03' # ETX ASCII control character (End of TeXt)

class ByteDataset:

    def __init__(self, config):
        """
        Initialize the ByteDataset with a given configuration.
        """
        self.config = config
        self.data = []
        self.chunks = []

        with open(config.filepath, "rb") as file:
            text = file.read()
            chunks = self.calc_chunks(text)
            if chunks > 0:
                self.data.append(text)
                self.chunks.append(chunks)

    def calc_chunks(self, text):
        """
        Calculate the number of chunks needed for a given text.
        """
        if len(text) == 0:
            return -1
        delimiters = len(self.config.stx_token) + len(self.config.etx_token)
        length = len(text) + delimiters
        chunks = length // self.config.window_size
        rest_bytes = length % self.config.window_size
        if rest_bytes > 0:
            chunks += 1
        return chunks


    def __len__(self):
        """
        Returns the total number of chunks available in the dataset.
        """
        return sum(self.chunks)
    
    def map_idx(self, i):
        """
        Maps a sequential index to a specific chunk in the dataset.
        """
        accum_chunks = 0
        for idx, num_chunks in enumerate(self.chunks):
            if accum_chunks + num_chunks > i:
                return idx, i - accum_chunks
            accum_chunks += num_chunks
        raise ValueError(f"Index {i} out of range")
    
    def __getitem__(self, i):
        """
        Retrieves the data for a specific chunk.
        """
        idx, offset = self.map_idx(i)
        start = offset * self.config.window_size
        end = start + self.config.window_size

        if offset == 0:
            input_ids = [b for b in self.config.stx_token]
            input_ids += [b for b in self.data[idx][start:end-len(self.config.etx_token)]]
        else:
            input_ids = [b for b in self.data[idx][start-len(self.config.stx_token):end]]

        # Fixed indentation issue
        if (offset + 1) * self.config.window_size >= len(self.data[idx]):
            input_ids += [b for b in self.config.etx_token]

        input_ids_tensor = torch.tensor(input_ids[:self.config.window_size], dtype=torch.int)
       # Pad sequence to desired length.
        padded_input_ids = torch.cat([input_ids_tensor, torch.zeros(self.config.window_size - len(input_ids_tensor), dtype=torch.int)])

       # Shift labels by one position for language model training
        labels_tensor = torch.cat([padded_input_ids[1:],torch.tensor([-100], dtype=torch.int)])
        padded_labels = torch.cat([labels_tensor, torch.zeros(self.config.window_size - len(labels_tensor), dtype=torch.int)])
        return padded_input_ids, padded_labels

       
class ByteDatasetCollator(object):
    def __call__(self, features):
        input_ids = [feature[0] for feature in features]
        labels = [feature[1] for feature in features]

        # Pad the sequences
        input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=0)
        labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100)
       
        # Create attention_mask based on input_ids not being equal to 0
        attention_mask = input_ids_padded.ne(0).int()

        return {
            'input_ids': input_ids_padded,
            'attention_mask': attention_mask,
            'labels': labels_padded,
        }


class MambaTrainer(Trainer):
    """
    Custom trainer class inheriting from Hugging Face's Trainer, with modifications specific to the ByteDataset training regime.
    
    Overrides:
        compute_loss: Custom loss computation method.
        save_model: Custom model saving method.
    """

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Compute the loss for a batch of inputs.
        
        Parameters:
            model (torch.nn.Module): The model being trained.
            inputs (dict): Batch of inputs.
            return_outputs (bool): Flag to return model outputs along with loss.
        
        Returns:
            torch.Tensor: Computed loss.
        """
        input_ids = inputs.pop("input_ids")
        lm_logits = model(input_ids).logits

        labels = input_ids.to(lm_logits.device)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        labels = labels[:, 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss()
        lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), labels.view(-1))

        return lm_loss

    def save_model(self, output_dir, _internal_call=None):
        """
        Save the model to a specified directory.
        
        Parameters:
            output_dir (str): Directory to save the model.
            _internal_call (bool): Internal flag, not used here.
            model_name (str): Name you want to use to save your model as
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        torch.save(self.model.state_dict(), f"{output_dir}/mambabyte.bin")