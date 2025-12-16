
from dataclasses import dataclass
import os
import json
from dataclasses import asdict
import typing
import torch

from common_ml.tags import VideoTag

from src.model import EuroSTT
from src.audio import audio_file_to_tensor
from src.tags import AugmentedTag


@dataclass(frozen=True)
class RuntimeConfig:
    pretty_trail: bool
    pretty_trail_buffer: int


class AudioBuffer:
    """Accumulates audio tensors and metadata for trailing buffer processing"""
    
    def __init__(self):
        self.tensors: list[torch.Tensor] = []
        self.filenames: list[str] = []
        self.total_duration: float = 0.0
    
    def add(self, tensor: torch.Tensor, filename: str, duration: float):
        """Add audio to buffer"""
        self.tensors.append(tensor)
        self.filenames.append(filename)
        self.total_duration += duration
    
    def get_combined_tensor(self) -> torch.Tensor:
        """Concatenate all tensors"""
        print([t.shape for t in self.tensors])
        return torch.cat(self.tensors)
    
    def get_first_filename(self) -> str:
        """Get filename of first buffered audio"""
        return self.filenames[0] if self.filenames else ""
    
    def clear(self):
        """Clear the buffer"""
        self.tensors = []
        self.filenames = []
        self.total_duration = 0.0
    
    def is_ready(self, threshold: float) -> bool:
        """Check if buffer has reached threshold duration"""
        return self.total_duration >= threshold
    
    def is_empty(self) -> bool:
        return len(self.tensors) == 0


class SpeechTagger:
    """Orchestrates audio loading, STT tagging, prettification, and file writing"""
    
    def __init__(self, cfg: RuntimeConfig, tags_out: str):
        self.cfg = cfg
        self.tags_out = tags_out
        self.model = EuroSTT()
        
        # Initialize buffer for pretty_trail feature
        self.buffer = AudioBuffer() if cfg.pretty_trail else None
    
    def tag(self, fname: str) -> None:
        """
        Process a single audio file and write tag files
        
        Args:
            fname: Path to audio file
        """
        # Load audio file to tensor
        audio_tensor, duration = audio_file_to_tensor(fname)
        
        print(audio_tensor.shape)
        # Generate raw word-level tags
        tags = self.model.tag(audio_tensor)
        
        # Write primary output only if we have tags
        if len(tags) > 0:
            self._write_tags(fname, tags, suffix="_tags.json")
         
        # Always add to trailing buffer if enabled (even if tags is empty)
        if self.cfg.pretty_trail:
            self._process_trailing_buffer(audio_tensor, fname, duration)

    
    def _process_trailing_buffer(self, audio_tensor: torch.Tensor, fname: str, duration: float):
        """Handle accumulation and processing of trailing buffer"""
        # Add to buffer
        assert self.buffer is not None
        self.buffer.add(audio_tensor, fname, duration)
        
        # Check if buffer is ready to process
        print(self.buffer.total_duration)
        if self.buffer.is_ready(self.cfg.pretty_trail_buffer):
            self._emit_prettified_trail()
    
    def _emit_prettified_trail(self):
        """Process accumulated buffer and emit prettified sentence-level tags"""
        assert self.buffer is not None
        if self.buffer.is_empty():
            return
        
        # Get combined audio
        combined_tensor = self.buffer.get_combined_tensor()
        first_fname = self.buffer.get_first_filename()
        
        # Run STT on combined audio
        tags = self.model.tag(combined_tensor)
        
        if len(tags) == 0:
            self.buffer.clear()
            return
        
        # Merge into sentence-level tags
        sentence_tags = self._merge_to_sentences(tags)
        
        augmented_tags = self._add_augmented_fields(sentence_tags, first_fname)
        
        # Write output
        self._write_tags(first_fname, augmented_tags, suffix="-prettified_tags.json")
        
        # Clear buffer
        self.buffer.clear()

    def _add_augmented_fields(self, tags: list[VideoTag], fname: str) -> list[AugmentedTag]:
        """Add augmented fields to tags"""
        return [
            AugmentedTag(
                start_time=tag.start_time,
                end_time=tag.end_time,
                text=tag.text,
                source_media=os.path.basename(fname),
                track="auto_captions"
            )
            for tag in tags
        ]
    
    def _merge_to_sentences(self, tags: list[VideoTag]) -> list[VideoTag]:
        """
        Merge word-level tags into sentence-level tags based on punctuation
        
        Args:
            tags: list of word-level tags (with punctuation from prettifier)
        
        Returns:
            list of sentence-level VideoTags
        """
        if len(tags) == 0:
            return []
        
        sentence_delimiters = {'.', '?', '!'}
        sentences = []
        current_words = []
        current_start = tags[0].start_time
        
        for i, tag in enumerate(tags):
            current_words.append(tag.text)
            
            # Check if this word ends with sentence delimiter
            if any(tag.text.endswith(delim) for delim in sentence_delimiters):
                # Create sentence tag
                sentence_tag = VideoTag(
                    start_time=current_start,
                    end_time=tag.end_time,
                    text=' '.join(current_words),
                )
                sentences.append(sentence_tag)
                
                # Start new sentence
                current_words = []
                if i + 1 < len(tags):
                    current_start = tags[i + 1].start_time
        
        # Handle remaining words (if no sentence delimiter at end)
        if current_words:
            sentence_tag = VideoTag(
                start_time=current_start,
                end_time=tags[-1].end_time,
                text=' '.join(current_words)
            )
            sentences.append(sentence_tag)
        
        return sentences
    
    def finalize(self):
        """Process any remaining buffered audio"""
        if self.cfg.pretty_trail and self.buffer and not self.buffer.is_empty():
            self._emit_prettified_trail()
    
    def _write_tags(self, fname: str, tags: typing.Union[list[VideoTag], list[AugmentedTag]], suffix: str) -> None:
        """Write tags to JSON file"""
        output_path = os.path.join(
            self.tags_out, 
            f"{os.path.basename(fname)}{suffix}"
        )
        with open(output_path, 'w') as fout:
            fout.write(json.dumps([asdict(tag) for tag in tags]))