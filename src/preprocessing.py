import re
import pandas as pd
import nltk
from nltk.corpus import words
from tokenizers import Tokenizer, models, pre_tokenizers, trainers


class PreModelPipeline:
    """
    Routes incoming Roman text through a strict 4-step pipeline
    before it touches the neural transliteration model.
    """

    def __init__(self, corpus_paths=None, vocab_size=5000):
        """
        Args:
            corpus_paths: List of CSV file paths to build dictionary and tokenizer.
                          Pass None to create a lightweight instance (tokenizer loaded separately).
            vocab_size: Target vocabulary size for BPE tokenizer.
        """
        try:
            nltk.data.find("corpora/words")
        except LookupError:
            nltk.download("words", quiet=True)

        self.english_dict = set(word.lower() for word in words.words())
        self.english_dict.update([
            "wifi", "laptop", "phone", "charge", "internet", "app",
            "browser", "computer", "email", "google", "download",
            "upload", "software", "hardware", "website", "online",
        ])

        self.fast_lookup = {}
        self.tokenizer = None
        if corpus_paths is None:
            return

        self._build_pipeline(corpus_paths, vocab_size)

    def normalize_roman(self, text):
        """
        STEP 2: Roman Normalization Layer.
        Compresses chaotic user spelling to reduce the model's vocabulary.
        Examples:
            'aaa' -> 'a', 'ee' -> 'i', 'oo' -> 'u', 'ph' -> 'f'
        Args:
            text: Raw Roman text input.
        Returns:
            Normalized lowercase string with only alphabetic characters.
        """
        word = str(text).lower().strip()

        word = re.sub(r"a{2,}", "a", word)   
        word = re.sub(r"e{2,}", "i", word)  
        word = re.sub(r"o{2,}", "u", word)   

        word = re.sub(r"ph", "f", word)       

        word = re.sub(r"[^a-z]", "", word)
        return word

    def _build_pipeline(self, corpus_paths, vocab_size):
        """
        Build the dictionary backoff and BPE tokenizer from training data.
        Args:
            corpus_paths: List of CSV file paths containing 'roman' and 'native' columns.
            vocab_size: Target BPE vocabulary size.
        """
        print("1. Loading training datasets to build Phase 2 modules...")
        dfs = []
        for path in corpus_paths:
            try:
                dfs.append(pd.read_csv(path))
            except FileNotFoundError:
                print(f"Warning: {path} not found. Skipping.")

        if not dfs:
            print("CRITICAL ERROR: No data found to build dictionary and tokenizer.")
            return

        df = pd.concat(dfs, ignore_index=True)

        print("2. Normalizing corpus to build Dictionary Backoff...")
        df["roman_norm"] = df["roman"].apply(self.normalize_roman)

        # STEP 3: Dictionary Backoff
        if "freq" in df.columns:
            df_freq = df.groupby(["roman_norm", "native"], as_index=False)["freq"].sum()
        else:
            df_freq = df.groupby(["roman_norm", "native"]).size().reset_index(name="freq")

        df_freq = df_freq.sort_values("freq", ascending=False)

        top_50k = df_freq.drop_duplicates(subset=["roman_norm"], keep="first").head(50000)
        self.fast_lookup = top_50k.set_index("roman_norm")["native"].to_dict()
        print(f"-> Dictionary Backoff built: {len(self.fast_lookup)} Top-K words loaded.")

        # STEP 4: Subword Tokenization (BPE)
        print("3. Training Byte-Pair Encoding (BPE) Tokenizer...")
        self.tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
        self.tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]"],
        )

        normalized_texts = df["roman_norm"].dropna().tolist()
        self.tokenizer.train_from_iterator(normalized_texts, trainer=trainer)
        print(f"-> Subword Tokenizer trained. Vocab size: {self.tokenizer.get_vocab_size()}")

    def is_english(self, word):
        """
        STEP 1: Language Detection.
        Check if the word is English and should bypass transliteration.
        Args:
            word: Lowercase word string.
        Returns:
            True if the word is English (length > 3 and in dictionary).
        """
        return word.lower() in self.english_dict and len(word) > 3

    def process_word(self, word):
        """
        Route a single incoming word through the strict 4-step logic flow.
        Args:
            word: Raw input word.
        Returns:
            Dictionary with keys: word, route, output, tokens.
        """
        original_word = str(word).strip()
        clean_word = original_word.lower()

        if self.is_english(clean_word):
            return {
                "word": original_word,
                "route": "English Bypass",
                "output": original_word,
                "tokens": None,
            }

        norm_word = self.normalize_roman(clean_word)

        if norm_word in self.fast_lookup:
            return {
                "word": original_word,
                "route": "Dictionary",
                "output": self.fast_lookup[norm_word],
                "tokens": None,
            }

        if self.tokenizer:
            encoded = self.tokenizer.encode(norm_word)
            return {
                "word": original_word,
                "route": "Neural Model",
                "output": "-> [PENDING INFERENCE]",
                "tokens": encoded.tokens,
            }

        return {"word": original_word, "route": "Error", "output": None, "tokens": None}

    def process_sentence(self, sentence):
        """
        Process a full sentence, evaluating word-by-word.
        Args:
            sentence: Full input sentence string.
        Returns:
            List of process_word result dictionaries.
        """
        words_in_sentence = sentence.split()
        return [self.process_word(w) for w in words_in_sentence]
