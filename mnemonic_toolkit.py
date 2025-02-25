#!/bin/env python3
"""
TODO:
- Do something to highlight duplicates. At least warn of their presence.
- it feels like option 4 should go after 6
- 4. and 
- 
"""

from bip32utils import BIP32Key
from bitcoinlib.wallets import Wallet
from collections import Counter, defaultdict
from colorama import Fore, Back, Style
from mnemonic import Mnemonic
from typing import List
import binascii
import colorama
import difflib
import os
import re
import readline
import subprocess
import sys
import tempfile
import textwrap
import datetime

# Initialize colorama (especially important on Windows)
colorama.init()


VALID_BIP39_LENGTHS = {12, 15, 18, 21, 24}
mnemo = Mnemonic("english")

def words_to_prefixes(words: List[str]) -> List[str]:
    """Shorten list of seed words to prefixes using get_prefix()."""
    return [get_prefix(word.lower()) for word in words if get_prefix(word.lower())]

def reconstruct_mnemonic_from_prefixes(words: List[str], wordlist: List[str]) -> str:
    """
    Reconstructs BIP-39 mnemonic from a list of prefixes or words using BIP-39 rules.
    
    Args:
        words (List[str]): List of potential BIP-39 prefixes or words.
        wordlist (List[str]): The BIP-39 wordlist for matching.
        
    Returns:
        str: Space-separated mnemonic string.
    """
    full_words = []

    for word in words:
        prefix = get_prefix(word)  # Get the 3-letter whole word or 4-letter prefix
        if prefix:
            if len(word) == 3:
                # 3-letter words must match exactly
                if prefix in wordlist:
                    full_words.append(prefix)
            else:
                # 4+ letter prefixes must match as the start of a word
                matches = [bip39_word for bip39_word in wordlist if bip39_word.startswith(prefix)]
                if len(matches) == 1:
                    full_words.append(matches[0])  # Append the unique match

    # Return the reconstructed mnemonic as a space-separated string
    return " ".join(full_words)

def get_prefix(word):
    """Returns the first 4 characters of a word if it meets the minimum length requirement.
       Returns the full word if it is exactly 3 letters.
       Returns None for words shorter than 3 letters."""
    if len(word) == 3:
        return word.lower()  # Return 3-letter word as is
    elif len(word) >= 4:
        return word[:4].lower()  # Return first `length` characters
    return None  # Ignore 1-2 letter words

#def prompt_seed_phrase(seed_phrase=None):
#    """Prompt for the seed phrase, allowing editing of the default suggestion."""
#    if seed_phrase is None:
#        seed_phrase = input(Fore.RED + "Enter your known-good mnemonic seed phrase: " + Style.RESET_ALL).strip()
#    else: 
#        print(f"Press Enter to use the default seed phrase, or edit it below:")
#        user_input = input(Fore.RED + f"Seed phrase [{seed_phrase}]: " + Style.RESET_ALL).strip()
#        if user_input:  # If user edits, use the new input
#            seed_phrase = user_input
#
#    # Remove arrows and extra spaces before returning
#    seed_phrase = seed_phrase.replace("→", "").replace("  ", " ").replace(">", "").replace("[", "").replace("]", "").strip()
#
#    return seed_phrase

import readline
from colorama import Fore, Style

def prompt_seed_phrase(seed_phrase=None):
    """
    Prompt for the seed phrase, allowing editing of the default suggestion.
    
    Args:
        seed_phrase (str): The default seed phrase to prepopulate the input.
    
    Returns:
        str: The user-provided or edited seed phrase.
    """
    if seed_phrase:
        print(f"Press Enter to use the default seed phrase, or edit it below:")
    else:
        seed_phrase = ""

    # Set up readline to prepopulate the input
    readline.set_startup_hook(lambda: readline.insert_text(seed_phrase))
    try:
        # Prompt user for input
        user_input = input(Fore.RED + "Seed phrase: " + Style.RESET_ALL).strip()
    finally:
        readline.set_startup_hook()  # Ensure the hook is cleared
    
    # If the user provides input, use it; otherwise, use the default
    if user_input:
        seed_phrase = user_input

    # Clean up the seed phrase by removing unwanted characters
    seed_phrase = seed_phrase.replace("→", "").replace("  ", " ").replace(">", "").replace("[", "").replace("]", "").strip()

    return seed_phrase

def story_to_private_key(story_text):
    mnemo = Mnemonic("english")
    wordlist = mnemo.wordlist
    words = story_text.strip().split()
    mnemonic_phrase = reconstruct_mnemonic_from_prefixes(words, wordlist)

    print(f"\nExtracted mnemonic keywords: {mnemonic_phrase}")

    mnemonic_words = mnemonic_phrase.split()

    if is_valid_mnemonic_length(mnemonic_words):
        #if validate_seed_phrase(mnemonic_phrase):
        #    print("\n### Mnemonic Recovery Phrase")
        #    print(f"{mnemonic_phrase}")
        #    print("\n### Seed Phrase Validation")
        #    print("🎉 The seed phrase is valid!")
        #    print("\n### Wallet Setup Guidance")
        #    print("When restoring your wallet, select Native SegWit (P2WPKH) for best performance and lower fees.")
        #else:
        #    print("\n❌ Invalid seed phrase. Please check your input and try again.")
        display_mnemonic_details(mnemonic_phrase)
    else:
        print(f"\n❌ Invalid mnemonic phrase. Found {len(mnemonic_words)} words. Please ensure your message has a valid BIP-39 length (12, 15, 18, 21, or 24 words).\n")

def suggest_replacements_for_unexpected_words(story_text, unexpected_words):
    prompt = []
    if unexpected_words:
        # Split story into individual words
        story_words = re.findall(r'\b\w+\b', story_text)

        # Create a placeholder map of full words matching the prefixes
        placeholder_map = {}
        for prefix in unexpected_words:
            # Find words in the story that start with the prefix
            matched_words = [word for word in story_words if re.fullmatch(rf'{prefix}[a-z]*', word.lower())]
            if matched_words:
                full_word = matched_words[0]  # Take the first match
                placeholder_map[full_word] = f"[{full_word}]"

        # Replace words in the story with placeholders
        story_with_placeholders = story_text
        for full_word, placeholder in placeholder_map.items():
            story_with_placeholders = re.sub(rf'\b{re.escape(full_word)}\b', placeholder, story_with_placeholders)

        # Construct the prompt with improved clarity and instructions
        prompt.append(f"""\
Please suggest alternative replacements for the placeholder words in the following message:

> {story_with_placeholders}

Instructions:
1. Replace each word in brackets `[...]` with a rare or uncommon term that retains the sentence's meaning.
2. Keep replacements concise and avoid overly technical terms.
3. Ensure the flow of the rewritten sentence remains natural.

Placeholder words to replace:""")
        for full_word in placeholder_map:
            prefix = get_prefix(full_word)
            if prefix:
                prompt.append(f"- {full_word} ({prefix})")
            else:
                prompt.append(f"- {full_word}")

        prompt.append("\nOnce you've selected replacements, rewrite the message incorporating your chosen words.")

    else:
        prompt.append("✅ No unintended mnemonic keywords found!")
    return '\n'.join(prompt)

def begin_prompt():
    #return("----- Begin AI Prompt -----" + Fore.YELLOW + Style.BRIGHT + Back.BLUE)
    return(Fore.YELLOW + Style.BRIGHT + Back.BLUE)

def end_prompt():
    #return(Style.RESET_ALL + "\n----- End AI Prompt -----")
    return(Style.RESET_ALL)

#def get_story_text():
#    now = datetime.datetime.now()
#    time_string = now.strftime("%H:%M")
#    print(Fore.RED + f"({time_string}) Enter text or q to quit (finish with Ctrl+D):" + Style.RESET_ALL)
#    story_text = sys.stdin.read().strip()
#    # Define a pattern for characters and sequences you want to remove
#    pattern = re.compile(r'[→>\[\]\(\)\*0-9~]')
#    
#    # Use sub() method to replace the unwanted characters with an empty string
#    story_text = pattern.sub('', story_text).strip()
#
#    return story_text

def get_story_text(default_text=""):
    """
    Prompts the user to enter or edit story text using readline for enhanced input capabilities.
    
    Args:
        default_text (str): Optional text to prepopulate the input field.
    
    Returns:
        str: The sanitized story text.
    """
    now = datetime.datetime.now()
    time_string = now.strftime("%H:%M")
    print(Fore.RED + f"({time_string}) Enter text or q to quit:" + Style.RESET_ALL)

    # Prepopulate the input with default text
    readline.set_startup_hook(lambda: readline.insert_text(default_text))
    try:
        story_text = input("Your story: ")
    except EOFError:  # Handle Ctrl+D gracefully
        print("\nInput aborted.")
        return ""
    finally:
        readline.set_startup_hook()  # Clear the hook to avoid affecting subsequent inputs

    # Check for quit command
    if story_text.lower().strip() == "q":
        print("Quitting input...")
        return ""

    # Sanitize the input to remove unwanted characters
    pattern = re.compile(r'[→>\[\]\(\)\*0-9~]')
    story_text = pattern.sub('', story_text).strip()

    return story_text

def is_valid_mnemonic_length(words):
    return len(words) in VALID_BIP39_LENGTHS

def keyword_prefix_completions(text, wordlist):
    """
    Identify words in `text` that correspond to any words in `wordlist` via:
      1) If the word is 3 letters, it must appear *exactly* (case-insensitive) in `wordlist`.
      2) If the word is 4+ letters, the first 4 letters must match the first 4 letters
         of some word in `wordlist` (case-insensitive), and the remainder must be all letters.
      3) Ignore 1-2 letter words, or 3-letter words not in `wordlist`.
    """

    # Split text into individual alphanumeric words
    story_words = re.findall(r'\b\w+\b', text)

    # Build sets of 4-letter prefixes and complete wordlist for quick lookup
    seed_prefixes = {get_prefix(w) for w in wordlist if len(w) >= 4}
    wordlist_lower = {w.lower() for w in wordlist}  # for exact 3-letter matching

    matched_words = []
    for word in story_words:
        prefix = get_prefix(word)  # Use get_prefix to determine relevant prefixes

        if prefix:
            if len(prefix) == 3 and prefix in wordlist_lower:
                # Exact match for 3-letter words
                matched_words.append(word)
            elif len(prefix) == 4 and prefix in seed_prefixes:
                # Prefix match for 4+ letter words
                remainder = word[len(prefix):]
                if remainder == "" or remainder.isalpha():
                    matched_words.append(word)

    return matched_words

def split_on_seed_word_prefixes(story_text, seed_words):
    """
    Splits the story text into lines based on ordered seed word prefixes.
    Combines words and punctuation that follow a prefix until the next prefix is found.
    """
    # Step 1: Collapse all whitespace into single spaces
    story_text = " ".join(story_text.split())
    
    # Step 2: Create a list of prefixes in the correct order
    seed_prefixes = [get_prefix(sw) for sw in seed_words]
    
    # Step 3: Split story text into tokens that preserve punctuation
    tokens = re.findall(r'\w+|[^\w\s]', story_text)  # Matches words and punctuation
    
    lines = []
    current_line = []
    prefix_index = 0

    for token in tokens:
        stripped = token.lower()
        current_prefix = seed_prefixes[prefix_index] if prefix_index < len(seed_prefixes) else None

        if current_prefix and stripped.startswith(current_prefix):
            # Flush the current line if not empty
            if current_line:
                lines.append("".join(current_line))
                current_line = []

            # Add the seed word as part of the new line
            current_line.append(token)
            prefix_index += 1  # Move to the next prefix
        else:
            # Accumulate non-seed words and punctuation
            if re.match(r'[^\w\s]', token):
                # If token is punctuation, append it without a space
                current_line[-1] += token
            else:
                # If it's a word, add a space before appending
                current_line.append(" " + token)

    # Flush any remaining non-seed words at the end
    if current_line:
        lines.append("".join(current_line))

    # Join the lines into the final result
    return "\n".join(lines)

def extract_all_mnemonic_prefixes(story_text, wordlist):
    story_words = re.findall(r'\b\w+\b', story_text.lower())
    matched_prefixes = []

    for word in story_words:
        prefix = get_prefix(word)  # returns 3 letters or 4-letter prefix
        if prefix is None:
            continue
        # Check if we have exactly 3 letters and it's in wordlist
        if len(prefix) == 3 and prefix in wordlist:
            matched_prefixes.append(prefix)
        # Or if we have 4+ letters, see if any BIP-39 word starts with that prefix
        elif len(prefix) == 4:
            matches = [w for w in wordlist if w.startswith(prefix)]
            if matches:
                matched_prefixes.append(prefix)

    return matched_prefixes

def get_full_word_from_prefix(prefix, wordlist):
    """
    Returns the full word from the wordlist that matches the prefix.
    
    Args:
        prefix (str): The prefix to match (3 or 4+ letters).
        wordlist (List[str]): The BIP-39 wordlist.
        
    Returns:
        str: The full word if found, otherwise None.
    """
    prefix = prefix.lower()  # Ensure lowercase comparison
    
    if len(prefix) == 3:
        # Exact match for 3-letter words
        if prefix in wordlist:
            return prefix
    elif len(prefix) >= 4:
        # Prefix match for 4+ letter words
        matches = [word for word in wordlist if word.startswith(prefix)]
        if matches:
            return matches[0]  # Return the first matching word
    
    return None  # No match found

def is_subsequence_in_order(required_sequence, actual_sequence):
    """
    Returns True if all items in required_sequence appear in actual_sequence
    in the same relative order (not necessarily consecutively).
    """
    it = iter(actual_sequence)
    return all(item in it for item in required_sequence)
            
def adjust_story_text(story_text, seed_words, last_mnemonic_keyword):
    # Check the length of seed_words
    if len(seed_words) in [12, 15, 18, 21, 24]:
        # Replace the last occurrence of the last mnemonic keyword
        if not last_mnemonic_keyword:
            return story_text  # No match, return the original story text

        if len(last_mnemonic_keyword) == 3:
            # Match the whole word for 3-letter keywords and everything after it
            pattern = rf'\b{re.escape(last_mnemonic_keyword)}\b(?!.*\b{re.escape(last_mnemonic_keyword)}\b).*'
        else:
            # Match the first 4 letters and everything after it
            pattern = rf'\b{re.escape(last_mnemonic_keyword[:4])}\w*\b(?!.*\b{re.escape(last_mnemonic_keyword[:4])}\w*\b).*'

        # Replace the last matched keyword and everything after it with "___"
        modified_story_text = re.sub(
            pattern,
            "",
            story_text,
            count=1
        )
        return modified_story_text
    elif len(seed_words) in [11, 14, 17, 20, 23]:
        if not last_mnemonic_keyword:
            return story_text  # No match, return the original story text

        # Match the last occurrence of the keyword and everything after it
        pattern = rf'(.*\b{re.escape(last_mnemonic_keyword[:4])}\w*)\b.*'
        modified_story_text = re.sub(
            pattern,
            r'\1',
            story_text,
            count=1
        )
        return modified_story_text.strip()
    else:
        # If the length doesn't match any expected value, return the story text as is
        return story_text

def validate_seed_phrase(seed_phrase: str) -> bool:
    """
    Validates a BIP-39 mnemonic seed phrase.
    Returns True if valid, False otherwise.
    """
    mnemo = Mnemonic("english")
    return mnemo.check(seed_phrase)

#def format_story_with_numbers(story_text, seed_words):
#    """
#    Formats the story text by marking each mnemonic keyword with its position in the sequence.
#
#    Parameters:
#    - story_text (str): The provided story text.
#    - seed_words (list): List of mnemonic keywords in order.
#
#    Returns:
#    - str: The story text with keywords marked with numbers.
#    """
#    import re
#
#    # Start with the original story text
#    formatted_text = story_text
#    used_indices = set()
#
#    for idx, word in enumerate(seed_words, start=1):
#        # Use regex to match the first occurrence of the keyword
#        # \b ensures we only match whole words
#        match = re.search(rf'\b{re.escape(word)}', formatted_text, re.IGNORECASE)
#        if match and idx not in used_indices:
#            # Insert the index marker before the word
#            start, end = match.span()
#            formatted_text = f"{formatted_text[:start]}[{idx}]{formatted_text[start:]}"
#            used_indices.add(idx)
#
#    return formatted_text

def format_story_with_numbers(story_text, seed_words):
    """
    Formats the story text by marking each mnemonic keyword with its position in the sequence,
    even if the same keyword appears multiple times.

    Parameters:
    - story_text (str): The provided story text.
    - seed_words (list): List of mnemonic keywords in order.

    Returns:
    - str: The story text with keywords marked with numbers.
    """
    import re

    # Start with the original story text
    formatted_text = story_text
    word_counts = {}

    for idx, word in enumerate(seed_words, start=1):
        # Track occurrences of each word to handle duplicates
        word_counts[word] = word_counts.get(word, 0) + 1

        # Use regex to match the nth occurrence of the word
        pattern = rf'(\b{re.escape(word)})'
        matches = list(re.finditer(pattern, formatted_text, re.IGNORECASE))

        if len(matches) >= word_counts[word]:
            # Find the match corresponding to the current occurrence
            match = matches[word_counts[word] - 1]
            start, end = match.span()
            formatted_text = f"{formatted_text[:start]}[{idx}]{formatted_text[start:]}"
    
    return formatted_text

def numbered_list(seed_prefixes):
    """
    Formats a list of mnemonic prefixes into a numbered vertical list.

    Parameters:
    - story_prefixes (list): List of mnemonic prefixes.

    Returns:
    - str: A string representation of the numbered prefixes list.
    """
    #return "\n".join(f"{idx + 1}. {prefix}" for idx, prefix in enumerate(story_prefixes))
    return ", ".join(f"{idx + 1}. {prefix}" for idx, prefix in enumerate(seed_prefixes))

def display_mnemonic_details(seed_phrase):
    """
    Validates the seed phrase and displays relevant guidance and warnings.
    
    Parameters:
    - seed_phrase (str): The BIP-39 mnemonic phrase to validate and display.
    - mnemo (Mnemonic): An instance of the Mnemonic class for validation.
    """
    mnemo = Mnemonic("english")
    if mnemo.check(seed_phrase):
        seed = mnemo.to_seed(seed_phrase)
        master_key = BIP32Key.fromEntropy(seed)
        private_key = binascii.hexlify(master_key.PrivateKey()).decode()
        master_public_key = binascii.hexlify(master_key.PublicKey()).decode()
        print(
f"""
### Mnemonic Recovery Phrase
> {seed_phrase}

### Seed Phrase Validation
🎉 The seed phrase is valid!
Master Private Key (WIF): {master_key.WalletImportFormat()}

### Wallet Setup Guidance
- If prompted for seed type, choose BIP-39 (not Electrum or SLIP39).
- If prompted for the type of addresses in your wallet, choose Native SegWit for best performance and lower fees.
- Your crypto wallet will derive both private and public keys from your seed phrase (and optional passphrase).

### ⚠️ Important Security Warning
- If your seed phrase was not randomly generated, it could be at risk of being cracked by attackers who attempt combinations of published words to steal your cryptocurrency.
- For enhanced protection, consider specifying a BIP-39 passphrase when setting up your wallet. This passphrase transforms your seed phrase into a unique wallet, ensuring that even if your seed phrase is exposed, your funds remain secure without the passphrase. 
- Alternatively, you can use this seed phrase as part of a **multi-signature wallet**, which requires multiple keys to authorize transactions and further reduces risk.
- 🔒 Always back up your seed phrase and passphrase securely, and never share them with anyone. Without them, you will not be able to recover your funds.
""")
    else:
        print("\n❌ Invalid seed phrase. Please check your input and try again.")

def apply_strikethrough_to_prefixes(story_text, unintended_prefixes):
    """
    Formats unintended prefixes in the story text with strikethrough markdown (~~).
    
    Args:
        story_text (str): Original story text.
        unintended_prefixes (list): List of prefixes to format.
        
    Returns:
        str: Story text with unintended prefixes formatted.
    """
    for prefix in unintended_prefixes:
        # Match the prefix as a whole word or as the start of a longer word
        pattern = rf'\b{re.escape(prefix)}\w*\b'
        story_text = re.sub(pattern, lambda m: f"~~{m.group(0)}~~", story_text, flags=re.IGNORECASE)
        
    story_text = re.sub(r'~{2,}', '~~', story_text)
    return story_text

def calculate_missing_score(missing_count):
    return max(0, 100 - (5 * missing_count))

def calculate_order_score(is_sequence_correct):
    return 100 if is_sequence_correct else 50

def calculate_unintended_score(unintended_count):
    return max(0, 100 - (3 * unintended_count))

def calculate_overall_score(missing_count, order_score, unintended_count):
    """
    Calculates the overall score based on missing count, order score, and unintended count.

    Args:
        missing_count (int): Number of missing prefixes.
        order_score (float): Score based on longest common subsequence (0-100).
        unintended_count (int): Number of unintended prefixes.

    Returns:
        float: Overall score (0-100).
    """
    missing_score = max(0, 100 - (5 * missing_count))
    unintended_score = max(0, 100 - (3 * unintended_count))
    
    # Combine scores with weights
    overall_score = (0.5 * missing_score) + (0.3 * order_score) + (0.2 * unintended_score)
    return overall_score

def longest_common_subsequence_length(seq1, seq2):
    """
    Computes the length of the longest common subsequence (LCS) between two sequences.

    Args:
        seq1 (list): The required sequence.
        seq2 (list): The actual sequence.

    Returns:
        int: The length of the LCS.
    """
    # Create a 2D table to store LCS lengths
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Fill the table using dynamic programming
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    return dp[m][n]

def calculate_order_score_lcs(required_sequence, actual_sequence):
    """
    Calculates a normalized score based on the longest common subsequence (LCS) length.

    Args:
        required_sequence (list): The required mnemonic sequence.
        actual_sequence (list): The actual mnemonic sequence.

    Returns:
        float: Normalized LCS score (0.0 to 100.0).
    """
    lcs_length = longest_common_subsequence_length(required_sequence, actual_sequence)
    max_possible_length = len(required_sequence)
    
    # Normalize the LCS length to a percentage score
    return (lcs_length / max_possible_length) * 100

def format_blockquote(story_text):
    """Formats multiline text as a Markdown blockquote."""
    lines = story_text.splitlines()
    formatted_text = "\n".join(f"> {line}" for line in lines)
    formatted_text += "\n"
    return formatted_text

def get_story_text_with_editor(initial_text=""):
    """
    Opens the user's preferred editor (from $EDITOR) to edit story text.
    
    Args:
        initial_text (str): Optional initial text to prepopulate in the editor.
    
    Returns:
        str: The edited story text.
    """
    # Get the user's preferred editor from $EDITOR or default to 'nano'
    editor = os.environ.get("EDITOR", "nano")
    
    # Create a temporary file
    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as tmpfile:
        tmpfile_name = tmpfile.name
        # Write the initial text to the temp file
        tmpfile.write(initial_text)
        tmpfile.flush()
    
    try:
        # Open the editor with the temporary file
        subprocess.run([editor, tmpfile_name], check=True)
        
        # Read the edited text back from the file
        with open(tmpfile_name, "r") as tmpfile:
            edited_text = tmpfile.read()
    finally:
        # Clean up: Delete the temporary file
        os.unlink(tmpfile_name)
    
    return edited_text.strip()

def find_duplicate_keywords_by_prefix(seed_prefixes_in_story):
    """
    Group words by their first four letters (case-insensitive).
    Return a dict of { prefix: [list of matching words] } 
    where each prefix has more than one matching word.
    """
    prefix_map = defaultdict(list)

    for word in seed_prefixes_in_story:
        prefix = get_prefix(word)
        if prefix:
            prefix_map[prefix].append(word)

    # Filter out prefixes with only one occurrence
    duplicates = {
        prefix: words
        for prefix, words in prefix_map.items()
        if len(words) > 1
    }
    return duplicates

def main():
    mnemonic_phrase = None  # Initialize to None to avoid UnboundLocalError
    mnemo = Mnemonic("english")
    wordlist = mnemo.wordlist

    # Security warning
    note = textwrap.dedent(
    f"""\
    {Fore.YELLOW}{Style.BRIGHT}⚠️  IMPORTANT SECURITY NOTE  ⚠️
    {Fore.CYAN}{Style.BRIGHT}Humans (including AI) are not a good source of randomness.
    {Fore.GREEN}Always rely on true randomness for security, and {Fore.RED}NEVER{Fore.GREEN} use anything that's been published or easily guessed.
    
    🔒 **Local AI Use Only**
    This toolkit is designed with the assumption that all AI-related features (e.g., generating overlays, validating stories) are handled locally, using tools like Ollama or similar offline AI models. 
    - **Do not use this toolkit with online AI services**, such as ChatGPT or any cloud-based AI, to avoid sharing sensitive data like seed phrases or private keys.
    - All data should remain on your device to ensure privacy and security.
    
    Remember: Security is your responsibility. Always back up your mnemonic phrases and passphrases securely, and never share them with anyone.
    {Style.RESET_ALL}\
    """)
    print(note)

    while True:
        print("\nWelcome to the Mnemonic Toolkit!")
        print("1. Generate a random private key.")
        print("2. Provide a BIP-39 mnemonic phrase.")
        print("3. Make a valid seed phrase from free-form text.")
        print("4. Decode and validate a story.")
        print("5. Check extracted BIP-39 words against a provided mnemonic seed phrase.")
        print("6. Generate a list of valid mnemonic phrases.")
        print("q. Exit.")
        choice = input(Fore.RED + "> " + Style.RESET_ALL).strip()

        if choice == "1":
            # Prompt for mnemonic length
            print("\nChoose the length of your mnemonic seed phrase:")
            print("Enter [12], 15, 18, 21, or 24:")
            user_input = input(Fore.RED + "> " + Style.RESET_ALL).strip()
        
            valid_lengths = {12: 128, 15: 160, 18: 192, 21: 224, 24: 256}
            if user_input.isdigit() and int(user_input) in valid_lengths:
                strength = valid_lengths[int(user_input)]
            else:
                print("Defaulting to 12-word mnemonic (128 bits).")
                strength = 128  # Default to 128 bits

            # Generate the mnemonic and seed
            #mnemo = Mnemonic("english")
            seed_phrase = mnemo.generate(strength=strength)
            seed = mnemo.to_seed(seed_phrase)

            # Display generated mnemonic and key details
            display_mnemonic_details(seed_phrase)

        elif choice == "2":
            seed_phrase = prompt_seed_phrase(seed_phrase=None)
            seed_phrase = re.sub(r'[^a-zA-Z\s]', '', seed_phrase)
            seed_words = seed_phrase.strip().split()
            seed_prefixes = words_to_prefixes(seed_words)

            if not is_valid_mnemonic_length(seed_prefixes):
                print(f"❌ Invalid mnemonic length: {len(seed_words)} words. BIP-39 only supports 12, 15, 18, 21, or 24 words.")
                continue
        
            display_mnemonic_details(seed_phrase)

        elif choice == "3":
            mnemo = Mnemonic("english")
            prompt = []
            prompt.append(begin_prompt())
            #print("Write an inspirational quote about patience and rewards. Aim for a length of about 12 to 24 words.")
            prompt.append("Write an inspirational quote about savings and compound interest and the rewards of time in the market for achieving goals and dreams. Aim for a length that contains about 12 to 24 mnemonic keywords.")
            prompt.append(end_prompt())
            print('\n'.join(prompt))
            
            # Option 4 stuff
            while True:
                #story_text = get_story_text()
                # Pre-prompt message
                preprompt = input("Enter message or (q)uit. Press Enter to start the editor...")
                if preprompt.lower() == 'q':
                    break  # user aborted
                story_text = get_story_text_with_editor()
                if story_text == "q":
                    break
        
                story_words = re.findall(r'\b\w+\b', story_text)
                mnemonic_phrase = reconstruct_mnemonic_from_prefixes(story_words, mnemo.wordlist)
                mnemonic_words = mnemonic_phrase.split()
                seed_prefixes = [get_prefix(word) for word in mnemonic_words if get_prefix(word)]
                numbered_story = format_story_with_numbers(story_text, seed_prefixes)
                prompt = []
                prompt.append(begin_prompt())
                prompt.append("### Annotated Draft")
                prompt.append(f"> {numbered_story}\n")
                prompt.append(f"Matched mnemonic keywords ({len(mnemonic_words)}): {', '.join(mnemonic_words)}\n")
                
                if len(mnemonic_words) in {11, 12, 14, 15, 17, 18, 20, 21, 23, 24}:
                    prompt.append("Mission accomplished! That's a length we can work with!\n")
                    prompt = [line for line in prompt if "AI Prompt" not in line] # remove the AI prompt
                    print("\n".join(prompt))
                    break
                elif len(mnemonic_words) < 12:
                    prompt.append("That's a bit shorter than 12 mnemonic keywords. Please lengthen it.")
                    prompt.append(end_prompt())
                    print("\n".join(prompt))
                elif len(mnemonic_words) > 24:
                    prompt.append("That's a bit longer than 24 mnemonic keywords. Please shorten it.")
                    prompt.append(end_prompt())
                    print("\n".join(prompt))
                else:
                    prompt.append("🚫 Aim for 12, 15, 18, 21, or 24 keywords.")
                    prompt.append(end_prompt())
                    print("\n".join(prompt))

            # Option 3 stuff
            input_seed_words = re.findall(r'\b\w+\b', mnemonic_phrase) # strip punctuation

            if len(input_seed_words) in [12, 15, 18, 21, 24]:
                input_seed_words = input_seed_words[:-1]
            
            # Reconstruct the words from prefixes/full words
            seed_phrase = reconstruct_mnemonic_from_prefixes(input_seed_words, mnemo.wordlist)
        
            if seed_phrase:
                possible_words = [mnemo.wordlist[i] for i in range(2048) if mnemo.check(seed_phrase + " " + mnemo.wordlist[i])]
                
                # Find the last matched mnemonic keyword in the story
                last_mnemonic_keyword = mnemonic_words[-1] if mnemonic_words else None
                
                # Adjust the story text based on the input_seed_words length
                modified_story_text = adjust_story_text(story_text, mnemonic_words, last_mnemonic_keyword)

                prompt = []
                prompt.append(begin_prompt())
                prompt.append(textwrap.dedent(
                    f"""\
                    ### Task: Complete the Message Using a Valid Mnemonic Keyword
                    
                    Complete the following message by selecting **only one word** from the provided list of valid mnemonic checksum keywords. No other words are allowed. Your goal is to ensure the final sentence is coherent, meaningful, and consistent with the message's tone.
                    
                    > {modified_story_text} ___
                    
                    ### Valid Mnemonic Keywords:
                    {possible_words}
                    
                    ### Rules:
                    1. **Only choose a word from the list.** Do not add any other word not provided.
                    2. Do not rewrite or complete the sentence using words outside the list.
                    3. The final sentence must remain coherent and meaningful with the chosen word.
                    
                    Now, select the best word from the list to complete the sentence.\
                    """))
                prompt.append(end_prompt())
                print('\n'.join(prompt))

                # Prompt the user to choose a final word
                manual_word = input(Fore.RED + "\nPress Enter to use the first valid word or provide your own final word: " + Style.RESET_ALL).strip()
                if manual_word and manual_word in possible_words:
                    final_mnemonic = seed_phrase + " " + manual_word
                else:
                    print("\nUsing the first valid final word by default.")
                    final_mnemonic = seed_phrase + " " + possible_words[0]
                    manual_word = possible_words[0]
        
                print(f"### Your message\n> {modified_story_text} {manual_word}")
                display_mnemonic_details(final_mnemonic)
            else:
                print("\n❌ Error: One or more of your words could not be reconstructed into valid BIP-39 words.")

        elif choice == "4":
            while True:
                #story_text = get_story_text()
                # Pre-prompt message
                preprompt = input("Enter message or (q)uit. Press Enter to start the editor...")
                if preprompt.lower() == 'q':
                    break  # user aborted
                story_text = get_story_text_with_editor()
                if story_text == "q":
                    break
                story_to_private_key(story_text)

        elif choice == "5":
            # 1. Prompt for the seed phrase (reuse existing prompt_seed_phrase function)
            input_seed_phrase = prompt_seed_phrase(seed_phrase=None)
            input_seed_phrase_cleaned = re.sub(r'[^a-zA-Z\s]', '', input_seed_phrase)
            input_seed_words = input_seed_phrase_cleaned.strip().split()
            
            # 2. Silently confirm that it meets BIP-39 length
            if not is_valid_mnemonic_length(input_seed_words):
                print(f"❌ Invalid mnemonic length: {len(input_seed_words)} words. Must be 12, 15, 18, 21, or 24.")
                continue
            
            # 3. Convert the seed words into prefixes (3-letter words stay as-is, 4+ letters truncated to 4)
            seed_prefixes = [get_prefix(word) for word in input_seed_words if get_prefix(word)]
            numbered_seed_prefixes = numbered_list(seed_prefixes)

            # 4. Convert the seed words into prefixes (3-letter words stay as-is, 4+ letters truncated to 4)
            try:
                seed_phrase = reconstruct_mnemonic_from_prefixes(seed_prefixes, wordlist)
                seed_words = seed_phrase.split()  # Rebuild seed_words from the reconstructed seed_phrase
            except ValueError as e:
                print(f"❌ Reconstruction failed: {e}")
                continue
            
            # 3. Also check the BIP-39 checksum (using the global 'mnemo' object from your script)
            if not mnemo.check(seed_phrase):
                print("❌ The provided seed phrase fails the BIP-39 checksum. Please verify it.")
                continue
            
            prompt = []
            prompt.append(begin_prompt())
            prompt.append(textwrap.dedent(
                f"""\
                I want to create a motivational quote or inspirational message that integrates the valid BIP-39 mnemonic phrase, utilizing the first 4 unique letters of each mnemonic keyword (the mnemonic keyword prefix). The goal is to make the seed phrase more memorable and easier to recall, while maintaining its integrity and security. I'll provide iterative feedback to ensure that the order of the keywords is preserved and that no unintended mnemonic keywords are introduced.

                ### Challenge
                Complete the following list of prefixes by expanding them creatively into meaningful full words. Your final output should be a coherent, inspirational message.

                ### Instructions\
                """))

            # Find the first word in seed_words with a length of 5+ letters
            example_index = next(
                (i for i, word in enumerate(seed_words) if len(word) >= 5),
                None  # Default to None if no such word exists
            )
            
            if example_index is not None:
                example_prefix = seed_prefixes[example_index]
                example_word = seed_words[example_index]
                prompt.append(f"1. Expand each 4-letter prefix into a full word (e.g., '{example_prefix}' → '{example_word}').")
            else:
                prompt.append("1. Expand each 4-letter prefix into a full word (e.g., 'exam' → 'example').")

            # Choose an example 3-letter word
            three_letter_word = next((prefix for prefix in seed_prefixes if len(prefix) == 3), 'you')

            prompt.append(
f"""\
2. Keep 3-letter prefixes unchanged (e.g., '{three_letter_word}' → '{three_letter_word}').
3. Use the completed words in the listed order to create a meaningful message.  
4. Incorporate just enough transitional words or phrases to maintain narrative flow, favoring rare or evocative word choices where possible.
5. Generate your own unique solution—do not copy the example.

### Required Prefix Sequence ({len(seed_prefixes)} total)
{numbered_seed_prefixes}

### Example (for structure reference only—do not copy)
Given prefixes: pati → like → grow → mome → pati → plan → seed → succ → that → bloo → time → bene
Example solution: Patience, like fertile soil, nurtures growth unseen—each moment of patience plants seeds of success that bloom with time's tender benefit.

### Your task
Complete the list with a meaningful original solution.\
""")
            prompt.append(end_prompt())
            print('\n'.join(prompt))

            while True:
                # 5. Prompt for the story text
                #story_text = get_story_text()
                # Pre-prompt message
                preprompt = input("Enter message or (q)uit. Press Enter to start the editor...")
                if preprompt.lower() == 'q':
                    break  # user aborted

                story_text = get_story_text_with_editor()
                if story_text.lower() == 'q':
                    break  # user aborted
            
                # 6. Derive story_words and their prefixes (3+ letters only)
                numbered_story = format_story_with_numbers(story_text, seed_prefixes)
                story_words = re.findall(r'\b\w+\b', story_text.lower())
                story_prefixes = extract_all_mnemonic_prefixes(story_text, mnemo.wordlist)
                seed_prefixes_in_story = [prefix for prefix in story_prefixes if prefix in seed_prefixes]
            
                # 7. Find missing prefixes
                #    (i.e. which seed_prefixes are not found at all or not enough times in the story)
                missing_prefixes = []
                seed_prefix_counts = Counter(seed_prefixes)
                story_prefix_counts = Counter(story_prefixes)
                
                for prefix, needed_count in seed_prefix_counts.items():
                    if story_prefix_counts[prefix] < needed_count:
                        missing_count = needed_count - story_prefix_counts[prefix]
                        missing_prefixes.extend([prefix] * missing_count)
            
                # 8. Check sequence order
                sequence_is_ok = is_subsequence_in_order(seed_prefixes, story_prefixes)
            
                # 9. Identify unintended prefixes (prefixes found in the story that do not exist in seed_prefixes)
                #    or exceed the required count
                unintended_prefixes = []
                for prefix, found_count in story_prefix_counts.items():
                    if prefix not in seed_prefix_counts:
                        unintended_prefixes.extend([prefix] * found_count)
                    elif found_count > seed_prefix_counts[prefix]:
                        # If story has more instances than the seed needs
                        extra_count = found_count - seed_prefix_counts[prefix]
                        unintended_prefixes.extend([prefix] * extra_count)
                unintended_count = len(unintended_prefixes)

                if unintended_prefixes:
                    draft_message = apply_strikethrough_to_prefixes(numbered_story, unintended_prefixes)
                else:
                    draft_message = numbered_story

                duplicates = find_duplicate_keywords_by_prefix(story_text)
            
                prompt = []
                prompt.append(begin_prompt())
                prompt.append(textwrap.dedent(
                    f"""\
                    Challenge: Integrate an inspirational quote as a mnemonic overlay for a BIP-39 seed phrase, ensuring it remains secure and high-entropy.
                    
                    ### Message Draft\
                    """))
                prompt.append(f"{format_blockquote(draft_message)}")
                prompt.append("### Required mnemonic prefix sequence")
                prompt.append(f"{numbered_seed_prefixes}\n")
                prompt.append("### Validation Checks")
            
                revise_prompt = []
                revise_prompt.append(begin_prompt())
                revise_prompt.append("### Revise")
                if missing_prefixes:
                    # Get the full-word completions for missing prefixes
                    completions_for_missing = keyword_prefix_completions(" ".join(missing_prefixes), wordlist)
                    hint_list = [
                        f"{prefix} ({get_full_word_from_prefix(prefix, wordlist) or 'unknown'})"
                        for prefix in missing_prefixes
                    ]
                    prompt.append(f"- ❌ **Missing prefixes**: Ensure all prefixes are included, particularly '{', '.join(hint_list)}'")
                    revise_prompt.append(f"- Rewrite the sentence where '{', '.join(hint_list)}' should appear.")
                else:
                    prompt.append(f"- ✅ All {len(seed_words)} required mnemonic prefixes are present!")
            
                if duplicates:
                    prompt.append("🚫 **Duplicate prefixes**: Minimize duplicate prefixes (first 4 letters):")
                    for prefix, words in duplicates.items():
                        prompt.append(f"- '{prefix}' appears {len(words)} times ({', '.join(words)})")

                if sequence_is_ok:
                    prompt.append("- ✅ The mnemonic prefixes appear in the correct order!")
                else:
                    prompt.append("- ❌ **Order**: Ensure the prefixes appear in the listed order.")
                    revise_prompt.append("- Adjust the order of the mnemonic words to match the sequence.")
            
                if unintended_prefixes:
                    prompt.append(f"- ❌ **Unintended Prefixes**: Replace any words that inadvertently match BIP-39 prefixes not in the required sequence, particularly '{','.join(unintended_prefixes)}'.")
                    revise_prompt.append("- Remove or replace words in strikethrough from the original draft.")
                    prompt = [line for line in prompt if "Fore.YELLOW" not in line] # remove the AI prompt
                    #prompt.append(end_prompt())
                    #print('\n'.join(prompt))
                else:
                    prompt.append("- ✅ No unintended prefixes discovered!\n")
                    prompt.append("🎉 Congratulations. You've solved the puzzle!")
                    prompt.append(end_prompt())
                    print('\n'.join(prompt))
                    print(f"### Your message\n> {story_text}")
                    display_mnemonic_details(seed_phrase)
                    #continue

                if len(missing_prefixes) > 0 or not sequence_is_ok or unintended_count > 0:
                    revise_prompt.append("\nNote: Complete the list of prefixes by expanding them creatively into meaningful full words. Your final output should be a coherent, inspirational message")
                    combined_prompt = prompt + revise_prompt
                    combined_prompt.append(end_prompt())
                    print('\n'.join(combined_prompt))

                # Required vs actual
                #print(f"Required Sequence: {seed_prefixes}")
                #print(f"Actual Sequence:   {seed_prefixes_in_story}\n")
                
                # Scoring metrics
                missing_count = len(missing_prefixes)
                order_score = calculate_order_score_lcs(seed_prefixes, story_prefixes)
                overall_score = calculate_overall_score(missing_count, order_score, unintended_count)
                print(f"Missing Count: {missing_count}")
                print(f"Order Score: {order_score:.2f}")
                print(f"Unintended Count: {unintended_count}")
                print(f"Overall Score: {overall_score:.2f}")
 
        elif choice == "6":
            mnemo = Mnemonic("english")

            prompt = []
            prompt.append(begin_prompt())
            prompt.append("### Mnemonic phrase flow challenge")
            prompt.append("Review the following list of mnemonic keyword seed phrases:\n")

            # Hard coded list of phrase lengths to iterate over
            phrase_lengths = [12, 15, 18, 21, 24]
            for phrase_length in phrase_lengths:
                # Calculate entropy length based on phrase length
                entropy_bits = (phrase_length * 11) - (phrase_length // 3)
                entropy_bytes = entropy_bits // 8  # Convert bits to bytes
                entropy = os.urandom(entropy_bytes)  # Entropy based on selected seed length
                mnemonic_phrase = mnemo.to_mnemonic(entropy)
                seed = mnemo.to_seed(mnemonic_phrase)
                prompt.append(f"- {phrase_length}-word mnemonic: {mnemonic_phrase}")

            prompt.append(textwrap.dedent(
                """
                Take a moment to review each phrase carefully. Consider:
                - Which phrase flows naturally when spoken aloud?
                - Which phrase contains keywords that could inspire a meaningful story or motivational message?
                - Are there any phrases with vivid or memorable imagery that spark an idea for a coherent message?
            
                Which of the provided mnemonic phrases do you find the easiest to build a coherent message from?
                Once you’ve made your selection, let me know, and I’ll give you the next instruction.\
                """))
            prompt.append(end_prompt())
            print("\n".join(prompt))

        elif choice == "q":
            print("Goodbye!")
            break

        else:
            print("Invalid choice. Please enter a valid option.")

if __name__ == "__main__":
    main()

