import os

import cohere
from tqdm import tqdm
from nltk.tokenize import sent_tokenize

def create_root_prompt(root_phrase):

    return f"""
    respond without chit-chat, with very short sentences, and in a single paragraph. write me a beautiful continuation to this start of a paragraph: "{root_phrase}"
    """.strip()

def main():
    final_set = []
    co = cohere.Client(os.environ['COHERE_API_KEY'])

    root_phrase="Does she love me? Or does she love me not?"
    root_sentence = root_phrase + " " + co.chat(model='command-nightly', message=create_root_prompt(root_phrase)).text
    continuations_from_root = sent_tokenize(root_sentence)
    final_set.append(root_sentence)

    second_root_phrase = " ".join(continuations_from_root[:len(sent_tokenize(root_phrase))+1])
    second_root_sentence = second_root_phrase + " " + co.chat(model='command-nightly', message=create_root_prompt(second_root_phrase)).text
    second_continuations_from_root = sent_tokenize(second_root_sentence)
    final_set.append(second_root_sentence)

    # repeat this process in a for loop
    for i in tqdm(range(8)):
        next_root_phrase = " ".join(second_continuations_from_root[:len(sent_tokenize(second_root_phrase))+1])
        next_root_sentence = next_root_phrase + " " + co.chat(model='command-nightly', message=create_root_prompt(next_root_phrase)).text
        next_continuations_from_root = sent_tokenize(next_root_sentence)
        final_set.append(next_root_sentence)

        second_root_phrase = next_root_phrase
        second_continuations_from_root = next_continuations_from_root

    # save final_set to a file
    with open('tree_of_thought_imgs/tree.txt', 'w') as f:
        f.write("\n".join(final_set))

if __name__ == '__main__':
    main()
