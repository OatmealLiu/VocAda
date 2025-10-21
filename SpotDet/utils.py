from termcolor import colored

def extract_nouns(nlp_model, caption, root=False):
    # some nouns that are commonly not wanted! just make our life easier
    not_wanted_nouns = [
        'where', 'what', 'which', 'when', 'who', 'whom', 'the image', 'an image', 'image', 'each', 'every', 'color',
        'left', 'right', 'the left', 'the right', 'the middle', 'middle', 'a variety', 'variety', 'piece', 'a piece',
        'types', 'type', 'photo', 'a photo',
    ]

    doc = nlp_model(caption)

    noun_chunks = []
    for chunk in doc.noun_chunks:
        # E.g.,: "three plastic containers"
        # if root is true, we use root noun only: "container", less context"
        # else we use the full noun: "three plastic containers"
        noun = chunk.root.text if root else chunk.text
        if len(noun) <= 2:
            continue
        elif noun.lower() in not_wanted_nouns:
            continue
        else:
            noun_chunks.append(str(noun))

    return noun_chunks


def clrprint(input, c='light_yellow'):
    print(colored(input, c))

