import sys
import csv
import glob
import itertools
import xml.etree.ElementTree as ET

import rfutils

flat = itertools.chain.from_iterable

XML_PATH = "/Users/canjo/data/link/nxt_switchboard_ann/xml"

SYNTAX_PATH = "%s/syntax/%s.syntax.xml"
MARKABLE_PATH = "%s/markable/%s.markable.xml"
TERMINALS_PATH = "%s/terminals/%s.terminals.xml"
REPAIRS_PATH = "%s/disfluency/%s.disfluency.xml"

DEFINITE_DETERMINERS = {
    'the',
    'this',
    'that',
    'th-',
    'these',
    'those',
    'my',
    'your',
    'his',
    'her',
    'its',
    'our',
    'their',
}

def depth_first_map(f, root):
    def traverse(node):
        yield f(node)
        yield from flat(map(traverse, node))
    return traverse(root)

def extract_syntax(identifier):
    """ Create map from NT id's to immediate terminal children """
    tree = ET.parse(SYNTAX_PATH % (XML_PATH, identifier))
    def traverse(node):
        id = node.attrib["{http://nite.sourceforge.net/}id"]
        children = []
        for child in node:
            if child.tag == '{http://nite.sourceforge.net/}child':
                child_id = extract_id(child.attrib['href'])
                children.append(('t', child_id))
            else:
                if child.tag == 'nt':
                    cat = child.attrib['cat']
                    if 'subcat' in child.attrib:
                        cat += '-' + child.attrib['subcat']
                    children.append((cat, parse_id(child.attrib["{http://nite.sourceforge.net/}id"])))
                yield from traverse(child)
        if node.tag == 'nt':
            nodecat = node.attrib.get('cat')
            if 'subcat' in node.attrib:
                nodecat += '-' + node.attrib['subcat']
            yield parse_id(id), (nodecat, children)
    return dict(traverse(tree.getroot()))

def extract_id(pointer):
    return parse_id(pointer.strip("'").split('id(')[-1].strip(')'))

def extract_reparanda(identifier):
    tree = ET.parse(REPAIRS_PATH % (XML_PATH, identifier))
    reparanda = set()
    def get_id(node):
        if node.tag == "{http://nite.sourceforge.net/}child":
            return extract_id(node.attrib['href'])
        else:
            return None
    for disfluency in tree.getroot():
        try:
            reparandum, repair = disfluency
        except ValueError:
            print("Wrongly formatted disfluency: %s" % str(disfluency.attrib), file=sys.stderr)
            continue
        new_reparanda = filter(None, depth_first_map(get_id, reparandum))
        reparanda.update(new_reparanda)
    return reparanda

def extract_terminals(identifier):
    tree = ET.parse(TERMINALS_PATH % (XML_PATH, identifier))
    def extract(word):
        if '{https://nite.sourcefourge.net/}id' in word.attrib:
            id = parse_id(word.attrib['{https://nite.sourceforge.net/}id'])
        else:
            id = parse_id(word.attrib['{http://nite.sourceforge.net/}id'])

        if word.tag == 'punc':
            result = {'pos': 'PUNCT', 'orth': word.text}
        elif word.tag == 'word':
            result = {'pos': word.attrib.get('pos'), 'orth': word.attrib.get('orth')}
        elif word.tag == 'sil' or word.tag == 'trace':
            result = {}
        else:
            raise ValueError("Unknown word tag: %s" % word.tag)
        
        return id, result
    
    return dict(map(extract, tree.getroot()))

def parse_id(s):
    return tuple(map(int, s.strip("'").lstrip('s').split('_')))

def is_definite(syntax, terminals, phrase):
    cats, ids = zip(*phrase)

    # an NP counts as definite if any of the following hold:
    def conditions():
        # condition 1: NP contains simple pronoun
        yield 'PRP' in cats

        # condition 2: NP contains proper name
        yield 'NNP' in cats
        yield 'NNPS' in cats

        # condition 3: NP contains a possessive pronoun 
        yield 'PRP$' in cats # determiner is possessive pronoun

        # condition 4: NP contains a definite determiner
        determiners = [terminals[id].get('orth') for id, cat in zip(ids, cats) if cat == 'DT']
        yield any(determiner.lower() in DEFINITE_DETERMINERS for determiner in determiners)

        # condition 5: first NP inside NP is an s-genitive, like "our nation's capital"
        if cats[0] == 'NP':
            _, np_children = syntax[ids[0]]
            *_, (last_child_type, last_child) = np_children
            yield last_child_type == 't' and terminals.get(last_child, {}).get('pos') == 'POS'

    return any(conditions())

def extract_markable(identifier):
    try:
        tree = ET.parse(MARKABLE_PATH % (XML_PATH, identifier))
    except FileNotFoundError:
        print("File not found: %s" % MARKABLE_PATH % (XML_PATH, identifier), file=sys.stderr)
        return {}
    def extract(markable):
        d = markable.attrib
        del d['{http://nite.sourceforge.net/}id']
        if d:
            child_id = extract_id(rfutils.the_only(markable).attrib['href'])
            return child_id, d
        else:
            return None
    return dict(filter(None, map(extract, tree.getroot())))

def extract_tokens_and_annotations(identifier, exclude_reparanda=True, exclude_uh=True, exclude_youknow=True):
    markable = extract_markable(identifier)
    syntax = extract_syntax(identifier)
    terminals = extract_terminals(identifier)
    if exclude_reparanda:
        reparanda = extract_reparanda(identifier)

    markable_terminals = markable.copy()
    for node, marks in markable.items():
        if node in syntax: # need to identify heads here...?
            cat, children = syntax[node]
            marks['cat'] = cat            
            labelled_children = [
                (terminals.get(child, {}).get('pos'), child) if type == 't' else (type, child)
                for type, child in syntax[node][-1] 
            ]
            if labelled_children:
                marks['definiteness'] = 'definite' if is_definite(syntax, terminals, labelled_children) else 'indefinite'
            markable_terminals.update({child:marks for _, child in children})

    seen = set()
    def traverse(nt):
        if nt in seen:
            pass
        else:
            children = syntax[nt][-1]
            for type, child in children:
                if type == 't':
                    if not (exclude_reparanda and child in reparanda):
                        attribs = terminals.get(child, {})
                        if attribs and not (exclude_uh and attribs.get('pos') == 'UH'):
                            if child in markable_terminals:
                                attribs.update(markable_terminals[child])
                            yield (child, attribs)
                else:
                    yield from traverse(child)
                seen.add(child)
    for s in sorted(syntax.keys()):
        tokens = list(traverse(s))
        nonpunct = [(id,attrib) for id, attrib in tokens if attrib['pos'] != 'PUNCT']
        if nonpunct:
            if exclude_youknow:
                yield filter_youknow(tokens)
            else:
                yield tokens

def filter_youknow(tokens):
    bad_indices = set()
    for i in range(len(tokens) - 1):
        if tokens[i][-1].get('orth').lower() == 'you' and tokens[i+1][-1].get('orth').lower() == 'know':
            bad_indices.add(i)
            bad_indices.add(i+1)
    return [token for i, token in enumerate(tokens) if i not in bad_indices]

def the_unique(xs):
    first, *rest = xs
    assert all(first == x for x in rest)
    return first

def run():
    identifiers = sorted([
        path.split("/")[-1].rstrip(".terminals.xml")
        for path in glob.glob(TERMINALS_PATH % (XML_PATH, "*"))
    ])
    for A, B in rfutils.blocks(identifiers, 2):
        identifier, sA = A.split('.')
        identifier2, sB = B.split('.')
        assert identifier == identifier2
        assert sA == 'A'
        assert sB == 'B'

        A_tokens = extract_tokens_and_annotations(A)
        B_tokens = extract_tokens_and_annotations(B)
        
        A_sentences = [
            ((the_unique(s_id for (s_id, _), _ in sentence), 'A'), sentence)
            for sentence in A_tokens if sentence
        ]
        B_sentences = [
            ((the_unique(s_id for (s_id, _), _ in sentence), 'B'), sentence)
            for sentence in B_tokens if sentence
        ]            
        sentences = sorted(A_sentences + B_sentences)
        
        for (s_id, participant), tokens in sentences:
            for (sentence_id, token_id), ann in tokens:
                assert s_id == sentence_id
                result = {
                    'dialogue': identifier,
                    'sentence_id': s_id,
                    'speaker': participant,                    
                    'token_id': token_id,
                }
                result.update(ann)
                yield result

def main(xml_path=None):
    if xml_path:
        global XML_PATH
        XML_PATH = xml_path
    lines = run()
    writer = csv.DictWriter(
        sys.stdout,
        "dialogue sentence_id speaker token_id orth pos definiteness animacy animconf anthro status statustype cat edin-note stan-note".split(),
    )
    writer.writeheader()
    writer.writerows(lines)

if __name__ == '__main__':
    main(*sys.argv[1:])
    
        
        
        
    
    



    
