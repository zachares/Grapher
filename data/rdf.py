from typing import List, Tuple

import os
from xml.etree.ElementTree import Element, SubElement
from xml.etree import ElementTree
from xml.dom import minidom


import logging

logger = logging.getLogger(__name__)
log = logger


def create_xml(
    data: List[List[str]],
    categories: List[str],
    ts_header: str,
    t_header: str
) -> Element:
    """ Returns an xml.etree.Element representation of the input data """
    benchmark = Element('benchmark')
    entries = SubElement(benchmark, 'entries')

    assert len(categories) == len(data)

    for idx, triples in enumerate(data):
        entry = SubElement(entries, 'entry', {'category': categories[idx], 'eid': 'Id%s' % (idx + 1)})
        t_entry = SubElement(entry, ts_header)
        for triple in triples:
            element = SubElement(t_entry, t_header)
            element.text = triple
    return benchmark


def xml_prettify(elem):
    """Return a pretty-printed XML string for the Element.
       source : https://pymotw.com/2/xml/etree/ElementTree/create.html
    """
    rough_string = ElementTree.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")


def save_webnlg_rdf(
    hyps: List[List[str]],
    refs: List[List[str]],
    categories: List,
    out_dir: str,
    iteration: str
) -> Tuple[str, str]:
    """ Saves the actual and estimated triples for a batch of knowledge graphs as xml files """
    if len(refs) != len(hyps):
        raise Exception(f"reference size {len(refs)} is not same as hypothesis size {len(hyps)}")

    ref_xml = create_xml(refs, categories, "modifiedtripleset", "mtriple")
    hyp_xml = create_xml(hyps, categories, "generatedtripleset", "gtriple")

    os.makedirs(out_dir, exist_ok=True)

    ref_fname = os.path.join(out_dir, f"ref_{iteration}.xml")
    hyp_fname = os.path.join(out_dir, f"hyp_{iteration}.xml")

    print(f"creating reference xml  file : [{ref_fname}]")
    print(f"creating hypothesis xml file : [{hyp_fname}]")

    with open(ref_fname, 'w', encoding='utf-8') as f:
        f.write(xml_prettify(ref_xml))

    with open(hyp_fname, 'w', encoding='utf-8') as f:
        f.write(xml_prettify(hyp_xml))

    return ref_fname, hyp_fname
