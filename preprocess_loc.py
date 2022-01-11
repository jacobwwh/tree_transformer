#create wrong operator localization data

from tree_sitter import Language, Parser
import json
import jsonlines
import os
from anytree import AnyNode, RenderTree
import anytree.search as search
import re
import codecs
import gzip
import pickle
from tqdm import tqdm
from anytree.exporter import JsonExporter
from anytree.importer import JsonImporter
from torchtext.vocab import Vocab
from collections import defaultdict,Counter

Language.build_library(
  # Store the library in the `build` directory
  'sitter/build/my-languages.so',
  # Include one or more languages
  [
	'sitter/tree-sitter-python',
  ]
)
PY_LANGUAGE = Language('sitter/build/my-languages.so', 'python')
ops=['==', '/', '+', 'or', '!=', '-', 'not', '%', '>', 'and', '*', '<', 'is', '<=', '>=', 'in']


def match_from_span(node, blob):
    lines = blob.split('\n')
    line_start = node.start_point[0]
    line_end = node.end_point[0]
    char_start = node.start_point[1]
    char_end = node.end_point[1]
    if line_start != line_end:
        return '\n'.join([lines[line_start][char_start:]] + lines[line_start+1:line_end] + [lines[line_end][:char_end]])
    else:
        return lines[line_start][char_start:char_end]


def match_from_span_reference(line_start, line_end, char_start, char_end, blob):
    lines = blob.split('\n')
    if line_start != line_end:
        return '\n'.join([lines[line_start][char_start:]] + lines[line_start+1:line_end] + [lines[line_end][:char_end]])
    else:
        return lines[line_start][char_start:char_end]


class pythonTree:
    def __init__(self, original, operator, parser):
        self.ori_blob = original['function']
        self.ori_label = original['label']
        self.ori_info = original['info']
        self.op_blob = operator['function']
        self.op_label = operator['label']
        self.op_info = operator['info']
        self.ori_op, self.mutated_op = self.get_operator()
        self.ori_tree = self.get_ast_nodes(self.ori_blob, self.op_blob, parser, self.ori_op, self.mutated_op, isWrong=False)
        self.op_tree = self.get_ast_nodes(self.op_blob, self.ori_blob, parser, self.mutated_op, self.ori_op, isWrong=True)

    def get_ast_nodes(self, blob, reference, parser, cur_op, mutated_op, isWrong):
        body = bytes(blob, "utf8")
        tree = parser.parse(body)
        tree = converttoany(tree, blob, reference, cur_op, mutated_op, isWrong)
        return tree

    def get_operator(self):
        parts = re.findall(r"`.*?`", self.op_info)
        ori_op = parts[0][1:-1]
        mutated_op = parts[1][1:-1]
        return ori_op, mutated_op

def matchop(identifier,op):
    op=op.split()
    ismatch=False
    if len(op)==1:
        return identifier==op[0]
    else:
        for subtoken in op:
            ismatch=ismatch or identifier==subtoken
        return ismatch

def converttoany(tree, blob, reference, cur_op, mutated_op, isWrong):
    nodeid = [0]
    currentnodetype = tree.root_node.type
    root = AnyNode(name=currentnodetype, index=0, label=0, is_operator=False)
    def traverse(tree, parent=None):            #parent: current anynode for tree
        nodeid[0] += 1
        for child in tree.children:
            if child.children:
                childnode = AnyNode(name=child.type, parent=parent, index=nodeid[0], label=0, is_operator=False)
            else:
                identifier = match_from_span(child, blob)
                if identifier in ops:
                    is_operator=True
                else:
                    is_operator=False
                if identifier==cur_op and isWrong:
                    char_start = child.start_point[1]
                    char_end = child.start_point[1] + len(mutated_op)
                    line_start = child.start_point[0]
                    line_end = child.end_point[0]
                    mutated_identifier = match_from_span_reference(line_start, line_end, char_start, char_end, reference)
                    if mutated_op == mutated_identifier:
                        childnode = AnyNode(name=identifier, parent=parent, index=nodeid[0], label=1, is_operator=is_operator)
                    else:
                        childnode = AnyNode(name=identifier, parent=parent, index=nodeid[0], label=0, is_operator=is_operator)
                else:
                    childnode = AnyNode(name=identifier, parent=parent, index=nodeid[0], label=0, is_operator=is_operator)
            traverse(child, parent=childnode)
    traverse(tree.root_node, parent=root)
    return root


def load_data(file_path):
    instances = []
    with open(file_path, encoding='utf-8') as f:
        json_list = list(f)
    pair = []
    for index, json_str in enumerate(json_list):
        sample = json.loads(json_str)
        pair.append(sample)
        if index % 2 != 0:
            instances.append(pair)
            pair = []
    return instances


def save_jsonl_gz(file_name, data):
    with gzip.GzipFile(file_name, 'w') as out_file:
        writer = codecs.getwriter('utf-8')
        for ele in data:
            writer(out_file).write(json.dumps(ele))
            writer(out_file).write('\n')

def save_jsonl(file_name, data):
    with open(file_name,'w') as f:
        for sample in data:
            f.write(json.dumps(sample))
            f.write('\n')


def process(dir,mode='train'):
    files = os.listdir(dir)
    parser = Parser()
    exporter = JsonExporter(indent=2, sort_keys=True)
    dataset=[]
    for file in files:
        file_path = os.path.join(dir, file)
        if not file_path.endswith('jsonl') and not file_path.endswith('all'):
            print(file_path)
            instances = load_data(file_path)
            parser.set_language(PY_LANGUAGE)
            new_instances = []
            for instance in tqdm(instances):
                ast = pythonTree(instance[0], instance[1], parser)
                bugnodes = search.findall(ast.op_tree, lambda node: node.label == 1)
                if len(bugnodes) == 1:
                    assert bugnodes[0].is_operator == True
                    new_instances.append({'buggy_function': ast.op_blob, 'tree': exporter.export(ast.op_tree)})
            save_jsonl(file_path + '.jsonl', new_instances)

def getvocab(path):
    wordcount=Counter()
    importer = JsonImporter()
    def dfs(node):
        nodename=node.name
        wordcount[nodename]+=1
        for child in node.children:
            dfs(child)
    with jsonlines.open(path) as reader:
        for sample in reader:
            root = importer.import_(sample['tree'])
            dfs(root)
    vocab=Vocab(wordcount,min_freq=2)
    print(len(vocab))
    vocabfile = open('wrongoperator/vocab_min2.pkl', 'wb')
    pickle.dump(vocab, vocabfile)
    vocabfile.close()
    vocab=Vocab(wordcount,min_freq=5)
    print(len(vocab))
    vocabfile = open('wrongoperator/vocab_min5.pkl', 'wb')
    pickle.dump(vocab, vocabfile)
    vocabfile.close()


if __name__ == '__main__':
    file_dir='wrongoperator/train/'
    process('wrongoperator/train/',mode='train')
    process('wrongoperator/dev/',mode='dev')
    process('wrongoperator/test/',mode='test')
    #getvocab('wrongoperator/train/train.jsonl')
