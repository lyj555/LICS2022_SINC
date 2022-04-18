# -*- coding: utf-8 -*-

"""
LIC2022 DuSinc dataset preprocessing
"""

import os
import json


def conv_to_gen_query(fin_file, fout_file, is_test=False):
    """
    原始数据集转换为Query生成模型训练所需的格式,
    格式：两列，\t 分割，第一列为input, 第二列为output
    第一部分input：location[SEP]context
    第二部分output：query，回复的文本 或者 不检索
    """
    fout = open(fout_file, "w", encoding="utf-8")
    if is_test:
        fout.write("src\n")
    else:
        fout.write("src\ttgt\n")
    with open(fin_file, encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())
            context = []
            # topical = " ".join(data["user_topical"])
            location = data["user_location"]
            for uttr in data["conversation"]:
                utterence = uttr["utterance"]
                if is_test:
                    context.append(utterence)
                    continue
                if uttr["role"] == "bot":
                    if "use_query" in uttr:
                        query = uttr["use_query"]
                    else:
                        query = "不 检索"
                    outstr = location + " [SEP] " + " [SEP] ".join(context) + "\t" + query
                    fout.write(outstr.strip().replace("\n", " ") + "\n")
                context.append(utterence)
            if is_test:
                outstr = location + " [SEP] " + " [SEP] ".join(context)
                fout.write(outstr.strip().replace("\n", " ") + "\n")
    fout.close()


def conv_to_gen_response(fin_file, fout_file, is_test=False):
    """
    原始数据集转换为知识对话生成模型训练所需的格式，
    格式：三列，\t 分割，
    第一列：knowledge
    第二列：location[SEP]context
    第三列：response
    """
    fout = open(fout_file, "w", encoding="utf-8")
    if is_test:
        fout.write("knowledge\tsrc\n")
    else:
        fout.write("knowledge\tsrc\ttgt\n")
    with open(fin_file, encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())
            context = []
            topical = " ".join(data["user_topical"])
            location = data["user_location"]
            if is_test:
                context = [uttr["utterance"] for uttr in data["conversation"][:-1]]
                if "use_knowledge" in data["conversation"][-1]:
                    knowledge = data["conversation"][-1]["use_knowledge"]
                else:
                    knowledge = ""
                # 对部分过长的知识进行截断，只保留前256个字符
                knowledge = knowledge.replace("\n", " ").replace("\t", " ")[:256]
                outstr = knowledge + "\t" + location + " [SEP] " + " [SEP] ".join(context)
                fout.write(outstr.rstrip().replace("\n", " ") + "\n")
                continue
            for uttr in data["conversation"]:
                if is_test:
                    context.append(uttr["utterance"])
                    continue
                if "use_kg_label" in uttr:
                    if uttr["use_kg_label"] == "true":
                        try:
                            knowledge = uttr["use_knowledge"].replace("\n", " ").replace("\t", " ")
                        except:
                            print(json.dumps(uttr, ensure_ascii=False, indent=2))
                    else:
                        knowledge = ""
                    response = uttr["utterance"]
                    outstr = knowledge + "\t" + location + " [SEP] " + " [SEP] ".join(context) + "\t" + response
                    fout.write(outstr.rstrip().replace("\n", " ") + "\n")
                context.append(uttr["utterance"])
    fout.close()


def prepare_sample(data_path, out_dir, mode):
    """
    generate two files, query and response data
    :param data_path: str
    :param mode: str
    :param out_dir: str
    :return: None
    """
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    query_data_path = os.path.join(out_dir, f"{mode}_query.txt")
    response_data_path = os.path.join(out_dir, f"{mode}_response.txt")

    conv_to_gen_query(data_path, query_data_path, is_test=False)
    conv_to_gen_response(data_path, response_data_path, is_test=False)


if __name__ == "__main__":
    out_dir = "./data/prepare_data"  # used to save prepared data
    train_path = "./data/DuSinc_release/train.txt"
    prepare_sample(train_path, out_dir, mode="train")

    dev_path = "./data/DuSinc_release/dev.txt"
    prepare_sample(dev_path, out_dir, mode="dev")

    test_query_path = "./data/DuSinc_release/test_query_1.txt"
    test_response_path = "./data/DuSinc_release/test_dial_1.txt"
    conv_to_gen_query(test_query_path,
                      os.path.join(out_dir, "test_query_1.txt"), is_test=True)
    conv_to_gen_response(test_response_path,
                         os.path.join(out_dir, "test_dial_1.txt"), is_test=True)
