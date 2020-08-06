from declare_based.src.constants import *
import re


def generate_rules(rules):
    if rules.strip() == "":
        rules = "True"
        return rules
    if "is" in rules:
        rules = rules.replace("is", "==")
    words = rules.split()
    for index, word in enumerate(words):
        if "A." in word:
            words[index] = "A[\"" + word[2:] + "\"]"
            if not words[index + 2].isdigit():
                words[index + 2] = "\"" + words[index + 2] + "\""
        elif "T." in word:
            words[index] = "T[\"" + word[2:] + "\"]"
            if not words[index + 2].isdigit():
                words[index + 2] = "\"" + words[index + 2] + "\""
        elif word == "same":
            words[index] = "A[\"" + words[index + 1] + "\"] == T[\"" + words[index + 1] + "\"]"
            words[index + 1] = ""
    words = list(filter(lambda word: word != "", words))
    rules = " ".join(words)
    return rules


def parse_decl(path):
    fo = open(path, "r+")
    lines = fo.readlines()
    result = {"activities": {}, "rules": {}}
    for line in lines:
        if line.startswith('activity'):
            if "A" not in result["activities"].keys():
                result["activities"]["A"] = line.split()[1]
            else:
                result["activities"]["B"] = line.split()[1]
        elif (line.startswith(INIT)
              or line.startswith(CHOICE)
              or line.startswith(EXCLUSIVE_CHOICE)):
            key = line.split("[")[0].strip()
            result["rules"][key] = {
                "activation_rules": generate_rules(line.split("|")[1]),
                "correlation_rules": None,
                "n": None
            }
        elif (line.startswith(EXISTENCE)
              or line.startswith(ABSENCE)
              or line.startswith(EXACTLY)):
            key = line.split("[")[0].strip()
            n = 1
            if any(map(str.isdigit, key)):
                n_str = re.search(r'\d+', key).group()
                key = key[:len(n_str)]
                n = int(n_str)

            result["rules"][key] = {
                "activation_rules": generate_rules(line.split("|")[1]),
                "correlation_rules": None,
                "n": n
            }
        elif (line.startswith(RESPONDED_EXISTENCE)
              or line.startswith(RESPONSE)
              or line.startswith(ALTERNATE_RESPONSE)
              or line.startswith(CHAIN_RESPONSE)
              or line.startswith(PRECEDENCE)
              or line.startswith(ALTERNATE_PRECEDENCE)
              or line.startswith(CHAIN_PRECEDENCE)
              or line.startswith(NOT_RESPONDED_EXISTENCE)
              or line.startswith(NOT_RESPONSE)
              or line.startswith(NOT_CHAIN_RESPONSE)
              or line.startswith(NOT_PRECEDENCE)
              or line.startswith(NOT_CHAIN_PRECEDENCE)):
            key = line.split("[")[0].strip()
            result["rules"][key] = {
                "activation_rules": generate_rules(line.split("|")[1]),
                "correlation_rules": generate_rules(line.split("|")[2]),
                "n": None
            }
    fo.close()
    return result