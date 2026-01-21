from jinja2 import Environment, ChoiceLoader, FileSystemLoader, PackageLoader
from typing import Any
# pyright: reportExplicitAny=none

def split_around_brackets(input_str: str) -> list[str]:
    raw_split = input_str.split(",")
    result: list[str] = []
    nested: list[str] = []
    for tag in raw_split:
        if tag.strip().startswith("(") and tag.strip().endswith(")"):
            result.append(tag)
            continue
        if tag.strip().startswith("("):
            nested.append(tag.strip())
            continue
        if nested:
            nested.append(tag.strip())
            if tag.strip().endswith(")"):
                result.append(", ".join(nested))
                nested = []
        else:
            result.append(tag)
    return result

def split_prompt(input_str: str) -> list[str]:
    lines = input_str.split("\n")
    lines = [i.strip() for i in lines]
    lines = [j for j in lines if not j.startswith("#")]
    lines = [k + "," if not k.endswith(",") else k for k in lines]
    lines = "\n".join(lines)
    tags = split_around_brackets(lines)
    tags = [t.strip() for t in tags]
    tags = [t for t in tags if t]
    return tags

def format_prompt(prompt: str) -> tuple[str, list[str]]:
    tags = split_prompt(prompt)
    formatted = ", ".join(tags)
    return (formatted, tags)


def apply_template(template: str, data: dict[str, Any] | None = None, template_dir: str = ".") -> str:
    if data is None:
        data = {}
    env = Environment(
        loader = ChoiceLoader([
            FileSystemLoader(template_dir),
            PackageLoader("diffusers_lab.prompting", "templates")
        ]),
        variable_start_string="[",
        variable_end_string="]"
        )
    tmpl = env.from_string(template)
    return tmpl.render(**data)

def get_il_prompts(main_prompt: str, reality: bool = False, data: dict[str, Any] | None = None, template_dir: str = ".") -> tuple[str, list[str], str, list[str]]:
    pos = '{% import "common.j2" as common %}' + \
        f'[common.il_pos_prefix({reality})],{main_prompt},[common.il_pos_postfix({reality})]'
    pos = apply_template(pos, data, template_dir)
    pos, pos_tags = format_prompt(pos)
    neg = '{% import "common.j2" as common %}' + \
        f'{main_prompt},[common.il_neg ({reality})]'
    neg = apply_template(neg, data, template_dir)
    neg, neg_tags = format_prompt(neg)
    return (pos, pos_tags, neg, neg_tags)