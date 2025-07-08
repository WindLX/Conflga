import re
from simpleeval import simple_eval
from typing import Any

from .console import debug


class ConflgaPreprocessor:
    MACRO_PATTERN = re.compile(r"^\s*#let\s+(\w+)\s*=\s*(.+)")
    TEMPLATE_PATTERN = re.compile(r"\{\{([^}]+)\}\}")

    def __init__(self):
        self.conflga_vars: dict[str, Any] = {}
        # 添加基本的布尔值和 None 支持
        self.base_names = {
            "true": True,
            "True": True,
            "false": False,
            "False": False,
            "null": None,
            "None": None,
            "@None": None,  # 支持 @None 语法
        }

    def _parse_macros(self, text: str) -> None:
        """
        解析并计算所有 #let 宏定义。
        支持变量引用（先后顺序敏感）
        """
        lines = text.splitlines()
        for line in lines:
            match = self.MACRO_PATTERN.match(line.strip())
            if match:
                key, expr = match.groups()
                try:
                    # 合并基本名称和已定义的变量
                    eval_names = {**self.base_names, **self.conflga_vars}
                    val = simple_eval(expr, names=eval_names)
                    self.conflga_vars[key] = val
                    debug(f"Macro defined: {key} = {val}")
                except Exception as e:
                    raise RuntimeError(
                        f"Marcos compute failed at line {lines.index(line)+1}: {key} = {expr} -> {e}"
                    )

    def _evaluate_template_expr(self, line: str) -> str:
        """
        处理 {{ expr }} 表达式，基于 self.conflga_vars 上下文。
        如果 {{}} 前后都有引号，且表达式求值结果不是 str 或 None，则去掉前后的引号。
        """

        def replacer(match):
            expr = match.group(1).strip()
            start, end = match.span()
            # 检查前后是否有引号
            before = line[start - 1] if start - 1 >= 0 else ""
            after = line[end] if end < len(line) else ""
            try:
                eval_names = {**self.base_names, **self.conflga_vars}
                result = simple_eval(expr, names=eval_names)
                # TOML 布尔值小写
                if isinstance(result, bool):
                    result_str = str(result).lower()
                elif result is None:
                    result_str = "@None"
                else:
                    result_str = str(result)
                debug(f"Evaluated expression: {{{{ {expr} }}}} -> {result}")

                # 检查是否需要去除引号
                if (
                    before == after
                    and before in ("'", '"')
                    and not isinstance(result, (str, type(None)))
                ):
                    # 去掉前后的引号
                    # 替换时，返回特殊标记，后续统一处理
                    return f"__REMOVE_QUOTE__{result_str}__REMOVE_QUOTE__"
                return result_str
            except Exception as e:
                raise RuntimeError(
                    f"Expression evaluation failed at line {line}: {{{{ {expr} }}}} -> {e}"
                )

        # 先替换模板表达式
        new_line = self.TEMPLATE_PATTERN.sub(replacer, line)
        # 再去掉特殊标记包裹的引号
        new_line = re.sub(
            r"(['\"])\s*__REMOVE_QUOTE__(.*?)__REMOVE_QUOTE__\s*\1", r"\2", new_line
        )
        return new_line

    def preprocess_text(self, toml_text: str) -> str:
        """
        预处理一个带宏的 TOML 字符串，返回干净的 TOML 字符串
        """
        self._parse_macros(toml_text)

        # 去掉宏行，只保留配置内容
        config_lines = [
            line
            for line in toml_text.splitlines()
            if not self.MACRO_PATTERN.match(line.strip())
        ]

        evaluated_lines = [self._evaluate_template_expr(line) for line in config_lines]

        return "\n".join(evaluated_lines)

    def get_vars(self) -> dict[str, Any]:
        return self.conflga_vars
