import re
import math
import random
from typing import Any

from simpleeval import simple_eval

from .console import debug


class ConflgaPreprocessor:
    _macro_pattern = re.compile(r"^\s*#let\s+(\w+)\s*=\s*(.+)")
    _template_pattern = re.compile(r"\{\{([^}]+)\}\}")
    _custom_functions = {
        # --- 数学计算 ---
        "abs": abs,  # 绝对值: abs(-5) -> 5
        "max": max,  # 最大值: max(2, 8, 5) -> 8
        "min": min,  # 最小值: min(2, 8, 5) -> 2
        "pow": pow,  # 幂计算: pow(2, 3) -> 8
        "round": round,  # 四舍五入: round(3.1415, 2) -> 3.14
        "sqrt": math.sqrt,  # 平方根: sqrt(16) -> 4.0
        "log": math.log,  # 自然对数: log(10) -> 2.302...
        "log10": math.log10,  # 以10为底的对数: log10(100) -> 2.0
        "exp": math.exp,  # e的指数: exp(2) -> 7.389...
        "sin": math.sin,  # 正弦: sin(0) -> 0.0
        "cos": math.cos,  # 余弦: cos(0) -> 1.0
        "tan": math.tan,  # 正切: tan(0) -> 0.0
        "asin": math.asin,  # 反正弦: asin(0) -> 0.0
        "acos": math.acos,  # 反余弦: acos(1) -> 0.0
        "atan": math.atan,  # 反正切: atan(0) -> 0.0
        "degrees": math.degrees,  # 弧度转角度: degrees(0) -> 0.0
        "radians": math.radians,  # 角度转弧度: radians(180) -> 3.141592653589793
        "tanh": math.tanh,  # 双曲正切: tanh(0) -> 0.0
        "ceil": math.ceil,  # 向上取整: ceil(3.2) -> 4
        "floor": math.floor,  # 向下取整: floor(3.8) -> 3
        "gcd": math.gcd,  # 最大公约数: gcd(8, 12) -> 4
        "lcm": math.lcm,  # 最小公倍数: lcm(4, 6) -> 12
        # --- 随机数 ---
        "random": lambda: random.random(),  # 生成0到1之间的随机数: random()
        "randint": lambda a, b: random.randint(
            a, b
        ),  # 生成指定范围内的随机整数: randint(1, 10) -> 5
        "normal": lambda mu, sigma: random.gauss(
            mu, sigma
        ),  # 生成正态分布随机数: normal(0, 1) -> 0.5
        "uniform": lambda a, b: random.uniform(
            a, b
        ),  # 生成指定范围内的随机浮点数: uniform(1, 10) -> 5.5
        "choice": lambda seq: random.choice(
            seq
        ),  # 从序列中随机选择一个元素: choice([1, 2, 3]) -> 2
        "shuffle": lambda seq: random.shuffle(seq)
        or seq,  # 打乱序列顺序: shuffle([1, 2, 3]) -> [3, 1]
        # --- 类型与数据结构 ---
        "len": len,  # 获取长度: len([1, 2, 3]) -> 3
        "str": str,  # 转换为字符串: str(123) -> '123'
        "int": int,  # 转换为整数: int('123') -> 123
        "float": float,  # 转换为浮点数: float('3.14') -> 3.14
        "range": range,  # 生成整数序列: range(1, 10) -> [1, 2, ..., 9]
        "sum": sum,  # 求和: sum([1, 2, 3]) -> 6
    }

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
            match = self._macro_pattern.match(line.strip())
            if match:
                key, expr = match.groups()
                try:
                    # 合并基本名称和已定义的变量
                    eval_names = {**self.base_names, **self.conflga_vars}
                    val = simple_eval(
                        expr,
                        names=eval_names,
                        functions=self._custom_functions,
                    )
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
                result = simple_eval(
                    expr,
                    names=eval_names,
                    functions=self._custom_functions,
                )
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
        new_line = self._template_pattern.sub(replacer, line)
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
            if not self._macro_pattern.match(line.strip())
        ]

        evaluated_lines = [self._evaluate_template_expr(line) for line in config_lines]

        return "\n".join(evaluated_lines)

    def get_vars(self) -> dict[str, Any]:
        return self.conflga_vars
