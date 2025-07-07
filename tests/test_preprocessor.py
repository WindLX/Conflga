import pytest
from conflga.preprocessor import ConflgaPreprocessor


class TestConflgaPreprocessor:
    """测试 ConflgaPreprocessor 类的功能"""

    def test_parse_macros_simple(self):
        """测试简单的宏定义解析"""
        preprocessor = ConflgaPreprocessor()
        text = """
#define PORT = 8080
#define HOST = "localhost"
#define DEBUG = True
#define ENABLED = true
"""
        preprocessor._parse_macros(text)
        assert preprocessor.conflga_vars["PORT"] == 8080
        assert preprocessor.conflga_vars["HOST"] == "localhost"
        assert preprocessor.conflga_vars["DEBUG"] is True
        assert preprocessor.conflga_vars["ENABLED"] is True

    def test_parse_macros_with_expressions(self):
        """测试包含表达式的宏定义"""
        preprocessor = ConflgaPreprocessor()
        text = """
#define BASE_PORT = 8000
#define API_PORT = BASE_PORT + 80
#define WEB_PORT = BASE_PORT + 3000
#define TOTAL_PORTS = API_PORT + WEB_PORT
"""
        preprocessor._parse_macros(text)
        assert preprocessor.conflga_vars["BASE_PORT"] == 8000
        assert preprocessor.conflga_vars["API_PORT"] == 8080
        assert preprocessor.conflga_vars["WEB_PORT"] == 11000
        assert preprocessor.conflga_vars["TOTAL_PORTS"] == 19080

    def test_parse_macros_string_operations(self):
        """测试字符串操作的宏定义"""
        preprocessor = ConflgaPreprocessor()
        text = """
#define APP_NAME = "MyApp"
#define VERSION = "1.0"
#define FULL_NAME = APP_NAME + " v" + VERSION
"""
        preprocessor._parse_macros(text)
        assert preprocessor.conflga_vars["APP_NAME"] == "MyApp"
        assert preprocessor.conflga_vars["VERSION"] == "1.0"
        assert preprocessor.conflga_vars["FULL_NAME"] == "MyApp v1.0"

    def test_parse_macros_invalid_expression(self):
        """测试无效表达式的宏定义"""
        preprocessor = ConflgaPreprocessor()
        text = "#define INVALID = undefined_var"

        with pytest.raises(RuntimeError, match="Marcos compute failed"):
            preprocessor._parse_macros(text)

    def test_parse_macros_order_dependency(self):
        """测试宏定义的顺序依赖性"""
        preprocessor = ConflgaPreprocessor()
        text = """
#define A = 10
#define B = A * 2
#define C = B + A
"""
        preprocessor._parse_macros(text)
        assert preprocessor.conflga_vars["A"] == 10
        assert preprocessor.conflga_vars["B"] == 20
        assert preprocessor.conflga_vars["C"] == 30

    def test_evaluate_template_expr_simple(self):
        """测试简单的模板表达式求值"""
        preprocessor = ConflgaPreprocessor()
        preprocessor.conflga_vars = {"PORT": 8080, "HOST": "localhost"}

        line = "server = {{ HOST }}:{{ PORT }}"
        result = preprocessor._evaluate_template_expr(line)
        assert result == "server = localhost:8080"

    def test_evaluate_template_expr_with_expressions(self):
        """测试包含表达式的模板求值"""
        preprocessor = ConflgaPreprocessor()
        preprocessor.conflga_vars = {"BASE": 8000, "OFFSET": 80}

        line = "port = {{ BASE + OFFSET }}"
        result = preprocessor._evaluate_template_expr(line)
        assert result == "port = 8080"

    def test_evaluate_template_expr_string_operations(self):
        """测试字符串操作的模板求值"""
        preprocessor = ConflgaPreprocessor()
        preprocessor.conflga_vars = {"NAME": "app", "ENV": "prod"}

        line = 'image = "{{ NAME }}-{{ ENV }}:latest"'
        result = preprocessor._evaluate_template_expr(line)
        assert result == 'image = "app-prod:latest"'

    def test_evaluate_template_expr_multiple_templates(self):
        """测试一行中的多个模板表达式"""
        preprocessor = ConflgaPreprocessor()
        preprocessor.conflga_vars = {"HOST": "localhost", "PORT": 8080, "PATH": "/api"}

        line = "url = http://{{ HOST }}:{{ PORT }}{{ PATH }}"
        result = preprocessor._evaluate_template_expr(line)
        assert result == "url = http://localhost:8080/api"

    def test_evaluate_template_expr_no_templates(self):
        """测试没有模板表达式的行"""
        preprocessor = ConflgaPreprocessor()
        line = "simple_config = true"
        result = preprocessor._evaluate_template_expr(line)
        assert result == "simple_config = true"

    def test_evaluate_template_expr_invalid(self):
        """测试无效的模板表达式"""
        preprocessor = ConflgaPreprocessor()
        preprocessor.conflga_vars = {"PORT": 8080}

        line = "host = {{ undefined_var }}"
        with pytest.raises(RuntimeError, match="Expression evaluation failed"):
            preprocessor._evaluate_template_expr(line)

    def test_preprocess_text_complete_workflow(self):
        """测试完整的预处理工作流程"""
        preprocessor = ConflgaPreprocessor()
        toml_text = """
#define APP_NAME = "MyApp"
#define VERSION = "1.0.0"
#define PORT = 8080
#define WORKERS = 4

[app]
name = "{{ APP_NAME }}"
version = "{{ VERSION }}"

[server]
port = {{ PORT }}
workers = {{ WORKERS }}
address = "0.0.0.0:{{ PORT }}"

[database]
max_connections = {{ WORKERS * 10 }}
"""

        result = preprocessor.preprocess_text(toml_text)
        expected = """
[app]
name = "MyApp"
version = "1.0.0"

[server]
port = 8080
workers = 4
address = "0.0.0.0:8080"

[database]
max_connections = 40
"""
        assert result.strip() == expected.strip()

    def test_preprocess_text_removes_macro_lines(self):
        """测试预处理过程中移除宏定义行"""
        preprocessor = ConflgaPreprocessor()
        toml_text = """
#define DEBUG = True
# This is a comment
config = {{ DEBUG }}
#define ANOTHER = "value"
another_config = "{{ ANOTHER }}"
"""

        result = preprocessor.preprocess_text(toml_text)
        lines = result.strip().split("\n")

        # 确保没有 #define 行
        for line in lines:
            assert not line.strip().startswith("#define")

        # 确保配置行被正确处理
        assert "config = true" in result
        assert 'another_config = "value"' in result
        assert "# This is a comment" in result  # 普通注释应该保留

    def test_preprocess_text_complex_expressions(self):
        """测试复杂表达式的预处理"""
        preprocessor = ConflgaPreprocessor()
        toml_text = """
#define BASE_URL = "https://api.example.com"
#define VERSION = "v1"
#define TIMEOUT = 30
#define RETRY_COUNT = 3

[api]
endpoint = "{{ BASE_URL }}/{{ VERSION }}"
timeout = {{ TIMEOUT }}
max_retries = {{ RETRY_COUNT }}
total_timeout = {{ TIMEOUT * RETRY_COUNT }}

[features]
enabled = {{ TIMEOUT > 10 }}
"""

        result = preprocessor.preprocess_text(toml_text)

        assert 'endpoint = "https://api.example.com/v1"' in result
        assert "timeout = 30" in result
        assert "max_retries = 3" in result
        assert "total_timeout = 90" in result
        assert "enabled = true" in result

    def test_preprocess_text_empty_input(self):
        """测试空输入的处理"""
        preprocessor = ConflgaPreprocessor()
        result = preprocessor.preprocess_text("")
        assert result == ""

    def test_preprocess_text_no_macros_or_templates(self):
        """测试没有宏或模板的普通 TOML"""
        preprocessor = ConflgaPreprocessor()
        toml_text = """
[app]
name = "simple_app"
port = 8080

[database]
host = "localhost"
"""

        result = preprocessor.preprocess_text(toml_text)
        assert result.strip() == toml_text.strip()

    def test_get_vars(self):
        """测试获取变量字典"""
        preprocessor = ConflgaPreprocessor()
        preprocessor.conflga_vars = {"A": 1, "B": "test"}

        vars_dict = preprocessor.get_vars()
        assert vars_dict == {"A": 1, "B": "test"}

        # 确保返回的是副本（或者至少可以安全访问）
        vars_dict["C"] = 3
        # 这不应该影响原始的 conflga_vars（除非故意设计为引用）

    def test_macro_pattern_regex(self):
        """测试宏定义的正则表达式模式"""
        pattern = ConflgaPreprocessor.MACRO_PATTERN

        # 测试有效的宏定义
        assert pattern.match("#define VAR = 123")
        assert pattern.match("#define VAR=123")
        assert pattern.match("#define VAR_NAME = 'value'")
        assert pattern.match("   #define SPACED = True   ")

        # 测试无效的宏定义
        assert not pattern.match("define VAR = 123")  # 缺少 #
        assert not pattern.match("#define")  # 不完整
        assert not pattern.match("# define VAR = 123")  # 空格在错误位置

    def test_template_pattern_regex(self):
        """测试模板表达式的正则表达式模式"""
        pattern = ConflgaPreprocessor.TEMPLATE_PATTERN

        # 测试有效的模板表达式
        assert pattern.search("{{ var }}")
        assert pattern.search("prefix {{ var }} suffix")
        assert pattern.search("{{ complex_expression + 1 }}")
        assert pattern.search("{{ 'string' + var }}")

        # 测试无效的模板表达式
        assert not pattern.search("{ var }")  # 单括号
        assert not pattern.search("{{var")  # 不完整
        assert not pattern.search("var}}")  # 不完整

    def test_preprocessor_state_isolation(self):
        """测试预处理器实例之间的状态隔离"""
        preprocessor1 = ConflgaPreprocessor()
        preprocessor2 = ConflgaPreprocessor()

        preprocessor1._parse_macros("#define VAR1 = 123")
        preprocessor2._parse_macros("#define VAR2 = 456")

        assert "VAR1" in preprocessor1.conflga_vars
        assert "VAR1" not in preprocessor2.conflga_vars
        assert "VAR2" in preprocessor2.conflga_vars
        assert "VAR2" not in preprocessor1.conflga_vars

    def test_whitespace_handling(self):
        """测试空白字符的处理"""
        preprocessor = ConflgaPreprocessor()
        toml_text = """

#define SPACED = 123


config = {{ SPACED }}


"""

        result = preprocessor.preprocess_text(toml_text)
        lines = result.split("\n")

        # 确保空行被保留
        assert "" in lines
        # 确保配置被正确处理
        assert "config = 123" in result

    def test_edge_cases_and_error_handling(self):
        """测试边界情况和错误处理"""
        preprocessor = ConflgaPreprocessor()

        # 测试包含语法错误的表达式
        preprocessor.conflga_vars = {"VAR": 123}
        with pytest.raises(RuntimeError):
            preprocessor._evaluate_template_expr("value = {{ VAR + }}")

    def test_nested_expressions(self):
        """测试嵌套和复杂的表达式（受 simple_eval 限制）"""
        preprocessor = ConflgaPreprocessor()
        text = """
#define BASE_VALUE = 10
#define MULTIPLIER = 3
#define RESULT = BASE_VALUE * MULTIPLIER
"""
        preprocessor._parse_macros(text)

        assert preprocessor.conflga_vars["BASE_VALUE"] == 10
        assert preprocessor.conflga_vars["MULTIPLIER"] == 3
        assert preprocessor.conflga_vars["RESULT"] == 30

        # 测试在模板中使用
        line = "base = {{ BASE_VALUE }}, result = {{ RESULT }}"
        result = preprocessor._evaluate_template_expr(line)
        assert result == "base = 10, result = 30"

    def test_boolean_and_comparison_operations(self):
        """测试布尔值和比较操作"""
        preprocessor = ConflgaPreprocessor()
        text = """
#define DEBUG = True
#define PORT = 8080
#define IS_DEV = PORT == 8080
#define IS_PROD = not IS_DEV
"""
        preprocessor._parse_macros(text)

        assert preprocessor.conflga_vars["DEBUG"] is True
        assert preprocessor.conflga_vars["IS_DEV"] is True
        assert preprocessor.conflga_vars["IS_PROD"] is False

        # 在模板中使用布尔值
        line = "debug_mode = {{ DEBUG and IS_DEV }}"
        result = preprocessor._evaluate_template_expr(line)
        assert result == "debug_mode = true"

    def test_base_names_support(self):
        """测试基本名称的支持"""
        preprocessor = ConflgaPreprocessor()
        text = """
#define ENABLED = true
#define DISABLED = false
#define NOTHING = null
"""
        preprocessor._parse_macros(text)

        assert preprocessor.conflga_vars["ENABLED"] is True
        assert preprocessor.conflga_vars["DISABLED"] is False
        assert preprocessor.conflga_vars["NOTHING"] is None

        # 测试在模板中使用
        line = "enabled = {{ ENABLED }}, disabled = {{ DISABLED }}, nothing = {{ NOTHING }}"
        result = preprocessor._evaluate_template_expr(line)
        assert result == "enabled = true, disabled = false, nothing = @None"
