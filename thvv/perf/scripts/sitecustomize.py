"""Python 启动自动加载（sitecustomize）——兼容性补丁。

补丁 1：evalscope 在 transformers >= 5.x 下的 token 计数错误
------------------------------------------------------------
transformers 5.x 的 ``apply_chat_template(tokenize=True)`` 返回值从旧的
``List[int]`` 变成了 ``BatchEncoding``（dict-like，含 input_ids / attention_mask
等键）。旧版 evalscope 的 ``tokenize_chat_messages`` 直接对其 ``len()``，得到的是
键数（2）而非 token 数，导致 ``check_prompt_length`` 把全部 prompt 过滤掉，最终
报 ``Dataset "custom_multi_turn" produced no conversations!``。

本模块在 evalscope 导入后，把 ``tokenize_chat_messages`` 替换为对返回值做归一化
的版本，始终返回平坦的 ``List[int]``。

补丁 2：tokenizer 未提供 chat_template 时的兜底
------------------------------------------------
部分模型的 tokenizer 仓库（如自建/内部 deepseek-v4-pro 分词器）只带 vocab 与
special tokens，``tokenizer_config.json`` 里没有 ``chat_template``，此时
``apply_chat_template`` 直接抛：

    ValueError: Cannot use chat template functions because tokenizer.chat_template
    is not set and no template argument was passed!

压测只需要「按聊天格式估算 prompt token 数」用于档位过滤，因此这里在检测到模板
缺失时，按 DeepSeek 风格注入一个最小模板（``<|role|>content<|end|>``），使统计
口径与 server 返回的 usage.prompt_tokens 量级一致，避免整档样本被过滤掉。

对未安装 evalscope 的环境静默跳过，无副作用。
"""
try:
    import evalscope.perf.plugin.datasets.utils as _u
except Exception:  # 无 evalscope 的环境，跳过
    pass
else:

    # tokenizer 缺 chat_template 时的兜底模板（DeepSeek 系列格式，通用性最好）
    _DEFAULT_CHAT_TEMPLATE = (
        "{% for message in messages %}"
        "<|{{ message['role'] }}|>{{ message['content'] }}<|end|>"
        "{% endfor %}"
        "{% if add_generation_prompt %}<|assistant|>{% endif %}"
    )

    def _normalize(result):
        """把 apply_chat_template 的返回值归一化为平坦的 List[int]；无法识别返回 None。"""
        if isinstance(result, dict) and 'input_ids' in result:
            ids = result['input_ids']
        elif hasattr(result, 'input_ids'):
            ids = result.input_ids
        elif isinstance(result, list):
            ids = result
        else:
            return None
        # tensor -> list
        ids = ids.tolist() if hasattr(ids, 'tolist') else list(ids)
        # 去掉意外多出的 batch 维度：[[ids]] -> [ids]
        if ids and isinstance(ids[0], list):
            return ids[0]
        return ids

    def _flatten_content(messages):
        """多模态消息的 content 可能是 [{type:text,text:...}] 列表，压成纯文本再套模板。"""
        out = []
        for msg in messages:
            if not isinstance(msg, dict):
                out.append(msg)
                continue
            content = msg.get('content')
            if isinstance(content, list):
                parts = []
                for part in content:
                    if isinstance(part, str):
                        parts.append(part)
                    elif isinstance(part, dict):
                        parts.append(part.get('text') or '')
                msg = {**msg, 'content': ''.join(parts)}
            out.append(msg)
        return out

    def _ensure_chat_template(tokenizer):
        """tokenizer 没有 chat_template 时注入兜底模板（幂等）。"""
        if getattr(tokenizer, 'chat_template', None):
            return
        try:
            tokenizer.chat_template = _DEFAULT_CHAT_TEMPLATE
        except Exception:  # 只读属性等极端情况，交给后续逻辑抛原始错误
            pass

    _orig = _u.tokenize_chat_messages

    def _patched(tokenizer, messages, add_generation_prompt=True):
        _ensure_chat_template(tokenizer)
        messages = _flatten_content(messages)
        result = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=add_generation_prompt
        )
        normalized = _normalize(result)
        if normalized is not None:
            return normalized
        # 兜底：交给原实现处理
        return _orig(tokenizer, messages, add_generation_prompt)

    _u.tokenize_chat_messages = _patched

    # 重要：evalscope 内部多处使用 ``from ...utils import tokenize_chat_messages``，
    # 名字在导入时就已绑定到旧函数。本模块导入 utils 时，父包 ``datasets`` 的
    # ``__init__`` 会预先导入 base/random_dataset 等模块，它们已经持有旧引用，
    # 只改 utils 属性对它们无效。因此把已加载模块里绑定到旧函数的引用一并替换。
    import sys as _sys

    for _mod in list(_sys.modules.values()):
        try:
            if getattr(_mod, 'tokenize_chat_messages', None) is _orig:
                _mod.tokenize_chat_messages = _patched
        except Exception:
            continue
