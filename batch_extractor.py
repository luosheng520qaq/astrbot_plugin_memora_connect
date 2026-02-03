"""
批量记忆提取模块
通过 LLM 从对话中提取记忆点和主题
"""
import json
import re
from datetime import datetime
from typing import Dict, List, Any
try:
    from astrbot.api import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


class BatchMemoryExtractor:
    """记忆提取器，通过LLM调用获取多个记忆点和主题"""
    
    def __init__(self, memory_system):
        self.memory_system = memory_system

    def _safe_load_json(self, text: str):
        try:
            return json.loads(text)
        except Exception:
            match = re.search(r"\{[\s\S]*\}", text)
            if not match:
                return None
            try:
                return json.loads(match.group(0))
            except Exception:
                return None
    
    async def extract_impressions_from_conversation(self, conversation_history: List[Dict[str, Any]], group_id: str) -> List[Dict[str, Any]]:
        """
        从对话中提取人物印象
        
        Args:
            conversation_history: 对话历史
            group_id: 群组ID
            
        Returns:
            人物印象列表
        """
        if not conversation_history:
            return []
        
        formatted_history = self._format_conversation_history(conversation_history)
        
        prompt = f"""请从以下对话中提取人物印象信息。

对话历史：
{formatted_history}

任务要求：
1. 识别对话中涉及的所有人物
2. 提取每个人物的印象描述
3. 为每个印象提供：
   - person_name: 人物姓名
   - summary: 印象摘要
   - score: 好感度分数（0-1）
   - details: 详细描述
   - confidence: 置信度（0-1）

返回格式：
{{
  "impressions": [
    {{
      "person_name": "张三",
      "summary": "友善且乐于助人",
      "score": 0.8,
      "details": "主动提供帮助，态度友好",
      "confidence": 0.9
    }}
  ]
}}

只返回JSON格式
"""

        try:
            provider = await self.memory_system.get_llm_provider()
            if not provider:
                return []
            
            response = await provider.text_chat(
                prompt=prompt,
                contexts=[],
                system_prompt="你是一个人物印象提取助手"
            )
            raw_text = (getattr(response, "completion_text", "") or "").strip()
            if not raw_text:
                return []
            data = self._safe_load_json(raw_text)
            if not isinstance(data, dict):
                return []
            impressions = data.get("impressions", [])
            
            # 过滤有效印象
            valid_impressions = []
            for impression in impressions:
                if impression.get("person_name") and impression.get("summary"):
                    valid_impressions.append({
                        "person_name": str(impression["person_name"]),
                        "summary": str(impression["summary"]),
                        "score": float(impression.get("score", 0.5)),
                        "details": str(impression.get("details", "")),
                        "confidence": float(impression.get("confidence", 0.7))
                    })
            
            return valid_impressions
            
        except Exception as e:
            logger.error(f"提取人物印象失败: {e}")
            return []
    
    async def extract_memories_and_themes(self, conversation_history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        通过LLM调用同时提取主题和记忆内容
        
        Args:
            conversation_history: 包含完整信息的对话历史，每项包含role, content, sender_name, timestamp
            
        Returns:
            包含主题和记忆内容的列表，每项包含theme, memory_content, confidence
        """
        if not conversation_history:
            return []
        
        # 构建包含完整信息的对话历史
        formatted_history = self._format_conversation_history(conversation_history)
        
        prompt = f"""请从以下对话中提取真正有价值的长期记忆。

对话历史：
{formatted_history}

任务要求：
1. **严格过滤**：忽略所有日常寒暄、无意义闲聊、简单问答、重复确认、指令交互等短期信息。
2. **只记录重要信息**：仅提取对长期相处有价值的事实、偏好、约定、重要经历或情感变化。
3. **合并同类项**：将碎片化的对话整合成完整的记忆条目，不要拆分成多条琐碎记忆。
4. **如果没有值得记录的信息，请返回空列表**。

记忆门槛（Confidence）：
- 1.0 (极重要)：重大人生事件、核心价值观、强烈情感宣泄、重要承诺。
- 0.8 (重要)：具体喜好（如"我不吃香菜"）、固定习惯、关键日程、人际关系变化。
- 0.6 (一般)：有回溯价值的趣事、具体的观点表达。
- **低于0.6的信息直接丢弃，不要生成记忆。**

返回字段说明：
- memory_type: "impression" (对人的评价/看法) 或 "normal" (事件/事实)
- theme: 核心主题词（如"饮食偏好", "工作规划"），包含人名
- content: 完整的记忆描述，包含时间、地点、人物、起因、经过、结果（使用第一人称"我"代表Bot）
- details: 补充细节
- confidence: 置信度评分 (0.6-1.0)

返回格式（JSON）：
{{
  "memories": [
    {{
      "theme": "饮食偏好",
      "content": "阿灿明确表示不吃香菜，因为觉得味道像肥皂",
      "details": "在讨论午餐选择时提到的",
      "participants": "我,阿灿",
      "location": "聊天",
      "emotion": "厌恶",
      "tags": "饮食,习惯",
      "confidence": 0.9,
      "memory_type": "normal"
    }}
  ]
}}
"""

        try:
            provider = await self.memory_system.get_llm_provider()
            if not provider:
                logger.warning("LLM提供商不可用，使用简单提取")
                return await self._fallback_extraction(conversation_history)
            
            try:
                response = await provider.text_chat(
                    prompt=prompt,
                    contexts=[],
                    system_prompt="你是一个专业的记忆提取助手，请准确提取对话中的关键信息。"
                )
                
                # 解析JSON响应
                result = self._parse_batch_response(response.completion_text)
                return result
                
            except Exception as e:
                # 网络错误或LLM服务不可用
                if "upstream" in str(e).lower() or "connection" in str(e).lower():
                    logger.warning(f"LLM服务连接失败，使用简单提取: {e}")
                else:
                    logger.error(f"LLM调用失败: {e}")
                return await self._fallback_extraction(conversation_history)
            
        except Exception as e:
            logger.error(f"批量记忆提取失败: {e}")
            return await self._fallback_extraction(conversation_history)
    
    def _format_conversation_history(self, history: List[Dict[str, Any]]) -> str:
        """格式化对话历史，包含完整信息，并区分Bot和用户发言"""
        formatted_lines = []
        for msg in history:
            content = msg.get('content', '')
            timestamp = msg.get('timestamp', '')
            role = msg.get('role', 'user')
            sender = msg.get('sender_name', '用户')
            
            # 格式化时间戳
            if isinstance(timestamp, (int, float)):
                dt = datetime.fromtimestamp(timestamp)
                time_str = dt.strftime('%m-%d %H:%M')
            else:
                time_str = str(timestamp)
            
            # 根据角色区分Bot和用户消息
            if role == "assistant":
                # Bot消息，标识为"[Bot]"
                formatted_lines.append(f"[{time_str}] [Bot]: {content}")
            else:
                # 用户消息，保持原格式
                formatted_lines.append(f"[{time_str}] {sender}: {content}")
        
        return "\n".join(formatted_lines)
    
    def _parse_batch_response(self, response_text: str) -> List[Dict[str, Any]]:
        """解析批量提取的LLM响应"""
        try:
            # 清理响应文本，处理中文引号和格式问题
            cleaned_text = response_text
            for old, new in [('"', '"'), ('"', '"'), (''', "'"), (''', "'"), ('，', ','), ('：', ':')]:
                cleaned_text = cleaned_text.replace(old, new)
            
            # 尝试多种JSON提取方式
            json_patterns = [
                r'\{[^{}]*"memories"[^{}]*\}',  # 简单JSON对象
                r'\{.*"memories"\s*:\s*\[.*\].*\}',  # 包含memories数组的完整对象
                r'\{.*\}',  # 最宽泛的匹配
            ]
            
            json_str = None
            for pattern in json_patterns:
                matches = re.findall(pattern, cleaned_text, re.DOTALL)
                if matches:
                    json_str = matches[-1]  # 取最后一个匹配
                    break
            
            if not json_str:
                return []
            
            # 修复常见的JSON格式问题
            json_str = re.sub(r',\s*}', '}', json_str)
            json_str = re.sub(r',\s*]', ']', json_str)
            json_str = re.sub(r'([{,]\s*)(\w+):', r'\1"\2":', json_str)  # 修复未加引号的键
            
            try:
                data = json.loads(json_str)
            except json.JSONDecodeError:
                # 更激进的修复，记录错误但不输出过多日志
                json_str = re.sub(r'([{,]\s*)"([^"]*)"\s*:\s*([^",}\]]+)([,\}])', r'\1"\2": "\3"\4', json_str)
                data = json.loads(json_str)
            
            memories = data.get("memories", [])
            if not isinstance(memories, list):
                return []
            
            # 过滤和验证记忆
            filtered_memories = []
            for i, mem in enumerate(memories):
                try:
                    if not isinstance(mem, dict):
                        continue
                    
                    # 安全地获取每个字段，确保类型正确
                    confidence = 0.7
                    try:
                        confidence_val = mem.get("confidence", 0.7)
                        if isinstance(confidence_val, (int, float)):
                            confidence = float(confidence_val)
                        elif isinstance(confidence_val, str):
                            confidence = float(confidence_val)
                    except (ValueError, TypeError):
                        confidence = 0.7
                    
                    theme = ""
                    try:
                        theme_val = mem.get("theme", "")
                        theme = str(theme_val).strip()
                    except (ValueError, TypeError):
                        theme = ""
                    
                    content = ""
                    try:
                        content_val = mem.get("content", "")
                        content = str(content_val).strip()
                    except (ValueError, TypeError):
                        content = ""
                    
                    details = ""
                    try:
                        details_val = mem.get("details", "")
                        details = str(details_val).strip()
                    except (ValueError, TypeError):
                        details = ""
                    
                    participants = ""
                    try:
                        participants_val = mem.get("participants", "")
                        participants = str(participants_val).strip()
                    except (ValueError, TypeError):
                        participants = ""
                    
                    location = ""
                    try:
                        location_val = mem.get("location", "")
                        location = str(location_val).strip()
                    except (ValueError, TypeError):
                        location = ""
                    
                    emotion = ""
                    try:
                        emotion_val = mem.get("emotion", "")
                        emotion = str(emotion_val).strip()
                    except (ValueError, TypeError):
                        emotion = ""
                    
                    tags = ""
                    try:
                        tags_val = mem.get("tags", "")
                        tags = str(tags_val).strip()
                    except (ValueError, TypeError):
                        tags = ""
                    
                    memory_type = "normal"
                    try:
                        memory_type_val = mem.get("memory_type", "normal")
                        memory_type = str(memory_type_val).strip().lower()
                    except (ValueError, TypeError):
                        memory_type = "normal"
                    
                    # 清理主题中的特殊字符
                    theme = re.sub(r'[^\w\u4e00-\u9fff,，]', '', theme)
                    
                    if theme and content and confidence > 0.3:
                        filtered_memories.append({
                            "theme": theme,
                            "content": content,
                            "details": details,
                            "participants": participants,
                            "location": location,
                            "emotion": emotion,
                            "tags": tags,
                            "confidence": max(0.0, min(1.0, confidence)),
                            "memory_type": memory_type if memory_type in ["normal", "impression"] else "normal"
                        })
                        
                except (ValueError, TypeError, AttributeError):
                    continue
            
            return filtered_memories
            
        except Exception:
            return []
    
    async def _fallback_extraction(self, history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """回退到简单提取模式"""
        if not history:
            return []
        
        # 简单关键词提取
        text = " ".join([msg.get('content', '') for msg in history])
        themes = self._extract_simple_themes(text)
        
        memories = []
        for theme in themes[:3]:
            memory_content = f"我们聊过关于{theme}的事情"
            memories.append({
                "theme": theme,
                "memory_content": memory_content,
                "confidence": 0.5
            })
        
        return memories
    
    def _extract_simple_themes(self, text: str) -> List[str]:
        """简单主题提取"""
        # 提取中文关键词
        words = re.findall(r'\b[\u4e00-\u9fff]{2,4}\b', text)
        word_freq = {}
        
        for word in words:
            if len(word) >= 2 and word not in ["你好", "谢谢", "再见"]:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # 返回频率最高的关键词
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in sorted_words[:5]]
