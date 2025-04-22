
from typing import List, Optional, Union, Type
import json
import re
import asyncio

from deepeval.test_case import LLMTestCase, ConversationalTestCase
from deepeval.metrics.base_metric import BaseMetric
from deepeval.models import DeepEvalBaseLLM
from deepeval.metrics.utils import initialize_model
from pydantic import BaseModel


class FluentVerdict(BaseModel):
    issue: str
    severity: str
    explanation: str


class FluentVerdicts(BaseModel):
    verdicts: List[FluentVerdict]


class FluentReason(BaseModel):
    reason: str


class FluencyTemplate:
    @staticmethod
    def analyze_text(actual_output: str):
        return f"""Analyze the following text for fluency issues like grammar errors, awkward phrasing, poor sentence structure, unnatural language, or repetitive and redundant expressions. Identify specific issues and categorize their severity as "high", "medium", or "low".

**
IMPORTANT: Please make sure to only return in JSON format, with the "verdicts" key mapping to a list of objects with "issue", "severity", and "explanation" fields. No words or explanation needed.
**

Example:
Text: 
I has been trying to solve this problems for many hours but it don't work. Can someone help me figure out what wrong with my code? I getting an error message every time I running it.

Example JSON:
{{
    "verdicts": [
        {{
            "issue": "Subject-verb agreement",
            "severity": "high",
            "explanation": "Multiple instances ('I has', 'it don't') where the subject and verb don't agree in number"
        }},
        {{
            "issue": "Missing articles",
            "severity": "medium", 
            "explanation": "Missing 'the' or 'a' in phrases like 'what wrong with' and 'getting an error'"
        }},
        {{
            "issue": "Incorrect verb forms",
            "severity": "medium",
            "explanation": "Inappropriate verb forms in 'I getting' and 'I running' instead of 'I am getting' and 'I am running'"
        }}
    ]
}}
===== END OF EXAMPLE ======

Text:
{actual_output}

JSON:
"""

    @staticmethod
    def generate_reason(verdicts: List[FluentVerdict], score: float):
        verdict_strs = []
        for v in verdicts:
            verdict_strs.append(f"{v.issue} ({v.severity}): {v.explanation}")
            
        return f"""Based on the fluency analysis and score, provide a concise summary explaining the fluency level. Explain why the text received this score, highlighting major strengths and weaknesses in its fluency.

**
IMPORTANT: Please make sure to only return in JSON format, with the 'reason' key providing the reason.
Example JSON:
{{
    "reason": "The text has [fluency assessment] because [specific reasons]."
}}
**

Fluency Score: {score}

Fluency Issues Identified:
{verdict_strs}

JSON:
"""


class FluencyMetric(BaseMetric):
    _required_params = [
        "actual_output",
    ]

    def __init__(
        self,
        threshold: float = 0.7,
        model: Optional[Union[str, DeepEvalBaseLLM]] = None,
        include_reason: bool = True,
        async_mode: bool = True,
        strict_mode: bool = False,
        verbose_mode: bool = False,
        evaluation_template: Type[FluencyTemplate] = FluencyTemplate,
    ):
        self.threshold = 1 if strict_mode else threshold
        self.model, self.using_native_model = initialize_model(model)
        self.evaluation_model = self.model.get_model_name()
        self.include_reason = include_reason
        self.async_mode = async_mode
        self.strict_mode = strict_mode
        self.verbose_mode = verbose_mode
        self.evaluation_template = evaluation_template
        self.error = None
        self.verdicts = []
        self.score = None
        self.reason = None
        self.success = None

    def measure(
        self,
        test_case: Union[LLMTestCase, ConversationalTestCase],
        _show_indicator: bool = True,
    ) -> float:
        if isinstance(test_case, ConversationalTestCase):
            test_case = test_case.turns[-1]
        
        # Check if required parameters exist
        self._check_parameters(test_case)

        self.evaluation_cost = 0 if self.using_native_model else None
        
        try:
            if self.async_mode:
                loop = asyncio.get_event_loop()
                self.score = loop.run_until_complete(
                    self.a_measure(test_case, _show_indicator=False)
                )
            else:
                self.verdicts = self._analyze_text(test_case.actual_output)
                self.score = self._calculate_score()
                self.reason = self._generate_reason()
                self.success = self.score >= self.threshold
            
            return self.score
        except Exception as e:
            self.error = str(e)
            return 0.0

    async def a_measure(
        self,
        test_case: Union[LLMTestCase, ConversationalTestCase],
        _show_indicator: bool = True,
    ) -> float:
        if isinstance(test_case, ConversationalTestCase):
            test_case = test_case.turns[-1]
        
        # Check if required parameters exist
        self._check_parameters(test_case)

        self.evaluation_cost = 0 if self.using_native_model else None
        
        try:
            self.verdicts = await self._a_analyze_text(test_case.actual_output)
            self.score = self._calculate_score()
            self.reason = await self._a_generate_reason()
            self.success = self.score >= self.threshold
            
            return self.score
        except Exception as e:
            self.error = str(e)
            return 0.0
    
    def _check_parameters(self, test_case):
        """Custom parameter checking"""
        for param in self._required_params:
            if not hasattr(test_case, param) or getattr(test_case, param) is None:
                error_msg = f"Missing required parameter: {param}"
                self.error = error_msg
                raise ValueError(error_msg)

    def _parse_response(self, res):
        """Handle different response types and extract data"""
        try:
            # Handle dictionary response
            if isinstance(res, dict):
                return res
            
            # Handle object with verdicts attribute
            if hasattr(res, 'verdicts'):
                return {'verdicts': res.verdicts}
                
            # Handle object with reason attribute
            if hasattr(res, 'reason'):
                return {'reason': res.reason}
                
            # Handle string response
            if isinstance(res, str):
                # Find JSON in string
                json_start = res.find('{')
                json_end = res.rfind('}') + 1
                
                if json_start != -1 and json_end > json_start:
                    json_str = res[json_start:json_end]
                    return json.loads(json_str)
            
            # Default empty result
            return {"verdicts": []}
        except Exception:
            return {"verdicts": []}

    def _analyze_text(self, actual_output: str) -> List[FluentVerdict]:
        prompt = self.evaluation_template.analyze_text(actual_output=actual_output)
        
        try:
            # Generate response
            if self.using_native_model:
                res, cost = self.model.generate(prompt, schema=FluentVerdicts)
                self.evaluation_cost += cost
            else:
                res = self.model.generate(prompt)
                
            # Parse response
            data = self._parse_response(res)
            
            # Process verdicts
            if "verdicts" in data and isinstance(data["verdicts"], list):
                result = []
                for item in data["verdicts"]:
                    if isinstance(item, dict) and "issue" in item and "severity" in item and "explanation" in item:
                        result.append(FluentVerdict(
                            issue=item["issue"],
                            severity=item["severity"],
                            explanation=item["explanation"]
                        ))
                    elif hasattr(item, "issue") and hasattr(item, "severity") and hasattr(item, "explanation"):
                        result.append(item)
                return result
            return []
        except Exception:
            return []

    async def _a_analyze_text(self, actual_output: str) -> List[FluentVerdict]:
        prompt = self.evaluation_template.analyze_text(actual_output=actual_output)
        
        try:
            # Generate response
            if self.using_native_model:
                res, cost = await self.model.a_generate(prompt, schema=FluentVerdicts)
                self.evaluation_cost += cost
            else:
                res = await self.model.a_generate(prompt)
                
            # Parse response
            data = self._parse_response(res)
            
            # Process verdicts
            if "verdicts" in data and isinstance(data["verdicts"], list):
                result = []
                for item in data["verdicts"]:
                    if isinstance(item, dict) and "issue" in item and "severity" in item and "explanation" in item:
                        result.append(FluentVerdict(
                            issue=item["issue"],
                            severity=item["severity"],
                            explanation=item["explanation"]
                        ))
                    elif hasattr(item, "issue") and hasattr(item, "severity") and hasattr(item, "explanation"):
                        result.append(item)
                return result
            return []
        except Exception:
            return []

    def _generate_reason(self) -> str:
        if not self.include_reason or not self.verdicts:
            return "The text is well-written and fluent with no significant issues detected."

        prompt = self.evaluation_template.generate_reason(
            verdicts=self.verdicts,
            score=format(self.score, ".2f"),
        )

        try:
            # Generate response
            if self.using_native_model:
                res, cost = self.model.generate(prompt, schema=FluentReason)
                self.evaluation_cost += cost
            else:
                res = self.model.generate(prompt)
                
            # Parse response
            data = self._parse_response(res)
            
            # Extract reason
            if "reason" in data and isinstance(data["reason"], str):
                return data["reason"]
            
            # Default reason based on verdicts
            if not self.verdicts:
                return "The text is well-written and fluent with no significant issues detected."
            else:
                issues = [f"{v.issue} ({v.severity})" for v in self.verdicts]
                return f"The text received a score of {self.score:.2f} due to the following issues: {', '.join(issues)}."
        except Exception:
            return "No specific issues were found in the text."

    async def _a_generate_reason(self) -> str:
        if not self.include_reason or not self.verdicts:
            return "The text is well-written and fluent with no significant issues detected."

        prompt = self.evaluation_template.generate_reason(
            verdicts=self.verdicts,
            score=format(self.score, ".2f"),
        )

        try:
            # Generate response
            if self.using_native_model:
                res, cost = await self.model.a_generate(prompt, schema=FluentReason)
                self.evaluation_cost += cost
            else:
                res = await self.model.a_generate(prompt)
                
            # Parse response
            data = self._parse_response(res)
            
            # Extract reason
            if "reason" in data and isinstance(data["reason"], str):
                return data["reason"]
            
            # Default reason based on verdicts
            if not self.verdicts:
                return "The text is well-written and fluent with no significant issues detected."
            else:
                issues = [f"{v.issue} ({v.severity})" for v in self.verdicts]
                return f"The text received a score of {self.score:.2f} due to the following issues: {', '.join(issues)}."
        except Exception:
            return "No specific issues were found in the text."

    def _calculate_score(self) -> float:
        if not self.verdicts:
            return 1.0  # Perfect score if no issues found
        
        # Calculate score based on severity of issues
        severity_weights = {
            "high": 0.4,
            "medium": 0.2,
            "low": 0.1
        }
        
        # Get counts by severity
        high_count = sum(1 for v in self.verdicts if v.severity.lower() == "high")
        medium_count = sum(1 for v in self.verdicts if v.severity.lower() == "medium")
        low_count = sum(1 for v in self.verdicts if v.severity.lower() == "low")
        
        # Calculate penalty
        penalty = (high_count * severity_weights["high"] + 
                   medium_count * severity_weights["medium"] + 
                   low_count * severity_weights["low"])
        
        # Ensure penalty doesn't exceed 1.0
        penalty = min(1.0, penalty)
        
        # Calculate score (1.0 = perfect, 0.0 = worst)
        score = 1.0 - penalty
        
        # If strict mode and below threshold, return 0
        return 0 if self.strict_mode and score < self.threshold else score

    def is_successful(self) -> bool:
        if self.error is not None:
            self.success = False
        else:
            try:
                self.success = self.score >= self.threshold
            except:
                self.success = False
        return self.success

    @property
    def __name__(self):
        return "Fluency"
