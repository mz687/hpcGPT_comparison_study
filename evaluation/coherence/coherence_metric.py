from typing import List, Optional, Union, Type
import json
import re

from deepeval.test_case import LLMTestCase, ConversationalTestCase
from deepeval.metrics.base_metric import BaseMetric
from deepeval.models import DeepEvalBaseLLM
from deepeval.metrics.utils import initialize_model
from pydantic import BaseModel


class CoherenceVerdict(BaseModel):
    aspect: str
    rating: str
    explanation: str


class CoherenceVerdicts(BaseModel):
    verdicts: List[CoherenceVerdict]


class CoherenceReason(BaseModel):
    reason: str


class CoherenceTemplate:
    @staticmethod
    def analyze_coherence(actual_output: str, expected_output: str):
        return f"""Evaluate whether the actual output and the expected output are semantically equivalent — that is, whether they convey the same essential meaning, even if their wording, phrasing, or emphasis differ. Your goal is not to check for surface similarity, but to assess whether the two outputs are meaningfully aligned. Consider the following aspects:
Factual information: Are the core facts consistent across both outputs?
Content completeness: Do both outputs include the same key ideas or omit important points?
Overall coherence: Do the two outputs align semantically as a whole, even if expressed differently?

**
IMPORTANT: Please make sure to only return in JSON format, with the "verdicts" key mapping to a list of objects. Each object should have "aspect" (the area being evaluated), "rating" (either "high", "medium", or "low"), and "explanation" fields. No words or explanation needed outside the JSON.
**

==== EXAMPLE ====

Actual Output:
The capital of France is Paris. It is located on the Seine River and is known for the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. Paris has a population of about 2.1 million people in the city proper.

Expected Output:
Paris is the capital and largest city of France. With an estimated population of 2.1 million residents, Paris is the center of France's economy, culture, and politics. The city contains many iconic landmarks such as the Eiffel Tower, the Louvre, and Notre-Dame.

Example JSON:
{{
    "verdicts": [
        {{
            "aspect": "Factual information",
            "rating": "high",
            "explanation": "Both outputs correctly state that Paris is the capital of France with a population of about 2.1 million people, and mention key landmarks like the Eiffel Tower and Louvre."
        }},
        {{
            "aspect": "Content completeness",
            "rating": "medium",
            "explanation": "The actual output includes the Seine River, which is not mentioned in the expected output. Meanwhile, the expected output emphasizes Paris’ role in economy, culture, and politics, which the actual output omits."
        }},
        {{
            "aspect": "Overall coherence",
            "rating": "high",
            "explanation": "Despite minor differences in focus, both outputs present consistent and complementary information about Paris without contradictions, conveying the same essential meaning."
        }}
    ]
}}
==== END OF EXAMPLE ====

Actual Output:
{actual_output}

Expected Output:
{expected_output}

JSON:
"""

    @staticmethod
    def generate_reason(verdicts: List[CoherenceVerdict], score: float):
        verdict_strs = []
        for v in verdicts:
            verdict_strs.append(f"{v.aspect} ({v.rating}): {v.explanation}")
            
        return f"""Based on the coherence analysis and score, provide a concise summary explaining the coherence level between the actual and expected outputs. Explain why this score was given, highlighting the strengths and weaknesses in the alignment.

**
IMPORTANT: Please make sure to only return in JSON format, with the 'reason' key providing the reason.
Example JSON:
{{
    "reason": "The outputs have [coherence assessment] because [specific reasons]."
}}
**

Coherence Score: {score}

Coherence Analysis:
{verdict_strs}

JSON:
"""


class CoherenceMetric(BaseMetric):
    _required_params = [
        "actual_output",
        "expected_output"
    ]

    def __init__(
        self,
        threshold: float = 0.7,
        model: Optional[Union[str, DeepEvalBaseLLM]] = None,
        include_reason: bool = True,
        async_mode: bool = True,
        strict_mode: bool = False,
        verbose_mode: bool = False,
        evaluation_template: Type[CoherenceTemplate] = CoherenceTemplate,
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
                import asyncio
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                
                self.score = loop.run_until_complete(
                    self.a_measure(test_case, _show_indicator=False)
                )
            else:
                self.verdicts = self._analyze_coherence(
                    test_case.actual_output, 
                    test_case.expected_output
                )
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
            self.verdicts = await self._a_analyze_coherence(
                test_case.actual_output,
                test_case.expected_output
            )
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

    def _analyze_coherence(self, actual_output: str, expected_output: str) -> List[CoherenceVerdict]:
        prompt = self.evaluation_template.analyze_coherence(
            actual_output=actual_output,
            expected_output=expected_output
        )
        
        try:
            # Generate response
            if self.using_native_model:
                res, cost = self.model.generate(prompt, schema=CoherenceVerdicts)
                self.evaluation_cost += cost
            else:
                res = self.model.generate(prompt)
                
            # Parse response
            data = self._parse_response(res)
            
            # Process verdicts
            if "verdicts" in data and isinstance(data["verdicts"], list):
                result = []
                for item in data["verdicts"]:
                    if isinstance(item, dict) and "aspect" in item and "rating" in item and "explanation" in item:
                        result.append(CoherenceVerdict(
                            aspect=item["aspect"],
                            rating=item["rating"],
                            explanation=item["explanation"]
                        ))
                    elif hasattr(item, "aspect") and hasattr(item, "rating") and hasattr(item, "explanation"):
                        result.append(item)
                return result
            return []
        except Exception:
            return []

    async def _a_analyze_coherence(self, actual_output: str, expected_output: str) -> List[CoherenceVerdict]:
        prompt = self.evaluation_template.analyze_coherence(
            actual_output=actual_output,
            expected_output=expected_output
        )
        
        try:
            # Generate response
            if self.using_native_model:
                res, cost = await self.model.a_generate(prompt, schema=CoherenceVerdicts)
                self.evaluation_cost += cost
            else:
                res = await self.model.a_generate(prompt)
                
            # Parse response
            data = self._parse_response(res)
            
            # Process verdicts
            if "verdicts" in data and isinstance(data["verdicts"], list):
                result = []
                for item in data["verdicts"]:
                    if isinstance(item, dict) and "aspect" in item and "rating" in item and "explanation" in item:
                        result.append(CoherenceVerdict(
                            aspect=item["aspect"],
                            rating=item["rating"],
                            explanation=item["explanation"]
                        ))
                    elif hasattr(item, "aspect") and hasattr(item, "rating") and hasattr(item, "explanation"):
                        result.append(item)
                return result
            return []
        except Exception:
            return []

    def _generate_reason(self) -> str:
        if not self.include_reason or not self.verdicts:
            return "The actual and expected outputs are highly coherent with no significant discrepancies."

        prompt = self.evaluation_template.generate_reason(
            verdicts=self.verdicts,
            score=format(self.score, ".2f"),
        )

        try:
            # Generate response
            if self.using_native_model:
                res, cost = self.model.generate(prompt, schema=CoherenceReason)
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
                return "The actual and expected outputs are highly coherent with no significant discrepancies."
            else:
                aspects = [f"{v.aspect} ({v.rating})" for v in self.verdicts]
                return f"The coherence score is {self.score:.2f} based on assessment of: {', '.join(aspects)}."
                
        except Exception:
            return "No detailed coherence assessment available."

    async def _a_generate_reason(self) -> str:
        if not self.include_reason or not self.verdicts:
            return "The actual and expected outputs are highly coherent with no significant discrepancies."

        prompt = self.evaluation_template.generate_reason(
            verdicts=self.verdicts,
            score=format(self.score, ".2f"),
        )

        try:
            # Generate response
            if self.using_native_model:
                res, cost = await self.model.a_generate(prompt, schema=CoherenceReason)
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
                return "The actual and expected outputs are highly coherent with no significant discrepancies."
            else:
                aspects = [f"{v.aspect} ({v.rating})" for v in self.verdicts]
                return f"The coherence score is {self.score:.2f} based on assessment of: {', '.join(aspects)}."
                
        except Exception:
            return "No detailed coherence assessment available."

    def _calculate_score(self) -> float:
        if not self.verdicts:
            return 1.0  # Perfect score if no verdicts found
        
        # Calculate score based on ratings
        rating_weights = {
            "high": 1.0,
            "medium": 0.6,
            "low": 0.2
        }
        
        # Calculate weighted average
        total_weight = 0
        weighted_sum = 0
        
        for verdict in self.verdicts:
            weight = 1.0  # Default weight for each aspect
            
            # Give higher weight to "Overall coherence" if present
            if verdict.aspect.lower() == "overall coherence":
                weight = 2.0
                
            rating = verdict.rating.lower()
            if rating in rating_weights:
                weighted_sum += rating_weights[rating] * weight
                total_weight += weight
        
        # Calculate final score
        if total_weight > 0:
            score = weighted_sum / total_weight
        else:
            score = 1.0
        
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
        return "Coherence"
