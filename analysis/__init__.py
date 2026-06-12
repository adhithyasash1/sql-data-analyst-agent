"""Analysis planning helpers for SQL result artifacts."""

from analysis.capabilities import (
    AnalysisSuggestion,
    ColumnCapability,
    DatasetCapability,
    RecipeBlock,
    analyze_dataset_capability,
    suggest_analyses,
)
from analysis.executor import AnalysisExecutionResult, AnalysisResultTable, execute_analysis_plan
from analysis.ml_runner import SupervisedRunResult, SupervisedRunTable, run_supervised_model
from analysis.planner import AnalysisPlan, FeatureDecision, plan_analysis
from analysis.supervised import (
    FeatureAssessment,
    PreflightGate,
    SupervisedPreflight,
    build_supervised_preflight,
    infer_problem_type,
)

__all__ = [
    "AnalysisExecutionResult",
    "AnalysisPlan",
    "AnalysisResultTable",
    "AnalysisSuggestion",
    "ColumnCapability",
    "DatasetCapability",
    "FeatureAssessment",
    "FeatureDecision",
    "PreflightGate",
    "RecipeBlock",
    "SupervisedPreflight",
    "SupervisedRunResult",
    "SupervisedRunTable",
    "analyze_dataset_capability",
    "build_supervised_preflight",
    "execute_analysis_plan",
    "infer_problem_type",
    "plan_analysis",
    "run_supervised_model",
    "suggest_analyses",
]
