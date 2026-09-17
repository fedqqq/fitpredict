"""Tabular data loading and metadata helpers."""

from fitpredict.data.dataset import (
    DatasetDiagnostics,
    DatasetSplits,
    TabularDataset,
    build_dataloader,
    build_datasets,
)
from fitpredict.data.loader import (
    TabularRows,
    load_data_metadata,
    load_tabular_data,
    metadata_from_rows,
)
from fitpredict.data.split import (
    SplitDiagnostics,
    TabularSplit,
    split_tabular_data,
)
from fitpredict.data.source_resolver import (
    RuntimeSourceContext,
    SourceResolver,
    resolve_source,
)
from fitpredict.data.tensorizer import (
    TensorizedBatch,
    aggregate_feature_tensors,
    tensorize_binding,
    tensorize_features,
    tensorize_source,
)

__all__ = [
    "DatasetDiagnostics",
    "DatasetSplits",
    "RuntimeSourceContext",
    "SourceResolver",
    "SplitDiagnostics",
    "TabularDataset",
    "TabularRows",
    "TabularSplit",
    "TensorizedBatch",
    "aggregate_feature_tensors",
    "build_dataloader",
    "build_datasets",
    "load_data_metadata",
    "load_tabular_data",
    "metadata_from_rows",
    "resolve_source",
    "split_tabular_data",
    "tensorize_binding",
    "tensorize_features",
    "tensorize_source",
]
