import hashlib
import os
import pickle
from abc import abstractmethod
from itertools import product
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from fasteners import InterProcessReaderWriterLock
from sklearn.base import BaseEstimator
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV

from vidur.config import BaseExecutionTimePredictorConfig, CacheConfig, ReplicaConfig
from vidur.entities import Batch, ExecutionTime
from vidur.entities.execution_time_predictor_request import (
    ExecutionTimePredictorRequest,
)
from vidur.execution_time_predictor.base_execution_time_predictor import (
    BaseExecutionTimePredictor,
)
from vidur.execution_time_predictor.sklearn_execution_time_predictor_batch import (
    SklearnExecutionTimePredictorBatch,
)
from vidur.logger import init_logger
from vidur.types import ExecutionTimePredictorCacheMode

logger = init_logger(__name__)


class SklearnExecutionTimePredictor(BaseExecutionTimePredictor):
    def __init__(
        self,
        predictor_config: BaseExecutionTimePredictorConfig,
        replica_config: ReplicaConfig,
        cache_config: CacheConfig,
    ) -> None:
        super().__init__(
            predictor_config=predictor_config,
            replica_config=replica_config,
            cache_config=cache_config,
        )
        os.makedirs(self._config.cache_dir, exist_ok=True)
        self._send_recv_model = None

        # These overheads are only for GQA models
        self._attention_prefill_batching_overhead_fraction = (
            (self._config.attention_prefill_batching_overhead_fraction)
            if self._model_config.num_q_heads > self._model_config.num_kv_heads
            else 0
        )
        self._attention_decode_batching_overhead_fraction = (
            (self._config.attention_decode_batching_overhead_fraction)
            if self._model_config.num_q_heads > self._model_config.num_kv_heads
            else 0
        )
        # TODO(amey): this won't work for orca scheduler
        self._max_tokens = self._config.prediction_max_tokens_per_request

        num_workers = (
            self._replica_config.num_pipeline_stages
            * self._replica_config.tensor_parallel_size
        )
        devices_per_node = self._replica_config.node_config.num_devices_per_node
        assert (
            num_workers < devices_per_node or num_workers % devices_per_node == 0
        ), "Number of workers should be less than devices per node or a multiple of devices per node"

        self._is_multi_node = num_workers > devices_per_node

        (
            self._compute_input_file,
            self._attention_input_file,
            self._all_reduce_input_file,
            self._send_recv_input_file,
            self._cpu_overhead_input_file,
        ) = self._get_input_files()
        self._predictions = self._predict_from_models()

    def get_batch_execution_time(
        self, batch: Batch, pipeline_stage: int
    ) -> ExecutionTime:
        sklearn_batch = SklearnExecutionTimePredictorBatch.from_batch(
            batch,
            self._config.kv_cache_prediction_granularity,
            self._config.prefill_chunk_size_prediction_granularity,
        )
        return self._get_execution_time(sklearn_batch, pipeline_stage)

    def get_execution_time(
        self,
        execution_time_predictor_requests: List[ExecutionTimePredictorRequest],
        pipeline_stage: int,
    ) -> ExecutionTime:
        sklearn_batch = SklearnExecutionTimePredictorBatch(
            execution_time_predictor_requests,
            self._config.kv_cache_prediction_granularity,
            self._config.prefill_chunk_size_prediction_granularity,
        )
        return self._get_execution_time(sklearn_batch, pipeline_stage)

    def _get_execution_time(
        self, batch: SklearnExecutionTimePredictorBatch, pipeline_stage: int
    ) -> ExecutionTime:
        if pipeline_stage == self._replica_config.num_pipeline_stages - 1:
            pipeline_parallel_communication_time = 0
        else:
            pipeline_parallel_communication_time = (
                self._get_pipeline_parallel_communication_time(batch)
            )

        if self._replica_config.tensor_parallel_size == 1:
            tensor_parallel_communication_time = 0
        else:
            tensor_parallel_communication_time = (
                self._get_tensor_parallel_communication_time(batch)
            )
            
        return ExecutionTime(
            self._num_layers_per_pipeline_stage,
            self._get_attention_rope_execution_time(batch),
            self._get_attention_kv_cache_save_execution_time(batch),
            self._get_attention_decode_execution_time(batch),
            self._get_attention_prefill_execution_time(batch),
            self._get_attention_layer_pre_proj_execution_time(batch),
            self._get_attention_layer_post_proj_execution_time(batch),
            self._get_mlp_layer_up_proj_execution_time(batch),
            self._get_mlp_layer_down_proj_execution_time(batch),
            self._get_mlp_layer_act_execution_time(batch),
            self._get_attn_norm_layer_act_execution_time(batch),
            self._get_mlp_norm_layer_act_execution_time(batch),
            self._get_add_layer_act_execution_time(batch),
            tensor_parallel_communication_time,
            pipeline_parallel_communication_time,
            self._get_schedule_time(batch),
            self._get_sampler_e2e_time(batch),
            self._get_prepare_inputs_e2e_time(batch),
            self._get_process_model_outputs_time(batch),
            self._get_ray_comm_time(batch),
        )

    def _get_input_files(self) -> Tuple[str, str, str, str, str]:
        input_files = [
            self._config.compute_input_file,
            self._config.attention_input_file,
            self._config.all_reduce_input_file,
            self._config.send_recv_input_file,
            self._config.cpu_overhead_input_file,
        ]
        for i in range(len(input_files)):
            input_files[i] = (
                input_files[i]
                .replace("{DEVICE}", self._replica_config.device)
                .replace("{MODEL}", self._model_config.get_name())
                .replace("{NETWORK_DEVICE}", self._replica_config.network_device)
            )

        return tuple(input_files)

    def _load_compute_df(self, file_path: str) -> pd.DataFrame:
        df = self._read_input_file(file_path)
        df = df.drop_duplicates()

        logger.debug(f"Length of complete compute df: {len(df)} {file_path}")
        logger.debug(f"num_q_heads: {self._model_config.num_q_heads}")
        logger.debug(f"embedding_dim: {self._model_config.embedding_dim}")
        logger.debug(f"mlp_hidden_dim: {self._model_config.mlp_hidden_dim}")
        logger.debug(f"use_gated_mlp: {self._model_config.use_gated_mlp}")
        logger.debug(f"vocab_size: {self._model_config.vocab_size}")
        logger.debug(
            f"num_tensor_parallel_workers: {self._replica_config.tensor_parallel_size}"
        )

        return df[
            (df["n_head"] == self._model_config.num_q_heads)
            & (df["n_kv_head"] == self._model_config.num_kv_heads)
            & (df["n_embd"] == self._model_config.embedding_dim)
            & (df["n_expanded_embd"] == self._model_config.mlp_hidden_dim)
            & (df["use_gated_mlp"] == self._model_config.use_gated_mlp)
            & (df["vocab_size"] == self._model_config.vocab_size)
            & (
                df["num_tensor_parallel_workers"]
                == self._replica_config.tensor_parallel_size
            )
        ]

    def _load_attention_df(self, file_path: str) -> pd.DataFrame:
        df = pd.read_csv(file_path)
        df = df.drop_duplicates()

        df = df[
            (df["n_embd"] == self._model_config.embedding_dim)
            & (df["n_q_head"] == self._model_config.num_q_heads)
            & (df["n_kv_head"] == self._model_config.num_kv_heads)
            & (df["block_size"] == self._block_size)
            & (
                df["num_tensor_parallel_workers"]
                == self._replica_config.tensor_parallel_size
            )
        ]

        # Filter out requests with context size greater than max tokens
        df["avg_kv_cache_size_per_request"] = df["kv_cache_size"] / df["batch_size"]
        df = df[
            df["avg_kv_cache_size_per_request"]
            <= self._config.prediction_max_tokens_per_request
        ]

        return df

    def _load_all_reduce_df(self, file_path: str) -> pd.DataFrame:
        df = self._read_input_file(file_path)
        return df[
            (df["num_workers"] == self._replica_config.tensor_parallel_size)
            & (df["devices_per_node"] == self._replica_config.tensor_parallel_size)
            & (df["collective"] == "all_reduce")
        ]

    def _load_send_recv_df(self, file_path: str) -> pd.DataFrame:
        if self._is_multi_node:
            devices_per_node = 1
        else:
            devices_per_node = 2

        df = self._read_input_file(file_path)
        filtered_df = df[
            (df["collective"] == "send_recv")
            & (df["devices_per_node"] == devices_per_node)
        ]
        return filtered_df

    def _load_cpu_overhead_df(self, file_path: str) -> pd.DataFrame:
        df = self._read_input_file(file_path)
        filtered_df = df[
            (df["model_name"] == self._model_config.get_name())
            & (
                df["tensor_parallel_degree"]
                == self._replica_config.tensor_parallel_size
            )
        ]
        return filtered_df

    def _read_input_file(self, file_path: str) -> pd.DataFrame:
        df = pd.read_csv(file_path)
        df = df.drop_duplicates()
        return df

    def _get_compute_df_with_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df_with_derived_features = df.copy()
        for column in [
            "time_stats.post_attention_layernorm.median",
            "time_stats.add.median",
            "time_stats.input_layernorm.median",
        ]:
            if column not in df_with_derived_features.columns:
                df_with_derived_features[column] = 0
            else:
                df_with_derived_features.fillna({column: 0}, inplace=True)
        return df_with_derived_features

    def _get_attention_df_with_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df_with_derived_features = df.copy()

        df_with_derived_features["num_tokens"] = df_with_derived_features[
            ["prefill_chunk_size", "batch_size"]
        ].max(axis=1)
        df_with_derived_features["is_decode"] = (
            df_with_derived_features["prefill_chunk_size"] == 0
        )
        assert (
            df_with_derived_features["is_decode"]
            == ~df_with_derived_features["is_prefill"]
        ).all()

        for column in [
            "time_stats.attn_kv_cache_save.median",
        ]:
            if column not in df_with_derived_features.columns:
                df_with_derived_features[column] = 0
            else:
                df_with_derived_features.fillna({column: 0}, inplace=True)
        if "time_stats.attn_prefill.median" not in df_with_derived_features.columns:
            df_with_derived_features["time_stats.attn_prefill.median"] = (
                df_with_derived_features["time_stats.attn.median"].where(
                    df_with_derived_features["is_prefill"], 0
                )
            )
        if "time_stats.attn_decode.median" not in df_with_derived_features.columns:
            df_with_derived_features["time_stats.attn_decode.median"] = (
                df_with_derived_features["time_stats.attn.median"].where(
                    df_with_derived_features["is_prefill"] == False, 0
                )
            )

        return df_with_derived_features

    def _get_all_reduce_df_with_derived_features(
        self, df: pd.DataFrame
    ) -> pd.DataFrame:
        df_with_derived_features = df.copy()
        # convert bytes to num tokens
        # each token is of size 2 * h bytes
        df_with_derived_features["num_tokens"] = (
            df_with_derived_features["size"] / self._model_config.embedding_dim / 2
        )
        return df_with_derived_features

    def _get_send_recv_df_with_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df_with_derived_features = df.copy()
        df_with_derived_features["num_tokens"] = (
            df_with_derived_features["size"] / self._model_config.embedding_dim / 2
        )
        return df_with_derived_features

    def _get_cpu_overhead_df_with_derived_features(
        self, df: pd.DataFrame
    ) -> pd.DataFrame:
        df_with_derived_features = df.copy()
        return df_with_derived_features

    @staticmethod
    def mean_absolute_percentage_error(y_true: np.array, y_pred: np.array) -> float:
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        # Handling the case where y_true is 0 separately to avoid division by zero
        zero_true_mask = y_true == 0
        non_zero_true_mask = ~zero_true_mask

        # For non-zero true values, calculate the absolute percentage error
        error = np.zeros_like(y_true, dtype=float)  # using float instead of np.float
        error[non_zero_true_mask] = (
            np.abs(
                (y_true[non_zero_true_mask] - y_pred[non_zero_true_mask])
                / y_true[non_zero_true_mask]
            )
            * 100
        )

        # For zero true values, if prediction is also 0, error is 0, else it is 100
        error[zero_true_mask] = np.where(y_pred[zero_true_mask] == 0, 0, 100)

        # Return the mean of the absolute percentage errors
        return np.mean(error)

    def _get_scorer(self) -> Any:
        return make_scorer(
            SklearnExecutionTimePredictor.mean_absolute_percentage_error,
            greater_is_better=False,
        )

    @abstractmethod
    def _get_grid_search_params(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def _get_estimator(self) -> BaseEstimator:
        pass

    def _get_model_hash(self, model_name: str, df: pd.DataFrame = None) -> str:
        config_str = str(self.to_dict())

        if df is None:
            combined_str = f"{config_str}_{model_name}"
        else:
            df_hash_str = hashlib.md5(df.to_json().encode("utf-8")).hexdigest()
            combined_str = f"{config_str}_{model_name}_{df_hash_str}"

        return hashlib.md5(combined_str.encode("utf-8")).hexdigest()[0:8]

    def _load_model_from_cache(self, model_name: str, model_hash: str) -> BaseEstimator:
        with InterProcessReaderWriterLock(
            f"{self._config.cache_dir}/{model_name}_{model_hash}_model_lock.file"
        ).read_lock():
            if self._config.cache_mode == ExecutionTimePredictorCacheMode.IGNORE_CACHE:
                return
            # check if model is in cache
            cache_file = f"{self._config.cache_dir}/{model_name}_{model_hash}.pkl"
            if not os.path.exists(cache_file):
                if (
                    self._config.cache_mode
                    == ExecutionTimePredictorCacheMode.REQUIRE_CACHE
                ):
                    raise Exception(
                        f"Model {model_name} not found in cache but required. Please train the model first."
                    )
                return

            logger.debug(f"Found model {model_name} in cache")
            model = pickle.load(open(cache_file, "rb"))
            return model

    def _store_model_in_cache(
        self, model_name: str, model_hash: str, model: BaseEstimator
    ) -> None:
        with InterProcessReaderWriterLock(
            f"{self._config.cache_dir}/{model_name}_{model_hash}_model_lock.file"
        ).write_lock():
            if self._config.cache_mode in [
                ExecutionTimePredictorCacheMode.IGNORE_CACHE,
                ExecutionTimePredictorCacheMode.REQUIRE_CACHE,
            ]:
                return
            # store model in cache
            cache_file = f"{self._config.cache_dir}/{model_name}_{model_hash}.pkl"
            pickle.dump(model, open(cache_file, "wb"), protocol=pickle.HIGHEST_PROTOCOL)

    def _store_training_prediction_data(
        self,
        model_name: str,
        model_hash: str,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str,
        model: BaseEstimator,
    ) -> None:
        df = df.copy()

        # convert the df to list of tuples
        df["prediction"] = model.predict(df[feature_cols])

        # store the prediction data
        df[feature_cols + [target_col, "prediction"]].to_csv(
            f"{self._config.cache_dir}/{model_name}_{model_hash}_training_predictions.csv",
            index=False,
        )

    def _train_model(
        self,
        model_name: str,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str,
    ) -> BaseEstimator:
        if len(df) == 0:
            raise Exception(f"Training data for model {model_name} is empty")

        model_hash = self._get_model_hash(model_name, df)

        cached_model = self._load_model_from_cache(model_name, model_hash)
        if cached_model:
            return cached_model

        model = self._get_estimator()
        grid_search_params = self._get_grid_search_params()

        if len(df) < self._config.k_fold_cv_splits:
            cv = 2
        else:
            cv = self._config.k_fold_cv_splits

        grid_search = GridSearchCV(
            estimator=model,
            param_grid=grid_search_params,
            scoring=self._get_scorer(),
            cv=cv,
            n_jobs=self._config.num_training_job_threads,
        )

        # we don't create a train/test split, because we want to use all data for training
        # and we don't care about overfitting, because we only want to predict execution time within the same domain
        X, y = df[feature_cols], df[target_col]

        grid_search.fit(X, y)
        score = grid_search.score(X, y)

        logger.info(
            f"Trained model {model_name} and found best parameters: {grid_search.best_params_} "
            f"with mean absolute percentage error (MEAP) {-score}%"
        )

        self._store_model_in_cache(model_name, model_hash, grid_search.best_estimator_)

        # TODO(t-nitinkedia): Training prediction data is debugging. Commenting it from critical path.
        # self._store_training_prediction_data(
        #     model_name=model_name,
        #     model_hash=model_hash,
        #     df=df,
        #     feature_cols=feature_cols,
        #     target_col=target_col,
        #     model=grid_search.best_estimator_,
        # )
        return grid_search.best_estimator_

    def _store_model_predication_cache(
        self,
        model_name: str,
        model_hash: str,
        predictions_dict: Dict[Tuple, float],
        predictions_df: pd.DataFrame,
    ) -> None:
        with InterProcessReaderWriterLock(
            f"{self._config.cache_dir}/{model_name}_{model_hash}_prediction_lock.file"
        ).write_lock():
            if self._config.cache_mode in [
                ExecutionTimePredictorCacheMode.IGNORE_CACHE,
                ExecutionTimePredictorCacheMode.REQUIRE_CACHE,
            ]:
                return
            cache_file = (
                f"{self._config.cache_dir}/{model_name}_{model_hash}_predictions.pkl"
            )
            pickle.dump(
                predictions_dict,
                open(cache_file, "wb"),
                protocol=pickle.HIGHEST_PROTOCOL,
            )
            predictions_df.to_csv(
                f"{self._config.cache_dir}/{model_name}_{model_hash}_predictions.csv",
                index=False,
            )

    def _load_model_predication_cache(
        self, model_name: str, model_hash: str
    ) -> Dict[Tuple, float]:
        with InterProcessReaderWriterLock(
            f"{self._config.cache_dir}/{model_name}_{model_hash}_prediction_lock.file"
        ).read_lock():
            if self._config.cache_mode == ExecutionTimePredictorCacheMode.IGNORE_CACHE:
                return
            cache_file = (
                f"{self._config.cache_dir}/{model_name}_{model_hash}_predictions.pkl"
            )

            if not os.path.exists(cache_file):
                if (
                    self._config.cache_mode
                    == ExecutionTimePredictorCacheMode.REQUIRE_CACHE
                ):
                    raise Exception(
                        f"Model {model_name} not found in cache but is required. Please train the model first."
                    )
                return

            logger.debug(f"Found model {model_name} predictions in cache")

            predictions = pickle.load(open(cache_file, "rb"))
            return predictions

    def _get_model_prediction(
        self, model_name: str, model: BaseEstimator, X: pd.DataFrame
    ) -> Dict[Tuple, float]:
        X = X.copy()

        model_hash = self._get_model_hash(model_name, df=None)

        cached_predictions = self._load_model_predication_cache(model_name, model_hash)
        if cached_predictions:
            return cached_predictions

        logger.info(f"Predicting execution time for model {model_name}")

        predictions_array = model.predict(X)

        # turn this into a dict, so we can store use it as a cache
        # the key is tuple for each row of X
        predictions_dict = dict(zip([tuple(x) for x in X.values], predictions_array))
        X["prediction"] = predictions_array  # predictions_df
        self._store_model_predication_cache(model_name, model_hash, predictions_dict, X)
        return predictions_dict

    def _train_compute_models(self) -> Dict[str, BaseEstimator]:
        compute_df = self._load_compute_df(self._compute_input_file)
        compute_df = self._get_compute_df_with_derived_features(compute_df)

        models = {}
        model_names = [
            "attn_pre_proj",
            "attn_post_proj",
            "mlp_up_proj",
            "mlp_down_proj",
            "mlp_act",
            "input_layernorm",
            "post_attention_layernorm",
            "attn_rope",
            "add",
        ]

        for model_name in model_names:
            logger.debug(
                f"Training model {model_name}, size of training data: {len(compute_df)}"
            )
            models[model_name] = self._train_model(
                model_name=model_name,
                df=compute_df,
                feature_cols=["num_tokens"],
                target_col=f"time_stats.{model_name}.median",
            )

        attention_df = self._load_attention_df(self._attention_input_file)
        attention_df = self._get_attention_df_with_derived_features(attention_df)

        model_names = [
            "attn_kv_cache_save",
        ]

        for model_name in model_names:
            models[model_name] = self._train_model(
                model_name=model_name,
                df=attention_df,
                feature_cols=["num_tokens"],
                target_col=f"time_stats.{model_name}.median",
            )

        if self._replica_config.num_pipeline_stages > 1 or self._replica_config.pd_disaggregation:
            send_recv_df = self._load_send_recv_df(self._send_recv_input_file)
            send_recv_df = self._get_send_recv_df_with_derived_features(send_recv_df)

            models["send_recv"] = self._train_model(
                model_name="send_recv",
                df=send_recv_df,
                feature_cols=["num_tokens"],
                target_col="time_stats.send_recv.median",
            )
            self._send_recv_model = models["send_recv"]

        if self._replica_config.tensor_parallel_size > 1:
            all_reduce_df = self._load_all_reduce_df(self._all_reduce_input_file)
            all_reduce_df = self._get_all_reduce_df_with_derived_features(all_reduce_df)

            models["all_reduce"] = self._train_model(
                model_name="all_reduce",
                df=all_reduce_df,
                feature_cols=["num_tokens"],
                target_col="time_stats.all_reduce.median",
            )
        return models

    def _train_cpu_overhead_models(self) -> Dict[str, BaseEstimator]:
        if self._config.skip_cpu_overhead_modeling:
            return {}

        models = {}
        model_names = [
            "schedule",
            "sampler_e2e",
            "prepare_inputs_e2e",
            "process_model_outputs",
            "ray_comm_time",
        ]

        cpu_overhead_df = self._load_cpu_overhead_df(self._cpu_overhead_input_file)
        cpu_overhead_df = self._get_cpu_overhead_df_with_derived_features(
            cpu_overhead_df
        )

        for model_name in model_names:
            if model_name == "ray_comm_time":
                target_col = "ray_comm_time_mean"
            else:
                target_col = f"{model_name}_median"

            models[model_name] = self._train_model(
                model_name=model_name,
                df=cpu_overhead_df,
                feature_cols=["batch_size"],
                target_col=target_col,
            )

        return models

    def _train_attention_layer_models(self) -> Dict[str, BaseEstimator]:
        attention_df = self._load_attention_df(self._attention_input_file)
        attention_df = self._get_attention_df_with_derived_features(attention_df)
        prefill_df = attention_df[~attention_df["is_decode"]]
        decode_df = attention_df[attention_df["is_decode"]]

        models = {}
        models["attn_prefill"] = self._train_model(
            model_name="attn_prefill",
            df=prefill_df,
            feature_cols=["kv_cache_size", "prefill_chunk_size"],
            target_col="time_stats.attn_prefill.median",
        )

        models["attn_decode"] = self._train_model(
            model_name="attn_decode",
            df=decode_df,
            feature_cols=["batch_size", "kv_cache_size"],
            target_col="time_stats.attn_decode.median",
        )

        return models

    def _predict_for_compute_models(self) -> Dict[str, Any]:
        logger.info("Getting predictions for compute operations")
        models = self._train_compute_models()
        predictions = {}

        model_names = [
            "attn_pre_proj",
            "attn_post_proj",
            "mlp_up_proj",
            "mlp_down_proj",
            "mlp_act",
            "attn_rope",
            "attn_kv_cache_save",
            "input_layernorm",
            "post_attention_layernorm",
            "add",
        ]

        if self._replica_config.num_pipeline_stages > 1 or self._replica_config.pd_disaggregation:
            model_names.append("send_recv")

        if self._replica_config.tensor_parallel_size > 1:
            model_names.append("all_reduce")

        num_token_range = np.arange(1, self._max_tokens + 1)
        X = pd.DataFrame({"num_tokens": num_token_range})

        for model_name in model_names:
            model = models[model_name]
            predictions[model_name] = self._get_model_prediction(model_name, model, X)

        return predictions

    def _predict_for_cpu_overhead_models(self) -> Dict[str, Any]:
        if self._config.skip_cpu_overhead_modeling:
            return {}

        logger.info("Getting predictions for CPU overhead operations")
        models = self._train_cpu_overhead_models()
        predictions = {}

        model_names = [
            "schedule",
            "sampler_e2e",
            "prepare_inputs_e2e",
            "process_model_outputs",
            "ray_comm_time",
        ]

        batch_size_range = np.arange(1, self._config.prediction_max_batch_size + 1)
        X = pd.DataFrame({"batch_size": batch_size_range})

        for model_name in model_names:
            model = models[model_name]
            predictions[model_name] = self._get_model_prediction(model_name, model, X)

        return predictions

    def _predict_for_attention_layer_models(self) -> Dict[str, Any]:
        logger.info("Getting predictions for attention operations")
        models = self._train_attention_layer_models()
        predictions = {}

        decode_batch_size_range = np.arange(
            1, self._config.prediction_max_batch_size + 1
        )
        decode_kv_cache_size_range = np.arange(
            0,
            self._config.prediction_max_tokens_per_request + 1,
            self._config.kv_cache_prediction_granularity,
        )
        decode_prefill_chunk_size_range = [0]
        decode_batch_size, decode_kv_cache_size, decode_prefill_chunk_size = zip(
            *product(
                decode_batch_size_range,
                decode_kv_cache_size_range,
                decode_prefill_chunk_size_range,
            )
        )

        prefill_batch_size_range = [1]
        prefill_kv_cache_size_range = np.arange(
            0,
            self._config.prediction_max_tokens_per_request + 1,
            self._config.kv_cache_prediction_granularity,
        )
        prefill_chunk_size_range = np.arange(
            self._config.prefill_chunk_size_prediction_granularity,
            self._config.prediction_max_prefill_chunk_size + 1,
            self._config.prefill_chunk_size_prediction_granularity,
        )
        prefill_batch_size, prefill_kv_cache_size, prefill_chunk_size = zip(
            *product(
                prefill_batch_size_range,
                prefill_kv_cache_size_range,
                prefill_chunk_size_range,
            )
        )

        attention_df = pd.DataFrame(
            {
                "batch_size": decode_batch_size + prefill_batch_size,
                "kv_cache_size": decode_kv_cache_size + prefill_kv_cache_size,
                "prefill_chunk_size": decode_prefill_chunk_size + prefill_chunk_size,
            }
        )

        attention_df["is_decode"] = attention_df["prefill_chunk_size"] == 0
        attention_df["num_tokens"] = attention_df[
            ["prefill_chunk_size", "batch_size"]
        ].max(axis=1)

        prefill_df = attention_df[~attention_df["is_decode"]]
        decode_df = attention_df[attention_df["is_decode"]]

        predictions["attn_prefill"] = self._get_model_prediction(
            "attn_prefill",
            models["attn_prefill"],
            prefill_df[["kv_cache_size", "prefill_chunk_size"]],
        )

        predictions["attn_decode"] = self._get_model_prediction(
            "attn_decode",
            models["attn_decode"],
            decode_df[["batch_size", "kv_cache_size"]],
        )

        return predictions

    def _predict_from_models(self) -> Dict[str, Any]:
        predictions = self._predict_for_compute_models()
        predictions.update(self._predict_for_cpu_overhead_models())
        predictions.update(self._predict_for_attention_layer_models())

        return predictions

    def _get_attention_layer_pre_proj_execution_time(
        self, batch: SklearnExecutionTimePredictorBatch
    ) -> float:
        return self._predictions["attn_pre_proj"][(batch.total_num_tokens_rounded,)]

    def _get_attention_layer_post_proj_execution_time(
        self, batch: SklearnExecutionTimePredictorBatch
    ) -> float:
        return self._predictions["attn_post_proj"][(batch.total_num_tokens_rounded,)]

    def _get_mlp_layer_up_proj_execution_time(
        self, batch: SklearnExecutionTimePredictorBatch
    ) -> float:
        return self._predictions["mlp_up_proj"][(batch.total_num_tokens_rounded,)]

    def _get_mlp_layer_down_proj_execution_time(
        self, batch: SklearnExecutionTimePredictorBatch
    ) -> float:
        return self._predictions["mlp_down_proj"][(batch.total_num_tokens_rounded,)]

    def _get_mlp_layer_act_execution_time(
        self, batch: SklearnExecutionTimePredictorBatch
    ) -> float:
        return self._predictions["mlp_act"][(batch.total_num_tokens_rounded,)]

    def _get_attn_norm_layer_act_execution_time(
        self, batch: SklearnExecutionTimePredictorBatch
    ) -> float:
        return self._predictions["input_layernorm"][(batch.total_num_tokens_rounded,)]

    def _get_mlp_norm_layer_act_execution_time(
        self, batch: SklearnExecutionTimePredictorBatch
    ) -> float:
        if not self._model_config.post_attn_norm:
            return 0

        return self._predictions["post_attention_layernorm"][
            (batch.total_num_tokens_rounded,)
        ]

    def _get_add_layer_act_execution_time(
        self, batch: SklearnExecutionTimePredictorBatch
    ) -> float:
        return self._predictions["add"][(batch.total_num_tokens_rounded,)]

    def _get_tensor_parallel_communication_time(
        self, batch: SklearnExecutionTimePredictorBatch
    ) -> float:
        return (
            self._predictions["all_reduce"][(batch._total_num_tokens_rounded,)]
            + self._config.nccl_cpu_launch_overhead_ms
            + self._config.nccl_cpu_skew_overhead_per_device_ms
            * self._replica_config.tensor_parallel_size**1.25
        )

    def _get_pipeline_parallel_communication_time(
        self, batch: SklearnExecutionTimePredictorBatch
    ) -> float:
        try:
            return self._predictions["send_recv"][(batch.total_num_tokens_rounded,)]
        except KeyError as e:
            logger.error(f"Failed to get send_recv prediction for batch {batch}")
            raise e

    def get_kv_transfer_time_cross_node(self, num_kv_tokens: int,
                                        decode_tp_size: int = 0) -> float:
        """KV transfer time (ms) for inter-node handoff. Analytical: kv_bytes_per_gpu / bw_per_gpu.
        When prefill_tp != decode_tp, the bottleneck is the side with fewer GPUs
        """
        prefill_tp = self._replica_config.tensor_parallel_size
        pp_size = self._replica_config.num_pipeline_stages
        effective_tp = min(prefill_tp, decode_tp_size) if decode_tp_size > 0 else prefill_tp

        node_cfg = self._replica_config.node_config
        bw_per_gpu_gbps = node_cfg.inter_node_bandwidth_gbps / node_cfg.num_devices_per_node

        if bw_per_gpu_gbps <= 0:
            return 0.0

        kv_scale = (
            2.0
            * self._model_config.num_layers
            * self._model_config.num_kv_heads
            / self._model_config.num_q_heads
        )
        kv_size_in_act_tokens = kv_scale * num_kv_tokens / (effective_tp * pp_size)
        kv_bytes_per_gpu = kv_size_in_act_tokens * self._model_config.embedding_dim * 2 

        time_s = kv_bytes_per_gpu / (bw_per_gpu_gbps * 1e9)
        return time_s * 1e3

    def get_kv_transfer_time(self, num_kv_tokens: int) -> float:
        """KV transfer time (ms) for intra-node handoff, using profiled send_recv model."""
        if self._send_recv_model is None:
            return 0.0

        tp_size = self._replica_config.tensor_parallel_size
        pp_size = self._replica_config.num_pipeline_stages
        kv_scale = (
            2.0
            * self._model_config.num_layers
            * self._model_config.num_kv_heads
            / self._model_config.num_q_heads
        )
        # send_recv model input unit: number of activation tokens
        kv_size_in_act_tokens = max(1, int(kv_scale * num_kv_tokens / (tp_size * pp_size)))

        key = (kv_size_in_act_tokens,)
        if key in self._predictions.get("send_recv", {}):
            return self._predictions["send_recv"][key]

        return float(
            self._send_recv_model.predict(
                pd.DataFrame({"num_tokens": [kv_size_in_act_tokens]})
            )[0]
        )

    def _get_attention_rope_execution_time(
        self, batch: SklearnExecutionTimePredictorBatch
    ) -> float:
        return self._predictions["attn_rope"][(batch.total_num_tokens_rounded,)]

    def _get_attention_kv_cache_save_execution_time(
        self, batch: SklearnExecutionTimePredictorBatch
    ) -> float:
        # don't use round up to the nearest multiple of 8 here, because we want to
        # predict the execution time for the exact number of tokens
        return self._predictions["attn_kv_cache_save"][(batch.total_num_tokens,)]

    def _get_attention_decode_execution_time(
        self, batch: SklearnExecutionTimePredictorBatch
    ) -> float:
        if batch.decode_batch_size == 0:
            return 0
        
        return self._predictions["attn_decode"][
            (batch.decode_batch_size, batch.decode_avg_kv_cache_size)
        ] * (
            1
            + self._attention_decode_batching_overhead_fraction
            * int(batch.decode_batch_size > 1)
        )

    def _get_attention_prefill_execution_time(
        self, batch: SklearnExecutionTimePredictorBatch
    ) -> float:
        if batch.prefill_batch_size == 0:
            return 0
        
        return self._predictions["attn_prefill"][
            (batch.prefill_agg_kv_cache_size, batch.prefill_agg_chunk_size)
        ] * (
            1
            + self._attention_prefill_batching_overhead_fraction
            * int(batch.prefill_batch_size > 1)
        )

    def _get_schedule_time(self, batch: SklearnExecutionTimePredictorBatch) -> float:
        if self._config.skip_cpu_overhead_modeling:
            return 0

        return self._predictions["schedule"][(batch.size,)]

    def _get_sampler_e2e_time(self, batch: SklearnExecutionTimePredictorBatch) -> float:
        if self._config.skip_cpu_overhead_modeling:
            return 0

        return self._predictions["sampler_e2e"][(batch.size,)]

    def _get_prepare_inputs_e2e_time(
        self, batch: SklearnExecutionTimePredictorBatch
    ) -> float:
        if self._config.skip_cpu_overhead_modeling:
            return 0

        return self._predictions["prepare_inputs_e2e"][(batch.size,)]

    def _get_process_model_outputs_time(
        self, batch: SklearnExecutionTimePredictorBatch
    ) -> float:
        if self._config.skip_cpu_overhead_modeling:
            return 0

        return self._predictions["process_model_outputs"][(batch.size,)]

    def _get_ray_comm_time(self, batch: SklearnExecutionTimePredictorBatch) -> float:
        if self._config.skip_cpu_overhead_modeling:
            return 0

        return self._predictions["ray_comm_time"][(batch.size,)]

    def to_dict(self) -> dict:
        # Note(t-nitinkedia): A weakness is that we don't consider the grid search params for say the random_forest.
        return {
            # Model config
            "model_provider": str(self._config.get_type()),
            "num_tensor_parallel_workers": self._replica_config.tensor_parallel_size,
            "num_q_heads": self._model_config.num_q_heads,
            "num_kv_heads": self._model_config.num_kv_heads,
            "embedding_dim": self._model_config.embedding_dim,
            "mlp_hidden_dim": self._model_config.mlp_hidden_dim,
            "use_gated_mlp": self._model_config.use_gated_mlp,
            "vocab_size": self._model_config.vocab_size,
            # Profiling data files
            "compute_input_file": self._compute_input_file,
            "attention_input_file": self._attention_input_file,
            "all_reduce_input_file": self._all_reduce_input_file,
            "send_recv_input_file": self._send_recv_input_file,
            "cpu_overhead_input_file": self._cpu_overhead_input_file,
            # Predictor config
            "k_fold_cv_splits": self._config.k_fold_cv_splits,
            "kv_cache_prediction_granularity": self._config.kv_cache_prediction_granularity,
            "prefill_chunk_size_prediction_granularity": self._config.prefill_chunk_size_prediction_granularity,
            "prediction_max_prefill_chunk_size": self._config.prediction_max_prefill_chunk_size,
            "prediction_max_batch_size": self._config.prediction_max_batch_size,
            "prediction_max_tokens_per_request": self._max_tokens,
            "block_size": self._block_size,
        }
