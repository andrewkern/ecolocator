import numpy as np
import pandas as pd
import os
import json
import logging
import tensorflow as tf
from .utils import (
    load_genotypes,
    sort_samples,
    normalize_locs,
    filter_snps,
    replace_missing_data,
    back_transform_env,
)
from .network import build_pgs_network, path_attributions, train_network, euclid_loss
from .settings import NETWORK_DEFAULTS as D

# Network settings, saved with a model and passed to build_pgs_network
# (except score_l2, which fit() resolves to one penalty first).
HYPERPARAMETERS = tuple(D)


def _json_value(value):
    """A setting as JSON can store it: sequences (also NumPy arrays) as lists
    of floats, NumPy scalars as Python numbers."""
    if isinstance(value, (list, tuple, np.ndarray)):
        return [float(v) for v in value]
    return value.item() if isinstance(value, np.generic) else value


def _column_mean_sd(*blocks):
    """Mean and SD of every column over the rows of all blocks together,
    without copying the blocks into one array."""
    n = sum(len(b) for b in blocks)
    mean = sum(b.sum(axis=0, dtype=float) for b in blocks) / n
    var = sum(np.square(b - mean).sum(axis=0) for b in blocks) / n
    return mean, np.sqrt(var)


class EcoLocator:
    """
    A neural network model for predicting sample locations and environmental
    covariates from genotype data.

    The network (network.build_pgs_network) has a location trunk and, for
    each covariate, a linear polygenic score of all SNPs plus a small
    correction head that also reads the predicted location. ``score_l2`` is
    the L2 penalty of the scores: a number fixes it; a sequence is searched,
    each fit keeping the penalty with the lowest covariate validation loss.
    """

    def __init__(
        self,
        cov_transforms: list = None,
        nlayers: int = D["nlayers"],
        width: int = D["width"],
        dropout_prop: float = D["dropout_prop"],
        input_dropout: float = D["input_dropout"],
        weight_decay: float = D["weight_decay"],
        head_width: int = D["head_width"],
        score_l2: float | tuple = D["score_l2"],
        loc_weight: float = D["loc_weight"],
        env_weight: float = D["env_weight"],
    ):
        self.cov_transforms = cov_transforms
        self.nlayers = nlayers
        self.width = width
        self.dropout_prop = dropout_prop
        self.input_dropout = input_dropout
        self.weight_decay = weight_decay
        self.head_width = head_width
        self.score_l2 = score_l2
        self.loc_weight = loc_weight
        self.env_weight = env_weight
        # "pgs" (build_pgs_network); load() sets "dense" for models saved
        # with network.build_network, which fit() does not refit.
        self.network_ = "pgs"

    @classmethod
    def load(cls, path: str) -> "EcoLocator":
        with open(os.path.join(path, "params.json")) as f:
            params = json.load(f)

        # Models saved with the dense network (network.build_network) have
        # no "network" field and fewer settings; the missing ones keep their
        # defaults. The saved model file predicts and explains as it was
        # fitted; fit() refuses to refit such a model.
        obj = cls(
            cov_transforms=params["cov_transforms"],
            **{k: params[k] for k in HYPERPARAMETERS if k in params},
        )
        obj.network_ = params.get("network", "dense")

        arrays = np.load(os.path.join(path, "arrays.npz"))
        obj.meanlong_ = arrays["meanlong"].item()
        obj.sdlong_ = arrays["sdlong"].item()
        obj.meanlat_ = arrays["meanlat"].item()
        obj.sdlat_ = arrays["sdlat"].item()
        obj.means_ = arrays["means"]
        obj.sds_ = arrays["sds"]
        obj._kept_snp_indices_ = arrays["kept_snp_indices"]

        obj.cov_names_ = params["cov_names"]
        obj.seed_ = params["seed"]
        obj.num_covs_ = len(obj.cov_names_)
        obj.transforms_ = params["transforms"]
        obj.score_l2_ = params.get("score_l2_fitted")

        obj.model_ = tf.keras.models.load_model(
            os.path.join(path, "model.keras"),
            custom_objects={"euclid_loss": euclid_loss},
        )

        return obj

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        self.model_.save(os.path.join(path, "model.keras"))
        np.savez(
            os.path.join(path, "arrays.npz"),
            meanlong=self.meanlong_,
            sdlong=self.sdlong_,
            meanlat=self.meanlat_,
            sdlat=self.sdlat_,
            means=self.means_,
            sds=self.sds_,
            kept_snp_indices=self._kept_snp_indices_,
        )
        params = {
            "seed": self.seed_,
            "cov_names": self.cov_names_,
            "num_covs": self.num_covs_,
            "transforms": self.transforms_,
            "cov_transforms": self.cov_transforms,
            "score_l2_fitted": _json_value(self.score_l2_),
            "network": self.network_,
            **{k: _json_value(getattr(self, k)) for k in HYPERPARAMETERS},
        }
        with open(os.path.join(path, "params.json"), "w") as f:
            json.dump(params, f)

    def fit(
        self,
        genotype_path: str,
        sample_data_path: str,
        max_epochs: int = 5000,
        patience: int = 100,
        batch_size: int = 32,
        min_mac: int = 2,
        max_snps: int = None,
        train_split: float = 0.9,
        seed: int = None,
        verbose: int = 1,
    ) -> "EcoLocator":
        if self.network_ != "pgs":
            raise ValueError(
                "this model was saved with the dense network (build_network); "
                "fit a new EcoLocator instead"
            )
        rng = np.random.default_rng(seed)

        genotypes, samples, _ = self._get_genotypes(genotype_path)
        sample_data, locs = sort_samples(samples, sample_data_path)

        num_covs = locs.shape[1] - 2
        cov_names = [c for c in sample_data.columns if c not in {"sampleID2", "x", "y"}]
        self.cov_names_ = cov_names
        if num_covs < 1:
            logging.warning(
                f"Found {num_covs} covariates in sample data. Proceeding with location data only."
            )

        (
            self.meanlong_,
            self.sdlong_,
            self.meanlat_,
            self.sdlat_,
            self.means_,
            self.sds_,
            self.transforms_,
            locs,
        ) = normalize_locs(locs, transforms=self.cov_transforms, cov_names=cov_names)

        genotypes, self._kept_snp_indices_ = filter_snps(
            genotypes, min_mac=min_mac, max_snps=max_snps, rng=rng
        )
        ac = replace_missing_data(genotypes, rng=rng)

        train = np.argwhere(~np.isnan(locs[:, 0])).flatten()
        test = rng.choice(train, round((1 - train_split) * len(train)), replace=False)
        train = np.array([x for x in train if x not in test])

        traingen = np.transpose(ac[:, train])
        testgen = np.transpose(ac[:, test])

        if num_covs > 0:
            trainlocs = [locs[train][:, 0:2], locs[train][:, 2:]]
            testlocs = [locs[test][:, 0:2], locs[test][:, 2:]]
        else:
            trainlocs = locs[train][:, 0:2]
            testlocs = locs[test][:, 0:2]

        self.num_covs_ = num_covs
        self.seed_ = seed
        self.model_, self.history_, self.score_l2_, self.score_l2_losses_ = (
            self._fit_network(
                traingen,
                testgen,
                trainlocs,
                testlocs,
                num_covs=num_covs,
                seed=seed,
                max_epochs=max_epochs,
                batch_size=batch_size,
                patience=patience,
                verbose=verbose,
            )
        )
        return self

    def _fit_network(
        self,
        traingen,
        testgen,
        trainlocs,
        testlocs,
        *,
        num_covs,
        seed,
        max_epochs,
        batch_size,
        patience,
        verbose,
    ):
        """Fit the network on one split: traingen/trainlocs to fit, and
        testgen/testlocs for early stopping and the choice of the score
        penalty. Genotypes are standardized with the means and SDs of both
        sets together. The network is fitted once per penalty of score_l2,
        from the same seed, and the fit with the lowest covariate validation
        loss (MSE) is kept. Returns that model, its history, its penalty,
        and each penalty's validation loss."""
        if len(testgen) == 0:
            raise ValueError(
                "no validation samples: train_split leaves none, and early "
                "stopping and the penalty search need them"
            )
        geno_mean, geno_sd = _column_mean_sd(traingen, testgen)
        penalties = (
            [0.0] if num_covs == 0 else [float(p) for p in np.atleast_1d(self.score_l2)]
        )
        if not penalties or not all(np.isfinite(p) and p >= 0 for p in penalties):
            raise ValueError(
                f"score_l2 must be one or more finite numbers >= 0, not {self.score_l2!r}"
            )
        settings = {k: getattr(self, k) for k in HYPERPARAMETERS if k != "score_l2"}
        best, losses = None, {}
        for penalty in penalties:
            if seed is not None:
                # Weight initialization, dropout masks, and batch shuffling
                # draw from TensorFlow's own RNG, which the NumPy seed does
                # not reach.
                tf.keras.utils.set_random_seed(seed)
            model = build_pgs_network(num_covs, geno_mean, geno_sd, penalty, **settings)
            history = train_network(
                model,
                traingen,
                testgen,
                trainlocs,
                testlocs,
                max_epochs=max_epochs,
                batch_size=batch_size,
                patience=patience,
                verbose=verbose,
            )
            # Early stopping restores the epoch with the lowest validation
            # loss; that epoch's covariate validation loss decides.
            h = history.history
            key = f"val_{model.output_names[1]}_loss" if num_covs > 0 else "val_loss"
            # A diverged fit (NaN loss) counts as the worst.
            val_loss = np.nan_to_num(h["val_loss"], nan=np.inf)
            loss = float(h[key][int(np.argmin(val_loss))])
            losses[penalty] = loss if np.isfinite(loss) else np.inf
            if best is None or losses[penalty] < losses[best[2]]:
                best = (model, history, penalty)
        return (*best, losses)

    def predict(
        self,
        genotype_path: str,
        sample_data_path: str,
    ) -> pd.DataFrame:
        genotypes, samples, _ = self._get_genotypes(genotype_path)
        _, locs = sort_samples(samples, sample_data_path)

        genotypes = genotypes[self._kept_snp_indices_, :, :]
        ac = replace_missing_data(genotypes)

        pred = np.argwhere(np.isnan(locs[:, 0])).flatten()
        predgen = np.transpose(ac[:, pred])

        prediction = self.model_.predict(predgen)
        if self.num_covs_ > 0:
            pred_longlat = prediction[0] * np.array(
                [self.sdlong_, self.sdlat_]
            ) + np.array([self.meanlong_, self.meanlat_])
            pred_env = back_transform_env(
                prediction[1], self.means_, self.sds_, self.transforms_
            )

            result = pd.DataFrame(pred_longlat, columns=["x", "y"])
            cov_cols = pd.DataFrame(pred_env, columns=self.cov_names_)
            result = pd.concat([result, cov_cols], axis=1)
            result.insert(0, "sampleID", samples[pred])
            return result
        else:
            pred_longlat = prediction * np.array(
                [self.sdlong_, self.sdlat_]
            ) + np.array([self.meanlong_, self.meanlat_])

            result = pd.DataFrame(pred_longlat, columns=["x", "y"])
            result.insert(0, "sampleID", samples[pred])
            return result

    def fit_predict_loo(
        self,
        genotype_path: str,
        sample_data_path: str,
        max_epochs: int = 5000,
        patience: int = 100,
        batch_size: int = 32,
        min_mac: int = 2,
        max_snps: int = None,
        max_folds: int = None,
        train_split: float = 0.9,
        seed: int = None,
        verbose: int = 1,
        fold_start: int = 0,
        fold_stop: int = None,
        shap_background_size: int = None,
    ) -> pd.DataFrame:
        """Leave-one-out fit and predict over samples with known coordinates.

        Fold k holds out the k-th sample (in genotype order) that has known
        coordinates. ``fold_start`` and ``fold_stop`` select folds
        ``[fold_start, fold_stop)``, so that one LOO run can be split across
        many jobs. When ``seed`` is set, each fold's validation split and
        network initialization depend only on ``seed`` and the fold index,
        so a fold gives the same prediction in any chunk.

        Each fold fits the network as fit() does, once per penalty of
        ``score_l2``, so a fold costs that many fits; the penalty each fold
        chose is in ``self.loo_score_l2_``. For a cheaper LOO, fix
        ``score_l2`` to one number, for example the penalty that fit() chose
        on all samples.

        With ``shap_background_size``, each fold also computes SHAP values
        for its held-out sample, from the model that did not see it, with a
        background of that many training samples. They are stored in
        ``self.loo_shap_``: a dict with ``sample_ids`` (n_folds),
        ``snp_ids`` (n_snps), ``outputs`` (x, y, then covariates), and
        ``values`` (n_folds x n_snps x n_outputs).
        """
        rng = np.random.default_rng(seed)
        genotypes, samples, all_snp_ids = self._get_genotypes(genotype_path)
        sample_data, locs = sort_samples(samples, sample_data_path)

        num_covs = locs.shape[1] - 2
        cov_names = [c for c in sample_data.columns if c not in {"sampleID2", "x", "y"}]
        if num_covs < 1:
            logging.warning(
                f"Found {num_covs} covariates in sample data. Proceeding with location data only."
            )

        genotypes, kept_snps = filter_snps(
            genotypes, min_mac=min_mac, max_snps=max_snps, rng=rng
        )
        ac = replace_missing_data(genotypes, rng=rng)
        fold_shap = []
        fold_penalties = []

        known = np.argwhere(~np.isnan(locs[:, 0])).flatten()
        fold_ids = np.arange(len(known))[fold_start:fold_stop]
        if max_folds is not None:
            fold_ids = fold_ids[:max_folds]
        all_predictions = []

        for n, fold in enumerate(fold_ids, start=1):
            i = known[fold]
            logging.info(
                f"LOO fold {fold} ({n} of {len(fold_ids)}): holding out {samples[i]}"
            )
            if seed is not None:
                fold_rng = np.random.default_rng([seed, fold])
                fold_seed = int(
                    np.random.SeedSequence([seed, fold]).generate_state(1)[0]
                )
            else:
                fold_rng = rng
                fold_seed = None
            loo_locs = locs.copy()
            loo_locs[i] = np.nan

            (meanlong, sdlong, meanlat, sdlat, means, sds, transforms, norm_locs) = (
                normalize_locs(
                    loo_locs, transforms=self.cov_transforms, cov_names=cov_names
                )
            )

            train = np.argwhere(~np.isnan(norm_locs[:, 0])).flatten()
            test = fold_rng.choice(
                train, round((1 - train_split) * len(train)), replace=False
            )
            train = np.array([x for x in train if x not in test])

            traingen = np.transpose(ac[:, train])
            testgen = np.transpose(ac[:, test])

            if num_covs > 0:
                trainlocs = [norm_locs[train][:, 0:2], norm_locs[train][:, 2:]]
                testlocs = [norm_locs[test][:, 0:2], norm_locs[test][:, 2:]]
            else:
                trainlocs = norm_locs[train][:, 0:2]
                testlocs = norm_locs[test][:, 0:2]

            predgen = np.transpose(ac[:, [i]])

            model, _, penalty, _ = self._fit_network(
                traingen,
                testgen,
                trainlocs,
                testlocs,
                num_covs=num_covs,
                seed=fold_seed,
                max_epochs=max_epochs,
                batch_size=batch_size,
                patience=patience,
                verbose=verbose,
            )
            fold_penalties.append(penalty)

            prediction = model.predict(predgen)
            if shap_background_size is not None:
                fold_shap.append(
                    self._loo_fold_shap(
                        model,
                        traingen,
                        predgen,
                        num_covs,
                        shap_background_size,
                        fold_rng,
                    )
                )
            if num_covs > 0:
                pred_longlat = prediction[0] * np.array([sdlong, sdlat]) + np.array(
                    [meanlong, meanlat]
                )
                pred_env = back_transform_env(prediction[1], means, sds, transforms)

                row = {
                    "sampleID": samples[i],
                    "x": pred_longlat[0, 0],
                    "y": pred_longlat[0, 1],
                }
                for j, name in enumerate(cov_names):
                    row[name] = pred_env[0, j]
                all_predictions.append(row)
            else:
                pred_longlat = prediction * np.array([sdlong, sdlat]) + np.array(
                    [meanlong, meanlat]
                )

                row = {
                    "sampleID": samples[i],
                    "x": pred_longlat[0, 0],
                    "y": pred_longlat[0, 1],
                }
                all_predictions.append(row)

            del model
            tf.keras.backend.clear_session()

        self.loo_score_l2_ = np.array(fold_penalties)

        if shap_background_size is not None:
            self.loo_shap_ = {
                "sample_ids": np.array(
                    [r["sampleID"] for r in all_predictions], dtype=str
                ),
                # Plain string arrays, so that np.load() reads them without pickle.
                "snp_ids": np.asarray(all_snp_ids, dtype=str)[kept_snps],
                "outputs": np.array(["x", "y"] + cov_names[:num_covs]),
                "values": (
                    np.stack(fold_shap)
                    if fold_shap
                    else np.empty((0, len(kept_snps), 2 + num_covs))
                ),
            }
        return pd.DataFrame(all_predictions)

    def _loo_fold_shap(self, model, traingen, predgen, num_covs, bg_size, rng):
        """SHAP values of one held-out sample: array (n_snps, n_outputs)."""
        import shap

        bg_size = min(bg_size, traingen.shape[0])
        background = traingen[rng.choice(traingen.shape[0], bg_size, replace=False)]
        background = background.astype(np.float32)
        x = predgen.astype(np.float32)
        if num_covs > 0:
            heads = [
                tf.keras.Model(inputs=model.input, outputs=model.output[0]),
                tf.keras.Model(inputs=model.input, outputs=model.output[1]),
            ]
        else:
            heads = [model]
        parts = [shap.GradientExplainer(h, background).shap_values(x)[0] for h in heads]
        return np.concatenate(parts, axis=1)

    def shap_values(
        self,
        genotype_path: str,
        sample_data_path: str,
        train_genotype_path: str,
        train_sample_data_path: str,
        background_size: int = 100,
        min_maf: float = None,
        seed: int = None,
        raw: bool = False,
    ) -> pd.DataFrame:
        import shap

        if not hasattr(self, "model_"):
            raise RuntimeError("EcoLocator must be fitted before calling shap_values")

        rng = np.random.default_rng(seed if seed is not None else self.seed_)

        # load and filter training genos for bg
        train_genotypes, train_samples, train_snp_ids = self._get_genotypes(
            train_genotype_path
        )
        train_genotypes = train_genotypes[self._kept_snp_indices_, :, :]
        train_ac = replace_missing_data(train_genotypes, rng=rng)
        _, train_locs = sort_samples(train_samples, train_sample_data_path)
        known = np.argwhere(~np.isnan(train_locs[:, 0])).flatten()
        traingen = np.transpose(train_ac[:, known]).astype(np.float32)

        # load and filter pred genos
        pred_genotypes, pred_samples, _ = self._get_genotypes(genotype_path)
        pred_genotypes = pred_genotypes[self._kept_snp_indices_, :, :]
        pred_ac = replace_missing_data(pred_genotypes)
        _, pred_locs = sort_samples(pred_samples, sample_data_path)
        unknown = np.argwhere(np.isnan(pred_locs[:, 0])).flatten()
        predgen = np.transpose(pred_ac[:, unknown]).astype(np.float32)

        # build bg
        bg_size = min(background_size, traingen.shape[0])
        bg_idx = rng.choice(traingen.shape[0], bg_size, replace=False)
        background = traingen[bg_idx, :]

        # wrap model heads
        if self.num_covs_ > 0:
            model_loc = tf.keras.Model(
                inputs=self.model_.input, outputs=self.model_.output[0]
            )
        else:
            model_loc = self.model_

        # compute shap
        expl_loc = shap.GradientExplainer(model_loc, background)
        shap_loc = expl_loc.shap_values(predgen)

        if self.num_covs_ > 0:
            model_env = tf.keras.Model(
                inputs=self.model_.input, outputs=self.model_.output[1]
            )
            expl_env = shap.GradientExplainer(model_env, background)
            shap_env = expl_env.shap_values(predgen)

        snp_ids = train_snp_ids[self._kept_snp_indices_]

        # apply min_maf filter to snp index list before building either format
        if min_maf is not None:
            af = np.mean(traingen, axis=0) / 2.0
            maf = np.minimum(af, 1 - af)
            mask = maf >= min_maf
            snp_ids = snp_ids[mask]
            shap_loc = shap_loc[:, mask, :]
            if self.num_covs_ > 0:
                shap_env = shap_env[:, mask, :]

        n_samples = predgen.shape[0]
        if n_samples == 1:
            logging.warning(
                "Attributing 1 sample — output is absolute SHAP values "
                "for that sample, not a mean across multiple samples."
            )

        if raw:
            rows = []
            for k, sample in enumerate(pred_samples[unknown]):
                row = {"sampleID": sample}
                for col_idx, snp_id in enumerate(snp_ids):
                    row[f"{snp_id}_x"] = shap_loc[k, col_idx, 0]
                    row[f"{snp_id}_y"] = shap_loc[k, col_idx, 1]
                    for j, cov in enumerate(self.cov_names_):
                        row[f"{snp_id}_{cov}"] = shap_env[k, col_idx, j]
                rows.append(row)
            return pd.DataFrame(rows)

        # default: mean absolute SHAP summary — rows=SNPs, cols=output variables
        output_cols = ["x", "y"] + self.cov_names_
        summary = {"snp_id": snp_ids}
        for var_idx, var in enumerate(output_cols):
            if var_idx < 2:
                vals = shap_loc[:, :, var_idx]  # shape (n_samples, n_snps)
            else:
                vals = shap_env[:, :, var_idx - 2]
            summary[var] = np.abs(vals).mean(axis=0)
        return pd.DataFrame(summary)

    def attribute(
        self,
        genotype_path: str,
        sample_data_path: str,
        background_size: int = 100,
        n_samples: int = 200,
        seed: int = None,
    ) -> pd.DataFrame:
        """SNP attributions of the covariate predictions, split by path.

        Samples with missing ``x``/``y`` in ``sample_data_path`` are
        attributed; up to ``background_size`` of the others, drawn at
        random, form the background. Returns one row per SNP: ``snp_id``,
        ``maf`` (over all samples of ``genotype_path``), and for each
        covariate ``c``:

        - ``c_weight``: the SNP's weight in the polygenic score of ``c``;
        - ``c_score``: mean over the attributed samples of
          ``|weight * z|``, with ``z`` the genotype standardized as the
          model does it: the exact SHAP value of the score alone, against
          the mean genotype of the training samples;
        - ``c_full``: mean ``|SHAP value|`` of the prediction of ``c`` by
          expected gradients (``n_samples`` draws per sample), through
          every path from the genotypes;
        - ``c_direct`` and ``c_location``: mean ``|SHAP value|`` of its two
          parts (network.path_attributions), the path through the score
          (the score and the correction head's use of it) and the path
          through the trunk and the predicted location. For each sample
          the signed parts add up to the signed full value.

        The values are in the units of the network's covariate output:
        SDs of the covariate in the data given to ``fit()``, after its
        transform (``cov_transforms``).
        """
        if not hasattr(self, "model_"):
            raise RuntimeError("EcoLocator must be fitted before calling attribute")
        if self.network_ != "pgs":
            raise ValueError(
                "attribute() needs the EcoLocator network; for a model of "
                "the dense network (build_network), use shap_values()"
            )
        if self.num_covs_ == 0:
            raise ValueError("the model has no covariates to attribute")
        rng = np.random.default_rng(seed if seed is not None else self.seed_)

        genotypes, samples, snp_ids = self._get_genotypes(genotype_path)
        genotypes = genotypes[self._kept_snp_indices_, :, :]
        gen = np.transpose(replace_missing_data(genotypes, rng=rng)).astype(np.float32)
        _, locs = sort_samples(samples, sample_data_path)
        unknown = np.isnan(locs[:, 0])
        known = np.flatnonzero(~unknown)
        if not unknown.any():
            raise ValueError("no sample to attribute: none has missing x/y")
        if len(known) == 0:
            raise ValueError("no background samples: every sample has missing x/y")
        background = gen[
            rng.choice(known, min(background_size, len(known)), replace=False)
        ]
        x = gen[unknown]

        direct, location = path_attributions(self.model_, x, background, n_samples, rng)
        z = tf.keras.Model(
            inputs=self.model_.input,
            outputs=self.model_.get_layer("standardize").output,
        )(x, training=False).numpy()
        freq = gen.mean(axis=0) / 2
        table = {
            "snp_id": np.asarray(snp_ids)[self._kept_snp_indices_],
            "maf": np.minimum(freq, 1 - freq),
        }
        for c, name in enumerate(self.cov_names_):
            weight = self.model_.get_layer(f"score_{c}").get_weights()[0][:, 0]
            table[f"{name}_weight"] = weight
            table[f"{name}_score"] = np.abs(z * weight).mean(axis=0)
            table[f"{name}_full"] = np.abs(direct[:, :, c] + location[:, :, c]).mean(
                axis=0
            )
            table[f"{name}_direct"] = np.abs(direct[:, :, c]).mean(axis=0)
            table[f"{name}_location"] = np.abs(location[:, :, c]).mean(axis=0)
        return pd.DataFrame(table)

    def _get_genotypes(self, genotype_path: str):
        if genotype_path.endswith(".zarr"):
            return load_genotypes(zarr_path=genotype_path)
        elif genotype_path.endswith((".vcf", ".vcf.gz")):
            return load_genotypes(vcf_path=genotype_path)
        else:
            return load_genotypes(matrix_path=genotype_path)
