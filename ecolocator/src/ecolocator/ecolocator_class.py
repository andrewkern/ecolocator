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
    transform_env,
    env_weights,
    env_errors,
)
from .network import (
    build_pgs_network,
    build_location_route,
    train_network,
    euclid_loss,
)

# Network settings, saved with a model and passed to build_pgs_network
# (except score_l2, which fit() resolves to one penalty first).
HYPERPARAMETERS = (
    "nlayers",
    "width",
    "dropout_prop",
    "input_dropout",
    "weight_decay",
    "head_width",
    "score_l2",
    "loc_weight",
    "env_weight",
)


class EcoLocator:
    """
    A neural network model for predicting sample locations and environmental
    covariates from genotype data.

    The network (network.build_pgs_network) has a location trunk and, for
    each covariate, a linear polygenic score of all SNPs plus a small
    correction head that also reads the predicted location. ``score_l2`` is
    the L2 penalty of the scores: a number fixes it; a sequence is searched,
    each fit keeping the penalty with the lowest covariate validation loss.

    After the network, fit() trains a location route
    (network.build_location_route): for each covariate, a small head that
    reads only the predicted location. ``env_weights_`` holds how much of
    each covariate the network predicts beyond its location route
    (utils.env_weights), and ``env_mse_`` the mean squared error of each
    covariate's prediction (utils.env_errors), both from cross-fitted
    predictions of all samples with known coordinates (``weight_folds`` in
    fit()). offset() weights each covariate by env_weights_ / env_mse_ for a
    genomic offset over several covariates.
    """

    def __init__(
        self,
        cov_transforms: list = None,
        nlayers: int = 4,
        width: int = 128,
        dropout_prop: float = 0.25,
        input_dropout: float = 0.1,
        weight_decay: float = 2e-4,
        head_width: int = 16,
        score_l2: float | tuple = (20.0, 60.0, 200.0, 600.0),
        loc_weight: float = 1.0,
        env_weight: float = 1.0,
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

    @classmethod
    def load(cls, path: str) -> "EcoLocator":
        with open(os.path.join(path, "params.json")) as f:
            params = json.load(f)

        # Models saved with the network of the submitted manuscript have
        # fewer settings; the missing ones keep their defaults. They predict
        # and explain as before.
        obj = cls(
            cov_transforms=params["cov_transforms"],
            **{k: params[k] for k in HYPERPARAMETERS if k in params},
        )

        arrays = np.load(os.path.join(path, "arrays.npz"))
        obj.meanlong_ = arrays["meanlong"].item()
        obj.sdlong_ = arrays["sdlong"].item()
        obj.meanlat_ = arrays["meanlat"].item()
        obj.sdlat_ = arrays["sdlat"].item()
        obj.means_ = arrays["means"]
        obj.sds_ = arrays["sds"]
        obj._kept_snp_indices_ = arrays["kept_snp_indices"]
        # Models saved before these arrays existed have none of them.
        obj.train_ids_ = arrays["train_ids"] if "train_ids" in arrays else None
        obj.crossfit_ids_ = arrays["crossfit_ids"] if "crossfit_ids" in arrays else None
        obj.crossfit_env_ = arrays["crossfit_env"] if "crossfit_env" in arrays else None

        obj.cov_names_ = params["cov_names"]
        obj.seed_ = params["seed"]
        obj.num_covs_ = len(obj.cov_names_)
        obj.transforms_ = params["transforms"]
        obj.score_l2_ = params.get("score_l2_fitted")
        weights = params.get("env_weights")
        obj.env_weights_ = None if weights is None else np.array(weights)
        errors = params.get("env_mse")
        obj.env_mse_ = None if errors is None else np.array(errors)

        # Models saved without a location route predict as before;
        # predict(location_route=True) and offset() need one.
        route_path = os.path.join(path, "location_route.keras")
        obj.location_route_ = (
            tf.keras.models.load_model(route_path)
            if os.path.exists(route_path)
            else None
        )

        obj.model_ = tf.keras.models.load_model(
            os.path.join(path, "model.keras"),
            custom_objects={"euclid_loss": euclid_loss},
        )

        return obj

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        self.model_.save(os.path.join(path, "model.keras"))
        if getattr(self, "location_route_", None) is not None:
            self.location_route_.save(os.path.join(path, "location_route.keras"))
        np.savez(
            os.path.join(path, "arrays.npz"),
            meanlong=self.meanlong_,
            sdlong=self.sdlong_,
            meanlat=self.meanlat_,
            sdlat=self.sdlat_,
            means=self.means_,
            sds=self.sds_,
            kept_snp_indices=self._kept_snp_indices_,
            # Plain string arrays, so that np.load() reads them without pickle.
            **(
                {"train_ids": self.train_ids_}
                if getattr(self, "train_ids_", None) is not None
                else {}
            ),
            **(
                {"crossfit_ids": self.crossfit_ids_, "crossfit_env": self.crossfit_env_}
                if getattr(self, "crossfit_env_", None) is not None
                else {}
            ),
        )
        params = {
            "seed": self.seed_,
            "cov_names": self.cov_names_,
            "num_covs": self.num_covs_,
            "transforms": self.transforms_,
            "cov_transforms": self.cov_transforms,
            "score_l2_fitted": self.score_l2_,
            "env_weights": (
                None
                if getattr(self, "env_weights_", None) is None
                else [float(w) for w in self.env_weights_]
            ),
            "env_mse": (
                None
                if getattr(self, "env_mse_", None) is None
                else [float(e) for e in self.env_mse_]
            ),
            **{k: getattr(self, k) for k in HYPERPARAMETERS},
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
        weight_folds: int = 5,
    ) -> "EcoLocator":
        """Fit the network on the samples with known coordinates.

        With covariates, fit() then trains the location route and computes
        ``env_weights_``. With ``weight_folds`` of 2 or more, the weights
        come from that many folds of the known samples: for each fold, the
        network (with the penalty this fit chose) and its location route
        are fitted on the other folds, and the fold's samples are
        predicted. This costs ``weight_folds`` more network fits. With
        ``weight_folds`` below 2, the weights come from the validation
        samples of this fit only: cheaper, but noisy when there are few
        validation samples."""
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
        self.train_ids_ = np.asarray(samples[np.concatenate([train, test])], dtype=str)
        self.model_, self.history_, self.score_l2_losses_ = self._fit_network(
            traingen,
            testgen,
            trainlocs,
            testlocs,
            num_covs,
            seed,
            max_epochs,
            batch_size,
            patience,
            verbose,
        )
        self.score_l2_ = min(self.score_l2_losses_, key=self.score_l2_losses_.get)
        self.location_route_, self.env_weights_, self.env_mse_ = None, None, None
        self.crossfit_ids_, self.crossfit_env_ = None, None
        if num_covs > 0:
            self.location_route_ = self._fit_location_route(
                self.model_,
                traingen,
                testgen,
                trainlocs,
                testlocs,
                seed,
                max_epochs,
                patience,
            )
            if weight_folds >= 2:
                self.env_weights_, self.env_mse_ = self._crossfit_weights(
                    ac,
                    samples,
                    np.concatenate([train, test]),
                    locs,
                    weight_folds,
                    train_split,
                    seed,
                    rng,
                    max_epochs,
                    batch_size,
                    patience,
                    verbose,
                )
            if self.env_weights_ is None:
                val_loc, val_env = self.model_.predict(testgen, verbose=0)
                self.env_weights_ = env_weights(
                    testlocs[1],
                    val_env,
                    self.location_route_.predict(val_loc, verbose=0),
                )
                self.env_mse_ = env_errors(testlocs[1], val_env)
        return self

    def _crossfit_weights(
        self,
        ac,
        samples,
        known,
        locs,
        folds,
        train_split,
        seed,
        rng,
        max_epochs,
        batch_size,
        patience,
        verbose,
    ):
        """Covariate weights and prediction errors from out-of-fold
        predictions of the known samples (indices ``known`` into ac and the
        normalized locs). Each fold's network uses the penalty of the main
        fit. The out-of-fold predictions (normalized units) are kept in
        crossfit_ids_ and crossfit_env_, for offset(). Returns (None, None)
        (with a warning) when a fold would leave fewer than 2 samples to
        fit."""
        folds = min(folds, len(known))
        smallest_rest = len(known) - int(np.ceil(len(known) / folds))
        if smallest_rest - max(1, round((1 - train_split) * smallest_rest)) < 2:
            logging.warning(
                f"Too few samples ({len(known)}) for {folds}-fold covariate weights; "
                "using the validation samples of the main fit."
            )
            return None, None
        order = rng.permutation(known)
        fold_of = np.arange(len(order)) % folds
        observed, predicted, routes, ids = [], [], [], []
        for k in range(folds):
            held, rest = order[fold_of == k], order[fold_of != k]
            val = rng.choice(
                rest, max(1, round((1 - train_split) * len(rest))), replace=False
            )
            fit_idx = np.setdiff1d(rest, val)
            fold_seed = (
                None
                if seed is None
                else int(np.random.SeedSequence([seed, 1, k]).generate_state(1)[0])
            )
            gen = {
                n: np.transpose(ac[:, i])
                for n, i in (("fit", fit_idx), ("val", val), ("held", held))
            }
            fit_locs = [locs[fit_idx][:, 0:2], locs[fit_idx][:, 2:]]
            val_locs = [locs[val][:, 0:2], locs[val][:, 2:]]
            model, _, _ = self._fit_network(
                gen["fit"],
                gen["val"],
                fit_locs,
                val_locs,
                self.num_covs_,
                fold_seed,
                max_epochs,
                batch_size,
                patience,
                verbose,
                penalties=[self.score_l2_],
            )
            route = self._fit_location_route(
                model,
                gen["fit"],
                gen["val"],
                fit_locs,
                val_locs,
                fold_seed,
                max_epochs,
                patience,
            )
            held_loc, held_env = model.predict(gen["held"], verbose=0)
            observed.append(locs[held][:, 2:])
            predicted.append(np.reshape(held_env, (len(held), -1)))
            ids.append(samples[held])
            routes.append(
                np.reshape(route.predict(held_loc, verbose=0), (len(held), -1))
            )
        observed, predicted = np.concatenate(observed), np.concatenate(predicted)
        self.crossfit_ids_ = np.asarray(np.concatenate(ids), dtype=str)
        self.crossfit_env_ = predicted
        return (
            env_weights(observed, predicted, np.concatenate(routes)),
            env_errors(observed, predicted),
        )

    def _fit_location_route(
        self,
        model,
        traingen,
        testgen,
        trainlocs,
        testlocs,
        seed,
        max_epochs,
        patience,
        draws: int = 5,
    ):
        """Train a location route on a fitted network's predicted locations.

        The route's training inputs are the predicted locations of the
        training samples with dropout active, ``draws`` times: the network
        predicts training samples more closely than new samples, and
        dropout gives inputs with error, as the correction heads see them
        in training. Validation samples (predicted without dropout) stop the
        training early. Returns the route."""
        if seed is not None:
            tf.keras.utils.set_random_seed(seed)
        noisy = [
            np.concatenate(
                [
                    model(traingen[i : i + 1024], training=True)[0].numpy()
                    for i in range(0, len(traingen), 1024)
                ]
            )
            for _ in range(draws)
        ]
        val_loc = model.predict(testgen, verbose=0)[0]
        route = build_location_route(self.num_covs_, self.head_width, self.weight_decay)
        # The route has a few hundred weights, so Keras's cost per step
        # dominates its training: large batches, and early stopping that
        # needs a real improvement (min_delta), keep it to seconds.
        route.fit(
            np.concatenate(noisy),
            np.tile(trainlocs[1], (draws, 1)),
            validation_data=(val_loc, testlocs[1]),
            epochs=min(max_epochs, 1000),
            batch_size=256,
            shuffle=True,
            verbose=0,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    min_delta=1e-4,
                    patience=min(patience, 20),
                    restore_best_weights=True,
                )
            ],
        )
        return route

    def _fit_network(
        self,
        traingen,
        testgen,
        trainlocs,
        testlocs,
        num_covs,
        seed,
        max_epochs,
        batch_size,
        patience,
        verbose,
        penalties=None,
    ):
        """Fit the network on one split: traingen/trainlocs to fit, and
        testgen/testlocs for early stopping and the choice of the score
        penalty. Genotypes are standardized with the means and SDs of both
        sets together. The network is fitted once per penalty of score_l2
        (or of ``penalties``),
        from the same seed, and the fit with the lowest covariate validation
        loss (MSE) is kept. Returns that model, its history, and each
        penalty's validation loss."""
        known = np.concatenate([traingen, testgen])
        geno_mean, geno_sd = known.mean(axis=0), known.std(axis=0)
        if penalties is None:
            penalties = (
                [0.0]
                if num_covs == 0
                else [float(p) for p in np.atleast_1d(self.score_l2)]
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
            losses[penalty] = float(h[key][int(np.argmin(h["val_loss"]))])
            if best is None or losses[penalty] < best[2]:
                best = (model, history, losses[penalty])
        return best[0], best[1], losses

    def predict(
        self,
        genotype_path: str,
        sample_data_path: str,
        location_route: bool = False,
    ) -> pd.DataFrame:
        """Predict location and covariates of the samples with missing x/y.

        With ``location_route``, also each covariate's location-route
        prediction (column <covariate>_location): what the predicted
        location alone tells about it."""
        genotypes, samples, _ = self._get_genotypes(genotype_path)
        _, locs = sort_samples(samples, sample_data_path)
        pred = np.argwhere(np.isnan(locs[:, 0])).flatten()
        return self._predict_rows(genotypes, samples, pred, location_route)

    def _predict_rows(self, genotypes, samples, pred, location_route=False):
        """Predictions of the samples at indices ``pred`` (see predict)."""
        genotypes = genotypes[self._kept_snp_indices_, :, :]
        ac = replace_missing_data(genotypes)
        predgen = np.transpose(ac[:, pred])

        prediction = self.model_.predict(predgen, verbose=0)
        if location_route:
            if self.num_covs_ == 0 or getattr(self, "location_route_", None) is None:
                raise ValueError("this model has no location route")
            route = back_transform_env(
                self.location_route_.predict(prediction[0], verbose=0).reshape(
                    len(pred), -1
                ),
                self.means_,
                self.sds_,
                self.transforms_,
            )
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
            if location_route:
                for j, name in enumerate(self.cov_names_):
                    result[f"{name}_location"] = route[:, j]
            result.insert(0, "sampleID", samples[pred])
            return result
        else:
            pred_longlat = prediction * np.array(
                [self.sdlong_, self.sdlat_]
            ) + np.array([self.meanlong_, self.meanlat_])

            result = pd.DataFrame(pred_longlat, columns=["x", "y"])
            result.insert(0, "sampleID", samples[pred])
            return result

    @property
    def offset_weights_(self) -> np.ndarray:
        """The weights offset() uses by default: env_weights_ / env_mse_
        (env_weights_ alone for a model without env_mse_)."""
        if getattr(self, "env_weights_", None) is None:
            return None
        errors = getattr(self, "env_mse_", None)
        if errors is None:
            return np.asarray(self.env_weights_, dtype=float)
        return np.asarray(self.env_weights_, dtype=float) / np.maximum(errors, 1e-12)

    @property
    def crossfit_predictions_(self) -> pd.DataFrame:
        """Out-of-fold covariate predictions of the training samples, from
        the weight folds of fit() (sampleID and one column per covariate);
        None if fit() ran without weight folds."""
        if getattr(self, "crossfit_env_", None) is None:
            return None
        env = back_transform_env(
            np.array(self.crossfit_env_, dtype=float),
            self.means_,
            self.sds_,
            self.transforms_,
        )
        result = pd.DataFrame(env, columns=self.cov_names_)
        result.insert(0, "sampleID", self.crossfit_ids_)
        return result

    def offset(
        self,
        genotype_path: str,
        sample_data_path: str,
        sites: pd.DataFrame,
        weights: np.ndarray = None,
        scale_by_error: bool = True,
    ) -> pd.DataFrame:
        """Genomic offset of every sample in the sample data to each site.

        ``sites`` has one row per site (for example a common garden or a
        future climate) and a column for every covariate; its index names
        the sites. The offset of sample i to site g is the weighted
        Euclidean distance between the covariates EcoLocator predicts for i
        and those of g, in normalized units:
        sqrt(sum_c (w_c / e_c) (z_gc - zhat_ic)^2).

        Every prediction comes from a model that did not see the sample:
        samples that fit() trained on get their out-of-fold prediction from
        the weight folds (crossfit_predictions_); all other samples (missing
        x/y, or not in the training data) get the fitted model's
        prediction. Training samples therefore need a fit with
        weight_folds >= 2.

        ``weights`` (w_c) defaults to ``env_weights_``, so covariates that
        the genotypes predict only through location count little. e_c is
        ``env_mse_``, the covariate's cross-fitted prediction error, so
        covariates the network predicts poorly count less; with
        ``scale_by_error=False``, or for a model without env_mse_, e_c = 1.

        Returns a DataFrame: sampleID, site, offset."""
        if self.num_covs_ == 0:
            raise ValueError("offset() needs a model with covariates")
        if weights is None:
            if getattr(self, "env_weights_", None) is None:
                raise ValueError("this model has no env_weights_; give weights")
            weights = self.env_weights_
        weights = np.asarray(weights, dtype=float)
        errors = getattr(self, "env_mse_", None)
        if scale_by_error and errors is not None:
            weights = weights / np.maximum(np.asarray(errors, dtype=float), 1e-12)

        genotypes, samples, _ = self._get_genotypes(genotype_path)
        _, locs = sort_samples(samples, sample_data_path)
        samples = np.asarray(samples, dtype=str)
        trained = getattr(self, "train_ids_", None)
        if trained is None:
            # A model saved before train_ids_ existed: only samples with
            # missing x/y are certainly unseen.
            unseen = np.isnan(locs[:, 0])
            logging.warning(
                "this model does not record its training samples; offsets "
                "only for samples with missing x/y"
            )
        else:
            unseen = np.isnan(locs[:, 0]) | ~np.isin(samples, trained)
        frames = []
        new = np.flatnonzero(unseen)
        if len(new):
            pred = self._predict_rows(genotypes, samples, new)
            frames.append(
                pd.DataFrame(
                    transform_env(
                        pred[self.cov_names_].to_numpy(),
                        self.means_,
                        self.sds_,
                        self.transforms_,
                    ),
                    index=pred.sampleID,
                )
            )
        seen = samples[~unseen] if trained is not None else np.array([], dtype=str)
        if len(seen):
            if getattr(self, "crossfit_env_", None) is None:
                raise ValueError(
                    f"{len(seen)} samples were used to fit this model; their offsets "
                    "need out-of-fold predictions: fit with weight_folds >= 2"
                )
            cv = pd.DataFrame(
                np.array(self.crossfit_env_, dtype=float), index=self.crossfit_ids_
            )
            frames.append(cv.loc[seen])
        z_pred = pd.concat(frames).to_numpy()
        ids = pd.concat(frames).index
        z_site = transform_env(
            sites[self.cov_names_].to_numpy(), self.means_, self.sds_, self.transforms_
        )
        rows = [
            pd.DataFrame(
                {
                    "sampleID": ids,
                    "site": site,
                    "offset": np.sqrt(((z - z_pred) ** 2) @ weights),
                }
            )
            for site, z in zip(sites.index, z_site)
        ]
        return pd.concat(rows, ignore_index=True)

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

            model, _, losses = self._fit_network(
                traingen,
                testgen,
                trainlocs,
                testlocs,
                num_covs,
                fold_seed,
                max_epochs,
                batch_size,
                patience,
                verbose,
            )
            fold_penalties.append(min(losses, key=losses.get))

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

    def _get_genotypes(self, genotype_path: str):
        if genotype_path.endswith(".zarr"):
            return load_genotypes(zarr_path=genotype_path)
        elif genotype_path.endswith((".vcf", ".vcf.gz")):
            return load_genotypes(vcf_path=genotype_path)
        else:
            return load_genotypes(matrix_path=genotype_path)
