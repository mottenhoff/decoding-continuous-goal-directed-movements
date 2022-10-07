import numpy as np

import PSID

def forward_feature_selection(n_features, n_folds, y, z, nx, n1, i):
        
    folds = np.array_split(np.arange(y.shape[0]), n_folds)
    
    features =          np.arange(y.shape[1])  # list of indices
    features_selected = np.array([], dtype=np.int32)           # list of indices
    max_scores =        np.array([])

    while features_selected.size < n_features:
        
        scores = np.array([])
        for feature in features:
            
            current_features = np.append(features_selected, feature)

            current_scores = np.array([])
            for fold in folds:
                y_test, y_train = y[np.ix_(fold, current_features)], np.delete(y[:, current_features], fold, axis=0)
                z_test, z_train = z[fold], np.delete(z, fold, axis=0)

                y_test = np.expand_dims(y_test, axis=1) if y_test.ndim == 1 else y_test

                id_sys = PSID.PSID(y_train, z_train, nx, n1, i,
                                    zscore_Y=True, zscore_Z=True)
                zh, yh, xh = id_sys.predict(y_test)

                # TODO: What to optimize for? zh, yh, xh?
                #       corr, RMSE, ... ?
                corr = np.corrcoef(z_test, zh, rowvar=False)

                current_scores = np.append(current_scores,
                                            np.array([corr[0, 3], corr[1, 4], corr[2, 5]]))

            scores = np.append(scores, np.mean(current_scores))


        # TODO: Score progression            
        best_feature = np.argmax(scores)
        features_selected = np.append(features_selected, best_feature)
        features = np.delete(features, best_feature)
        max_scores = np.append(max_scores, np.max(scores))
        print(f'Best current selection: {features_selected} with r={np.max(scores):.2f}')

        # TODO: stop if 2x decrease in a row and return to best feature set
        if features_selected.size > 2:
            if features_selected[-1] < features_selected[-3] and \
                features_selected[-2] < features_selected[-3]:

                i = np.argmax(max_scores)

                print(f'Performance decrease, returning best previous option [n={i}  r={np.max(scores):.2f}]')
                return features_selected[:i+1]

    return features_selected
