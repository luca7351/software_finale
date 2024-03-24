from numpy import ravel
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import time
import statistics


# ##############################################################################
#                                 REPORT 1
# ##############################################################################
# Primo dei sette report, si utilizza il Multi Layer Perceptron sul dataset
# relativo al soggetto 25 parametrizzando il max_iter del modello con valori
# (100, 150, 200, 250, 300) e si visualizza la configurazione ottimale e lo score
# --> __init__() : imposta il path del report
# --> exec() : esegue quanto riportato sopra
class Report1:
    def __init__(self, path_gen):
        self.path = path_gen
        self.path_file = path_gen + "whole_data_25.csv"

    def exec(self):
        total_data = read_csv(self.path_file)
        training_data, testing_data, training_labels, testing_labels = train_test_split(
            total_data.iloc[:, 7:len(total_data.columns)], total_data.iloc[:, 6])

        mlp = MLPClassifier()
        parameters = {'max_iter': (100, 150, 200, 250, 300)}
        gs = GridSearchCV(mlp, parameters)
        gs.fit(training_data, ravel(training_labels))
        print("Best param:")
        print(gs.best_params_)

        config_path = self.path + 'modelConfiguration.json'
        with open(config_path, 'w') as f:
            json.dump(gs.best_params_, f)
        with open(config_path, 'r') as f:
            params = json.load(f)

        model = MLPClassifier(**params).fit(training_data, ravel(training_labels))
        labels = model.predict(testing_data)
        score = accuracy_score(ravel(testing_labels), labels)
        print("Score:")
        print(score)


# ##############################################################################
#                                 REPORT 2
# ##############################################################################
# Secondo dei sette report, si utilizza il Multi Layer Perceptron sul dataset
# totale si visualizza lo score e le iterazione compiute
# --> __init__() : imposta il path del report
# --> exec() : esegue quanto riportato sopra
class Report2:
    def __init__(self, path_gen):
        self.path = path_gen
        self.path_file = path_gen + "EMOCON_VALENCE.CSV"

    def exec(self):
        total_data = read_csv(self.path_file)

        training_data, testing_data, training_labels, testing_labels = train_test_split(
            total_data.iloc[:, 1:len(total_data.columns)], total_data.iloc[:, 0])

        mlp = MLPClassifier(max_iter=200).fit(training_data, ravel(training_labels))

        labels = mlp.predict(testing_data)
        n_iter = mlp.n_iter_
        score = accuracy_score(ravel(testing_labels), labels)
        print("Score:")
        print(score)
        print("Iterazioni:")
        print(n_iter)


# ##############################################################################
#                                 REPORT 3
# ##############################################################################
# Terzo dei sette report, si utilizza il MLP, il Random Forest e il Gradient Boosting
# sul dataset totale per effettuare un primo confrontro tra i tre modelli
# --> __init__() : imposta il path del report
# --> exec_gen() : effettua una prova iniziale stampando lo score di una iterazione
#                  dei modelli Random forest e Gradient boosting
# --> exec_mlp() : effettua 5 iterazioni del modello MLP stampando, a termine,
#                  le statistiche degli score misurati
# --> exec_grd() : effettua 5 iterazioni del modello Gradient Boosting stampando, a termine,
#                  le statistiche degli score misurati
# --> exec_rnd() : effettua 5 iterazioni del modello Random Forest stampando, a termine,
#                  le statistiche degli score misurati
# --> exec() : manda in esecuzione le funzioni definite sopra
class Report3:
    def __init__(self, path_gen):
        self.path = path_gen
        self.path_file = path_gen + "EMOCON_VALENCE.CSV"

    def exec_gen(self):
        total_data = read_csv(self.path_file)

        training_data, testing_data, training_labels, testing_labels = train_test_split(
            total_data.iloc[:, 1:len(total_data.columns)], total_data.iloc[:, 0])

        model1 = RandomForestClassifier()
        model1.fit(training_data, ravel(training_labels))

        predicted_labels1 = model1.predict(testing_data)
        score1 = accuracy_score(ravel(testing_labels), predicted_labels1)
        print('Score RandomForestClassifier: %1.2f' % score1)

        model2 = GradientBoostingClassifier()
        model2.fit(training_data, ravel(training_labels))

        predicted_labels2 = model2.predict(testing_data)
        score2 = accuracy_score(ravel(testing_labels), predicted_labels2)
        print('Score GradientBoostingClassifier: %1.2f' % score2)

    def exec_mlp(self):
        total_data = read_csv(self.path_file)

        i = 0
        all_score = [0, 0, 0, 0, 0]
        while i < 5:
            training_data, testing_data, training_labels, testing_labels = train_test_split(
                total_data.iloc[:, 1:len(total_data.columns)], total_data.iloc[:, 0])

            model = MLPClassifier()
            model.fit(training_data, ravel(training_labels))

            predicted_labels = model.predict(testing_data)
            score = accuracy_score(ravel(testing_labels), predicted_labels)
            print('(Tentativo %1.0f) Score MLPClassifier: %1.8f' % (i + 1, score))
            all_score[i] = score
            i = i + 1

        array_scores = pd.DataFrame(all_score, columns=['Score stats'])
        print(array_scores.describe())

    def exec_grd(self):
        total_data = read_csv(self.path_file)

        i = 0
        all_score = [0, 0, 0, 0, 0]

        while i < 5:
            training_data, testing_data, training_labels, testing_labels = train_test_split(
                total_data.iloc[:, 1:len(total_data.columns)], total_data.iloc[:, 0])

            model = GradientBoostingClassifier()
            model.fit(training_data, ravel(training_labels))

            predicted_labels = model.predict(testing_data)
            score = accuracy_score(ravel(testing_labels), predicted_labels)
            print('(Tentativo %1.0f) Score GradientBoostingClassifier: %1.8f' % (i + 1, score))
            all_score[i] = score
            i = i + 1

        array_scores = pd.DataFrame(all_score, columns=['Score stats'])
        print(array_scores.describe())

    def exec_rnd(self):
        total_data = read_csv(self.path_file)

        i = 0
        all_score = [0, 0, 0, 0, 0]

        while i < 5:
            training_data, testing_data, training_labels, testing_labels = train_test_split(
                total_data.iloc[:, 1:len(total_data.columns)], total_data.iloc[:, 0])

            model = RandomForestClassifier()
            model.fit(training_data, ravel(training_labels))

            predicted_labels = model.predict(testing_data)
            score = accuracy_score(ravel(testing_labels), predicted_labels)
            print('(Tentativo %1.0f) Score RandomForestClassifier: %1.8f' % (i + 1, score))
            all_score[i] = score
            i = i + 1

        array_scores = pd.DataFrame(all_score, columns=['Score stats'])
        print(array_scores.describe())

    def exec(self):
        self.exec_gen()
        self.exec_mlp()
        self.exec_grd()
        self.exec_rnd()


# ##############################################################################
#                                 REPORT 4
# ##############################################################################
# Quarto dei sette report, si analizzano e mostrano le features importances dei
# due modelli Gradient Boosting e Random Forest
# --> __init__() : imposta il path del report
# --> feat_imp_rnd() : addesstra un modello Random Forest e calcola le features
#                       importance
# --> feat_imp_grd() : addesstra un modello Gradient Boosting e calcola le
#                       features importance
# --> zero_values() : Calcola i valori a zero delle variabili Meditation e Attention
# --> exec() : manda in esecuzione le funzioni definite sopra
class Report4:
    def __init__(self, path_gen):
        self.path = path_gen
        self.path_file = path_gen + 'EMOCON_VALENCE.CSV'

    def feat_imp_rnd(self):
        total_data = read_csv(self.path_file)

        training_data, testing_data, training_labels, testing_labels = train_test_split(
            total_data.iloc[:, 1:len(total_data.columns)], total_data.iloc[:, 0])

        features_names = list(training_data.columns)
        model = RandomForestClassifier()
        model.fit(training_data, ravel(training_labels))

        importances = model.feature_importances_
        std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)

        model_importances = pd.Series(importances, index=features_names)

        fig, ax = plt.subplots()
        model_importances.plot.bar(yerr=std, ax=ax)
        ax.set_title("Feature importances random forest")

        fig.tight_layout()

    def feat_imp_grd(self):
        total_data = read_csv(self.path_file)

        training_data, testing_data, training_labels, testing_labels = train_test_split(
            total_data.iloc[:, 1:len(total_data.columns)], total_data.iloc[:, 0])

        features_names = list(training_data.columns)
        model = GradientBoostingClassifier()
        model.fit(training_data, ravel(training_labels))

        importances = model.feature_importances_

        model_importances = pd.Series(importances, index=features_names)

        fig, ax = plt.subplots()
        model_importances.plot.bar(ax=ax)
        ax.set_title("Feature importances gradient boosting")

        fig.tight_layout()

    def zero_values(self):
        total_data = read_csv(self.path_file)
        n_total = total_data.shape[0]
        preprocessed = total_data.iloc[:, [9, 18]]
        print(preprocessed.describe())
        count0_attention = (preprocessed['Attention'] == 0).sum()
        count0_meditation = (preprocessed['Meditation'] == 0).sum()
        print('''Valori pari a 0 per la variabile Meditation: %1.0f su %1.0f cioè %1.5f''' % (
            count0_meditation, n_total, count0_meditation / n_total))
        print('Valori pari a 0 per la variabile Attention: %1.0f su %1.0f cioè %1.5f' % (
            count0_attention, n_total, count0_attention / n_total))

    def exec(self):
        self.feat_imp_rnd()
        self.feat_imp_grd()
        self.zero_values()


# ##############################################################################
#                                 REPORT 5
# ##############################################################################
# Quinto dei sette report, si analizzano le prestazioni dei due modelli Random
# Forest e Gradient Boosting dei sottoinsiemi di features dati dai dispositivi
# H4 e H7
# --> __init__() : imposta il path del report
# --> rnd_h4() : score del Random Forest su caratteristiche H4
# --> grd_h4() : score del Gradient Boosting su caratteristiche H4
# --> rnd_h7() : score del Random Forest su caratteristiche H7
# --> grd_h7() : score del Gradient Boosting su caratteristiche H7
# --> exec() : manda in esecuzione le funzioni definite sopra
class Report5:
    def __init__(self, path_gen):
        self.path = path_gen
        self.path_file = path_gen + 'EMOCON_VALENCE.CSV'

    def rnd_h4(self):
        total_data = read_csv(self.path_file)

        i = 0
        all_score = [0, 0, 0, 0, 0]
        while i < 5:
            training_data, testing_data, training_labels, testing_labels = train_test_split(
                total_data.iloc[:, 1:9], total_data.iloc[:, 0])

            model = RandomForestClassifier()
            model.fit(training_data, ravel(training_labels))

            predicted_labels = model.predict(testing_data)
            score = accuracy_score(ravel(testing_labels), predicted_labels)
            print('(Tentativo %1.0f) E4 score RandomForestClassifier: %1.8f' % (i + 1, score))
            all_score[i] = score
            i = i + 1

        array_scores = pd.DataFrame(all_score, columns=['E4 score stats'])
        print(array_scores.describe())

    def rnd_h7(self):
        total_data = read_csv(self.path_file)

        i = 0
        all_score = [0, 0, 0, 0, 0]
        while i < 5:
            training_data, testing_data, training_labels, testing_labels = train_test_split(
                total_data.iloc[:, 10:19], total_data.iloc[:, 0])

            model = RandomForestClassifier()
            model.fit(training_data, ravel(training_labels))

            predicted_labels = model.predict(testing_data)
            score = accuracy_score(ravel(testing_labels), predicted_labels)
            print('(Tentativo %1.0f) E7 and Neurosky score RandomForestClassifier: %1.8f'
                  % (i + 1, score))
            all_score[i] = score
            i = i + 1

        array_scores = pd.DataFrame(all_score, columns=['E7 and Neurosky score stats'])
        print(array_scores.describe())

    def grd_h4(self):
        total_data = read_csv(self.path_file)

        i = 0
        all_score = [0, 0, 0, 0, 0]
        while i < 5:
            training_data, testing_data, training_labels, testing_labels = train_test_split(
                total_data.iloc[:, 1:9], total_data.iloc[:, 0])

            model = GradientBoostingClassifier()
            model.fit(training_data, ravel(training_labels))

            predicted_labels = model.predict(testing_data)
            score = accuracy_score(ravel(testing_labels), predicted_labels)
            print('(Tentativo %1.0f) E4 score GradientBoostingClassifier: %1.8f'
                  % (i + 1, score))
            all_score[i] = score
            i = i + 1

        array_scores = pd.DataFrame(all_score, columns=['E4 score stats'])
        print(array_scores.describe())

    def grd_h7(self):
        total_data = read_csv(self.path_file)

        i = 0
        all_score = [0, 0, 0, 0, 0]
        while i < 5:
            training_data, testing_data, training_labels, testing_labels = train_test_split(
                total_data.iloc[:, 10:19], total_data.iloc[:, 0])

            model = GradientBoostingClassifier()
            model.fit(training_data, ravel(training_labels))

            predicted_labels = model.predict(testing_data)
            score = accuracy_score(ravel(testing_labels), predicted_labels)
            print('(Tentativo %1.0f) E7 and Neurosky score GradientBoostingClassifier: %1.8f'
                  % (i + 1, score))
            all_score[i] = score
            i = i + 1

        array_scores = pd.DataFrame(all_score, columns=['E7 and Neurosky score stats'])
        print(array_scores.describe())

    def exec(self):
        self.rnd_h4()
        self.rnd_h7()
        self.grd_h4()
        self.grd_h7()


# ##############################################################################
#                                 REPORT 6
# ##############################################################################
# Seseto dei sette report, si analizzano le prestazioni dei due modelli Random
# Forest e Gradient Boosting andando ad analizzare un insieme di feature
# (inizialmente 3) sempre crescente, dove ogni iterazione si introsuce la feature
# più importante che non fa già parte del sottoinsieme.
# i risultati vengono stampati in formato tabellare e grafico
# --> __init__() : imposta il path del report
# --> fase1_rnd(): calcolo e ordinamente features per il random forest
# --> fase1_grd(): calcolo e ordinamente features per il gradient boosting
# --> fase2_rnd(): calcolo tempi di esecuzione medi e score medio per insiemi
#                  crescenti di features più importanti usando random forest
# --> fase2_grd(): calcolo tempi di esecuzione medi e score medio per insiemi
#                  crescenti di features più importanti usando gradient boosting
# --> fase3_rnd(): stampa risultati del random forest graficamente e tabella
# --> fase3_grd(): stampa risultati del gradient boosting graficamente e tabella
# --> exec(): esegue le funzioni riposrtate sopra in modo ordinato
class Report6:
    def __init__(self, path_gen):
        self.path = path_gen
        self.path_file = path_gen + 'EMOCON_VALENCE.CSV'

    def fase1_rnd(self):
        # Calcolo statistiche features importance
        total_data = read_csv(self.path_file)
        features_names = list(total_data.iloc[:, 1:len(total_data.columns)].columns)

        # dataframe che conterrà tutti i valori delle cinque iterazioni
        df = pd.DataFrame(columns=features_names)

        i = 0
        while i < 5:
            training_data, testing_data, training_labels, testing_labels = train_test_split(
                total_data.iloc[:, 1:len(total_data.columns)], total_data.iloc[:, 0])

            model = RandomForestClassifier()
            model.fit(training_data, ravel(training_labels))
            importances = model.feature_importances_

            # creo un DataFrame contenente le caratteristiche dell'iterazione
            model_importances = pd.DataFrame([importances], columns=features_names)

            # concateno le caratteristiche dell'iterazione al dataframe generale
            df = pd.concat([df, model_importances], axis=0, ignore_index=True)
            i = i + 1

        # Calcolo statistiche
        df_stat = df.mean()

        # ##### Ordinamento statistiche
        # traslo il dataframe e gli associo il nome delle features
        df_stat = df_stat.T
        df_stat = df_stat.iloc[[0]]
        df_stat.set_axis(features_names, axis=1, inplace=True)

        # ordino in modo decrescente le colonne in base al valore delle caratteristiche
        df_stat = df_stat.sort_values(by='0', axis=1, ascending=False)

        path_car = self.path + 'sorted_stat_importance_rnd.csv'
        df_stat.to_csv(path_car, index=False)

    def fase1_grd(self):
        # ##### Calcolo statistiche features importance
        total_data = read_csv(self.path_file)
        features_names = list(total_data.iloc[:, 1:len(total_data.columns)].columns)
        # dataframe che conterrà tutti i valori delle cinque iterazioni
        df = pd.DataFrame(columns=features_names)

        i = 0
        while i < 5:
            training_data, testing_data, training_labels, testing_labels = train_test_split(
                total_data.iloc[:, 1:len(total_data.columns)], total_data.iloc[:, 0])

            model = GradientBoostingClassifier()
            model.fit(training_data, ravel(training_labels))

            importances = model.feature_importances_

            # creo un DataFrame contenente le caratteristiche dell'iterazione
            model_importances = pd.DataFrame([importances], columns=features_names)

            # concateno le caratteristiche dell'iterazione al dataframe generale
            df = pd.concat([df, model_importances], axis=0, ignore_index=True)
            i = i + 1

        # Calcolo statistiche
        df_stat = df.mean()

        # ##### Ordinamento statistiche
        # traslo il dataframe e gli associo il nome delle features
        df_stat = df_stat.T
        df_stat = df_stat.iloc[[0]]
        df_stat.set_axis(features_names, axis=1, inplace=True)

        # ordino in modo decrescente le colonne in base al valore delle caratteristiche
        df_stat = df_stat.sort_values(by='0', axis=1, ascending=False)

        path_car = self.path + 'sorted_stat_importance_grd.csv'
        df_stat.to_csv(path_car, index=False)

    def fase2_rnd(self):
        path_car = self.path + 'sorted_stat_importance_rnd.csv'
        df_stat = read_csv(path_car)
        total_data = read_csv(self.path_file)
        features_data = total_data.iloc[:, 1:len(total_data.columns)]
        target_data = total_data.iloc[:, 0]
        n = 3

        start_path = self.path + 'interm_results_rnd_'
        end_path = '.csv'
        while n <= len(features_data.columns):
            # trovo le N-len features meno rilevanti
            n_proc_features = df_stat.iloc[:, n:len(features_data.columns)]

            # per ogni feature tolgo dal dataset totale
            n_dataset = features_data.copy()
            for feature in n_proc_features.columns:
                n_dataset = n_dataset.drop(feature, axis=1)

            # addestro per 5 volte il modello e trovo i valori di score e tempi
            i = 0
            times = np.float64([0, 0, 0, 0, 0])
            scores = np.float64([0, 0, 0, 0, 0])
            while i < 5:
                start = time.time()

                training_data, testing_data, training_labels, testing_labels = train_test_split(
                    n_dataset, target_data)
                model = RandomForestClassifier()
                model.fit(training_data, ravel(training_labels))

                end = time.time()

                predicted_labels = model.predict(testing_data)
                score = accuracy_score(ravel(testing_labels), predicted_labels)
                # mi salvo dati dello score e tempi
                times[i] = end - start
                scores[i] = score
                i = i + 1

            # calcolo statistiche
            avg_times = statistics.mean(times)
            avg_scores = statistics.mean(scores)

            # salvo valore delle statistiche
            data = [{'score': avg_scores, 'times': avg_times}]
            df_elem = pd.DataFrame(data)

            path_n = start_path + str(n - 3) + end_path
            df_elem.to_csv(path_n, index=False)

            n = n + 1

    def fase2_grd(self):
        path_car = self.path + 'sorted_stat_importance_grd.csv'
        df_stat = read_csv(path_car)
        total_data = read_csv(self.path_file)
        features_data = total_data.iloc[:, 1:len(total_data.columns)]
        target_data = total_data.iloc[:, 0]
        n = 3

        start_path = self.path + 'interm_results_grd_'
        end_path = '.csv'
        while n <= len(features_data.columns):
            # trovo le N-len features meno rilevanti
            n_proc_features = df_stat.iloc[:, n:len(features_data.columns)]

            # per ogni feature tolgo dal dataset totale
            n_dataset = features_data.copy()
            for feature in n_proc_features.columns:
                n_dataset = n_dataset.drop(feature, axis=1)

            # addestro per 5 volte il modello e trovo i valori di score e tempi
            i = 0
            times = np.float64([0, 0, 0, 0, 0])
            scores = np.float64([0, 0, 0, 0, 0])
            while i < 5:
                start = time.time()

                training_data, testing_data, training_labels, testing_labels = train_test_split(
                    n_dataset, target_data)
                model = GradientBoostingClassifier()
                model.fit(training_data, ravel(training_labels))

                end = time.time()

                predicted_labels = model.predict(testing_data)
                score = accuracy_score(ravel(testing_labels), predicted_labels)
                # mi salvo dati dello score e tempi
                times[i] = end - start
                scores[i] = score
                i = i + 1

            # calcolo statistiche
            avg_times = statistics.mean(times)
            avg_scores = statistics.mean(scores)

            # salvo valore delle statistiche
            data = [{'score': avg_scores, 'times': avg_times}]
            df_elem = pd.DataFrame(data)

            path_n = start_path + str(n - 3) + end_path
            df_elem.to_csv(path_n, index=False)

            n = n + 1

    def fase3_rnd(self):
        n = 0
        start_path = self.path + 'interm_results_rnd_'
        end_path = '.csv'

        labels = []
        scores = []
        times = []
        times_s = []
        while n < 16:
            n_path = start_path + str(n) + end_path
            n_data = pd.read_csv(n_path)
            n_data = n_data.iloc[0]
            n_label = str(n + 3)
            labels.append(n_label)
            scores.append(n_data['score'])
            times.append(n_data['times'])
            times_s.append(time.strftime("%M:%S", time.gmtime(n_data['times'])))
            n = n + 1

        ################################################################################
        df_date = pd.DataFrame({'score': scores, 'time': times_s}, index=labels)
        print(df_date)
        print(" ")

        fig, ax1 = plt.subplots()
        ax1.set_xlabel('first N important features')
        ax1.set_ylabel('score', color="tab:red", weight='bold')
        ax1.plot(scores, color="tab:red", marker='.', markerfacecolor='maroon')
        ax1.tick_params(axis='y', labelcolor="tab:red")

        # si mostrano le due curve sullo stesso grafico
        ax2 = ax1.twinx()
        ax2.set_ylabel('tempo di esecuzione', color="tab:blue", weight='bold')
        ax2.plot(times, color="tab:blue", marker='.', markerfacecolor='darkblue')
        ax2.tick_params(axis='y', labelcolor="tab:blue")

        y_minutes = [time.strftime("%M:%S", time.gmtime(x)) for x in np.arange(0, 700, 100)]
        y_t = [x for x in np.arange(0, 700, 100)]

        plt.yticks(y_t, y_minutes)
        plt.xticks(np.arange(0, 16, step=1), np.arange(3, 19, step=1))

        plt.title("Statistiche random forest", weight='bold', fontsize=16, pad=16)
        plt.grid()
        fig.tight_layout()
        plt.show()

    def fase3_grd(self):
        n = 0
        start_path = self.path + 'interm_results_grd_'
        end_path = '.csv'

        labels = []
        scores = []
        times = []
        times_s = []
        while n < 16:
            n_path = start_path + str(n) + end_path
            n_data = pd.read_csv(n_path)
            n_data = n_data.iloc[0]
            n_label = str(n + 3)
            labels.append(n_label)
            scores.append(n_data['score'])
            times.append(n_data['times'])
            times_s.append(time.strftime("%M:%S", time.gmtime(n_data['times'])))
            n = n + 1

        ################################################################################
        df_date = pd.DataFrame({'score': scores, 'time': times_s}, index=labels)
        print(df_date)
        print(" ")

        fig, ax1 = plt.subplots()
        ax1.set_xlabel('first N important features')
        ax1.set_ylabel('score', color="tab:red", weight='bold')
        ax1.plot(scores, color="tab:red", marker='.', markerfacecolor='maroon')
        ax1.tick_params(axis='y', labelcolor="tab:red")

        # si mostrano le due curve sullo stesso grafico
        ax2 = ax1.twinx()
        ax2.set_ylabel('tempo di esecuzione', color="tab:blue", weight='bold')
        ax2.plot(times, color="tab:blue", marker='.', markerfacecolor='darkblue')
        ax2.tick_params(axis='y', labelcolor="tab:blue")

        y_minutes = [time.strftime("%M:%S", time.gmtime(x)) for x in np.arange(0, 900, 100)]
        y_t = [x for x in np.arange(0, 900, 100)]

        plt.yticks(y_t, y_minutes)
        plt.xticks(np.arange(0, 16, step=1), np.arange(3, 19, step=1))

        plt.title("Statistiche gradient boosting", weight='bold', fontsize=16, pad=16)
        plt.grid()
        fig.tight_layout()
        plt.show()

    def exec(self):
        self.fase1_rnd()
        self.fase2_rnd()
        self.fase3_rnd()

        self.fase1_grd()
        self.fase2_grd()
        self.fase3_grd()


# ##############################################################################
#                                 REPORT 7
# ##############################################################################
# Ultimo report, si analizzano le prestazioni dei due modelli andando a togliere
# ogni volta le feature più importante fino a quando si arriva alle 3 features
# meno importanti. I risultati si mostrano graficamente e in modo tabellare come
# il report 6.
# i risultati vengono stampati in formato tabellare e grafico
# --> __init__() : imposta il path del report
# --> fase1_rnd(): calcolo e ordinamente features per il random forest
# --> fase1_grd(): calcolo e ordinamente features per il gradient boosting
# --> fase2_rnd(): calcolo tempi di esecuzione medi e score medio per insiemi
#                  decrescenti di features meno importanti usando random forest
# --> fase2_grd(): calcolo tempi di esecuzione medi e score medio per insiemi
#                  crescenti di features più importanti usando gradient boosting
# --> fase3_rnd(): stampa risultati del random forest graficamente e tabella
# --> fase3_grd(): stampa risultati del gradient boosting graficamente e tabella
# --> exec(): esegue le funzioni riportate sopra in modo ordinato
class Report7:
    def __init__(self, path_gen):
        self.path = path_gen
        self.path_file = path_gen + 'EMOCON_VALENCE.CSV'

    def fase1_rnd(self):
        # Calcolo statistiche features importance
        total_data = read_csv(self.path_file)
        features_names = list(total_data.iloc[:, 1:len(total_data.columns)].columns)

        # dataframe che conterrà tutti i valori delle cinque iterazioni
        df = pd.DataFrame(columns=features_names)

        i = 0
        while i < 5:
            training_data, testing_data, training_labels, testing_labels = train_test_split(
                total_data.iloc[:, 1:len(total_data.columns)], total_data.iloc[:, 0])

            model = RandomForestClassifier()
            model.fit(training_data, ravel(training_labels))
            importances = model.feature_importances_

            # creo un DataFrame contenente le caratteristiche dell'iterazione
            model_importances = pd.DataFrame([importances], columns=features_names)

            # concateno le caratteristiche dell'iterazione al dataframe generale
            df = pd.concat([df, model_importances], axis=0, ignore_index=True)
            i = i + 1

        # Calcolo statistiche
        df_stat = df.mean()

        # ##### Ordinamento statistiche
        # traslo il dataframe e gli associo il nome delle features
        df_stat = df_stat.T
        df_stat = df_stat.iloc[[0]]
        df_stat.set_axis(features_names, axis=1, inplace=True)

        # ordino in modo decrescente le colonne in base al valore delle caratteristiche
        df_stat = df_stat.sort_values(by='0', axis=1, ascending=False)

        path_car = self.path + 'sorted_stat_importance_rnd.csv'
        df_stat.to_csv(path_car, index=False)

    def fase1_grd(self):
        # ##### Calcolo statistiche features importance
        total_data = read_csv(self.path_file)
        features_names = list(total_data.iloc[:, 1:len(total_data.columns)].columns)
        # dataframe che conterrà tutti i valori delle cinque iterazioni
        df = pd.DataFrame(columns=features_names)

        i = 0
        while i < 5:
            training_data, testing_data, training_labels, testing_labels = train_test_split(
                total_data.iloc[:, 1:len(total_data.columns)], total_data.iloc[:, 0])

            model = GradientBoostingClassifier()
            model.fit(training_data, ravel(training_labels))

            importances = model.feature_importances_

            # creo un DataFrame contenente le caratteristiche dell'iterazione
            model_importances = pd.DataFrame([importances], columns=features_names)

            # concateno le caratteristiche dell'iterazione al dataframe generale
            df = pd.concat([df, model_importances], axis=0, ignore_index=True)
            i = i + 1

        # Calcolo statistiche
        df_stat = df.mean()

        # ##### Ordinamento statistiche
        # traslo il dataframe e gli associo il nome delle features
        df_stat = df_stat.T
        df_stat = df_stat.iloc[[0]]
        df_stat.set_axis(features_names, axis=1, inplace=True)

        # ordino in modo decrescente le colonne in base al valore delle caratteristiche
        df_stat = df_stat.sort_values(by='0', axis=1, ascending=False)

        path_car = self.path + 'sorted_stat_importance_grd.csv'
        df_stat.to_csv(path_car, index=False)

    def fase2_rnd(self):
        path_car = self.path + 'sorted_stat_importance_rnd.csv'
        df_stat = read_csv(path_car)
        total_data = read_csv(self.path_file)
        features_data = total_data.iloc[:, 1:len(total_data.columns)]
        target_data = total_data.iloc[:, 0]
        n = 1

        start_path = self.path + '/interm_results_rnd_c_'
        end_path = '.csv'

        while n <= len(features_data.columns) - 3:
            # trovo le N features più rilevanti
            features_to_drop = df_stat.iloc[:, 0:n]

            # tolgo dal dataset totale le N features più importanti
            n_dataset = features_data.copy()
            for feature in features_to_drop.columns:
                n_dataset = n_dataset.drop(feature, axis=1)

            # addestro per 5 volte il modello e ottengo la media degli score e tempi
            i = 0
            times = np.float64([0, 0, 0, 0, 0])
            scores = np.float64([0, 0, 0, 0, 0])
            while i < 5:
                start = time.time()

                training_data, testing_data, training_labels, testing_labels = train_test_split(
                    n_dataset, target_data)
                model = RandomForestClassifier()
                model.fit(training_data, ravel(training_labels))

                end = time.time()

                predicted_labels = model.predict(testing_data)
                score = accuracy_score(ravel(testing_labels), predicted_labels)

                # memorizzo dati dello score e tempi
                times[i] = end - start
                scores[i] = score
                i = i + 1

            # calcolo statistiche
            avg_times = statistics.mean(times)
            avg_scores = statistics.mean(scores)

            # salvo valore delle statistiche
            data = [{'score': avg_scores, 'times': avg_times}]
            df_elem = pd.DataFrame(data)

            # stringa risultato intermedio
            path_n = start_path + str(n) + end_path
            df_elem.to_csv(path_n, index=False)

            n = n + 1

    def fase2_grd(self):
        path_car = self.path + 'sorted_stat_importance_grd.csv'
        df_stat = read_csv(path_car)
        total_data = read_csv(self.path_file)
        features_data = total_data.iloc[:, 1:len(total_data.columns)]
        target_data = total_data.iloc[:, 0]
        n = 1

        start_path = self.path + '/interm_results_grd_c_'
        end_path = '.csv'

        while n <= len(features_data.columns) - 3:
            # trovo le N features più rilevanti
            features_to_drop = df_stat.iloc[:, 0:n]

            # tolgo dal dataset totale le N features più importanti
            n_dataset = features_data.copy()
            for feature in features_to_drop.columns:
                n_dataset = n_dataset.drop(feature, axis=1)

            # addestro per 5 volte il modello e ottengo la media degli score e tempi
            i = 0
            times = np.float64([0, 0, 0, 0, 0])
            scores = np.float64([0, 0, 0, 0, 0])
            while i < 5:
                start = time.time()

                training_data, testing_data, training_labels, testing_labels = train_test_split(
                    n_dataset, target_data)
                model = GradientBoostingClassifier()
                model.fit(training_data, ravel(training_labels))

                end = time.time()

                predicted_labels = model.predict(testing_data)
                score = accuracy_score(ravel(testing_labels), predicted_labels)

                # memorizzo dati dello score e tempi
                times[i] = end - start
                scores[i] = score
                i = i + 1

            # calcolo statistiche
            avg_times = statistics.mean(times)
            avg_scores = statistics.mean(scores)

            # salvo valore delle statistiche
            data = [{'score': avg_scores, 'times': avg_times}]
            df_elem = pd.DataFrame(data)

            # stringa risultato intermedio
            path_n = start_path + str(n) + end_path
            df_elem.to_csv(path_n, index=False)

            n = n + 1

    def fase3_rnd(self):
        total_data = read_csv(self.path_file)
        features_data = total_data.iloc[:, 1:len(total_data.columns)]

        start_path = self.path + 'interm_results_rnd_c_'
        end_path = '.csv'

        n = 1
        i = 0
        labels = []
        scores = []
        times = []
        times_s = []
        while n < len(features_data.columns) - 3:
            n_path = start_path + str(n) + end_path
            n_data = pd.read_csv(n_path)
            n_data = n_data.iloc[0]
            n_label = str(17 - i)
            labels.append(n_label)
            scores.append(n_data['score'])
            times.append(n_data['times'])
            times_s.append(time.strftime("%M:%S", time.gmtime(n_data['times'])))
            n = n + 1
            i = i + 1
        ################################################################################
        df_t = pd.DataFrame({'score': scores, 'time': times_s}, index=labels)
        print(df_t)
        print(" ")

        fig, ax1 = plt.subplots()
        ax1.set_xlabel('less N important features (tatal 18 features)')
        ax1.set_ylabel('score', color="tab:red", weight='bold')
        ax1.plot(scores, color="tab:red", marker='.')
        ax1.tick_params(axis='y', labelcolor="tab:red")

        ax2 = ax1.twinx()
        ax2.set_ylabel('tempo di esecuzione', color="tab:blue", weight='bold')
        ax2.plot(times, color="tab:blue", marker='.')
        ax2.tick_params(axis='y', labelcolor="tab:blue")

        y_minutes = [time.strftime("%M:%S", time.gmtime(x)) for x in np.arange(0, 1000, 100)]
        y_t = [x for x in np.arange(0, 1000, 100)]
        plt.yticks(y_t, y_minutes)

        x_label = np.arange(3, 18, step=1)
        x_label = sorted(x_label, reverse=True)
        plt.xticks(np.arange(0, 15, step=1), x_label)

        plt.title("Statistiche random forest", weight='bold', fontsize=16, pad=16)
        plt.grid()
        fig.tight_layout()
        plt.show()

    def fase3_grd(self):
        total_data = read_csv(self.path_file)
        features_data = total_data.iloc[:, 1:len(total_data.columns)]

        start_path = self.path + 'interm_results_grd_c_'
        end_path = '.csv'

        n = 1
        i = 0
        labels = []
        scores = []
        times = []
        times_s = []
        while n < len(features_data.columns) - 3:
            n_path = start_path + str(n) + end_path
            n_data = pd.read_csv(n_path)
            n_data = n_data.iloc[0]
            n_label = str(17 - i)
            labels.append(n_label)
            scores.append(n_data['score'])
            times.append(n_data['times'])
            times_s.append(time.strftime("%M:%S", time.gmtime(n_data['times'])))
            n = n + 1
            i = i + 1
        ################################################################################
        df_t = pd.DataFrame({'score': scores, 'time': times_s}, index=labels)
        print(df_t)
        print(" ")

        fig, ax1 = plt.subplots()
        ax1.set_xlabel('less N important features (tatal 18 features)')
        ax1.set_ylabel('score', color="tab:red", weight='bold')
        ax1.plot(scores, color="tab:red", marker='.')
        ax1.tick_params(axis='y', labelcolor="tab:red")

        ax2 = ax1.twinx()
        ax2.set_ylabel('tempo di esecuzione', color="tab:blue", weight='bold')
        ax2.plot(times, color="tab:blue", marker='.')
        ax2.tick_params(axis='y', labelcolor="tab:blue")

        y_minutes = [time.strftime("%M:%S", time.gmtime(x)) for x in np.arange(0, 1000, 100)]
        y_t = [x for x in np.arange(0, 1000, 100)]
        plt.yticks(y_t, y_minutes)

        x_label = np.arange(3, 18, step=1)
        x_label = sorted(x_label, reverse=True)
        plt.xticks(np.arange(0, 15, step=1), x_label)

        plt.title("Statistiche gradient boosting", weight='bold', fontsize=16, pad=16)
        plt.grid()
        fig.tight_layout()
        plt.show()

    def exec(self):
        self.fase1_rnd()
        self.fase2_rnd()
        self.fase3_rnd()

        self.fase1_grd()
        self.fase2_grd()
        self.fase3_grd()


if __name__ == "__main__":
    print("Inserisci path dei dataset:")
    path_dataset = input()
    while 1:
        print("""Benvenuto!\n
      Immetti uno dei seguenti numeri per eseguire uno dei report o uscire:\n
      \t1)\tVisualizza Report 1, sul soggetto 25 utilizzando MLP\n
      \t2)\tVisualizza Report 2, sulla variabile Arousal utilizzando MLP\n
      \t3)\tVisualizza Report 3, sulla variabile Arousal utilizzando MLP, Random Forest e Gradient Boosting\n
      \t4)\tVisualizza Report 4, sul calcolo delle features importance\n
      \t5)\tVisualizza Report 5, sul calcolo delle statistiche usando H4 oppure H7\n
      \t6)\tVisualizza Report 6, sull'analisidelle feature più importanti\n
      \t7)\tVisualizza Report 7, sull'analisi delle feature meno importanti\n
      \t8)\tEsci\n""")
        risposta = input()
        if risposta == '1':
            r = Report1(path_dataset)
            print("Eseguo il Report 1, attendere i risultati...")
            r.exec()
        elif risposta == '2':
            r = Report2(path_dataset)
            print("Eseguo il Report 2, attendere i risultati...")
            r.exec()
        elif risposta == '3':
            r = Report3(path_dataset)
            print("Eseguo il Report 3, attendere i risultati...")
            r.exec()
        elif risposta == '4':
            r = Report4(path_dataset)
            print("Eseguo il Report 4, attendere i risultati...")
            r.exec()
        elif risposta == '5':
            r = Report5(path_dataset)
            print("Eseguo il Report 5, attendere i risultati...")
            r.exec()
        elif risposta == '6':
            r = Report6(path_dataset)
            print("Eseguo il Report 6, attendere i risultati...")
            r.exec()
        elif risposta == '7':
            r = Report7(path_dataset)
            print("Eseguo il Report 7, attendere i risultati...")
            r.exec()
        elif risposta == '8':
            break
        else:
            print("Comando non valido")
