from numpy import ravel
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from glob import glob
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import statistics


# ##############################################################################
#                                 Dati
# ##############################################################################
# Imposta il path generale da dove ricavare il dataset totale EMOCON_VALENCE.CSV
# e dove riportare i file risultati
# -> file_path()    : ritorna il path di dove si trova il file
# -> get_path()     : ritorna il path del file comprensivo del filename
class Dati:
    def __init__(self, path_gen, name_file):
        if name_file == "":
            self.filename = "EMOCON_VALENCE.CSV"
        else:
            self.filename = name_file
        self.path = path_gen

    def file_path(self):
        return self.path + self.filename

    def get_path(self):
        return self.path


# ##############################################################################
#                                 MetodoML
# ##############################################################################
# Classe utilizzata per addestrare i modelli di ML (Gradient Boosting e Random
# Forest), viene inizializzata fornendogli il dataset già diviso attraverso la
# tecnica 'train test split', viene fornito inoltre il modello che si vuole
# implementare.
# -> set_tipo()         : imposta il tipo del modello di ML (Random Forest o
#                       Gradient Boosting.
# -> train_model()      : addestra il modello con i parametri inseriti
# -> get_type()         : ritorna il modello di ML usato
# -> get_score()        : ritorna lo score del modello addestrato
# -> get_feature_importance()   : ritorna le feature importance del modello
#                               addestrato.
# -> get_feature_names(): ritorna il nome delle feature del dataset
# -> prediction()       : presi dei dati di ingresso (classe Dati()) mostra le
#                       le previsioni del modello addestrato

class MetodoML:
    def __init__(self, training_d, testing_d, training_l, testing_l, tipo_modello):
        self.training_data = training_d
        self.training_labels = training_l
        self.testing_data = testing_d
        self.testing_labels = testing_l
        self.tipo = tipo_modello
        self.model_g = GradientBoostingClassifier()
        self.model_r = RandomForestClassifier()

    def set_tipo(self, tipo_modello):
        self.tipo = tipo_modello

    def train_model(self):
        if self.tipo == "GradientBoosting":
            self.model_g.fit(self.training_data, ravel(self.training_labels))
        elif self.tipo == "RandomForest":
            self.model_r.fit(self.training_data, ravel(self.training_labels))
        else:
            print("Modello ML inserito non valido.")

    def get_type(self):
        return self.tipo

    def get_score(self):
        if self.tipo == "GradientBoosting":
            model_ml = self.model_g
        elif self.tipo == "RandomForest":
            model_ml = self.model_r
        else:
            print("Modello ML non valido.")
            return
        predicted_l = model_ml.predict(self.testing_data)
        a_s = accuracy_score(ravel(self.testing_labels), predicted_l)
        return a_s

    def get_feature_importance(self):
        if self.tipo == "GradientBoosting":
            model_ml = self.model_g
        elif self.tipo == "RandomForest":
            model_ml = self.model_r
        else:
            print("Modello ML non valido.")
            return

        importances = model_ml.feature_importances_
        return importances

    def get_feature_names(self):
        features_names = list(self.training_data.columns)
        return features_names

    def prediction(self, data):
        path_file = data.file_path()
        feat_data = read_csv(path_file)
        if self.tipo == "RandomForest":
            pred_labels = self.model_r.predict(feat_data)
        elif self.tipo == "GradientBoosting":
            pred_labels = self.model_g.predict(feat_data)
        else:
            print("Modello ML non previsto.")
            return
        return pred_labels


# ##############################################################################
#                                 FeatImportance
# ##############################################################################
# Elabora le feature importance calcolandone la media per 5 iterazioni e
# mostrandole graficamente, tutto questo per entrambi i modelli di ML.
# -> add_fi ()  : recupera e calcola la media per cinque iterazioni delle
#               feature importance.
# -> plot_fi()  : mostra graficamente i valori precedentemente calcolati.
# -> get_fi()   : ritorna i valori delle feature importance precedentemente
#               calcolati.
class FeatImportance:
    def __init__(self, file):
        self.dati = file
        self.fi_grd = None
        self.fi_rnd = None
        self.fi_grd_np = None
        self.fi_rnd_np = None

    def add_fi(self, model_type):
        path_file = self.dati.file_path()
        total_data = read_csv(path_file)
        features_names = list(total_data.iloc[:, 1:len(total_data.columns)])
        # dataframe che conterrà tutti i valori delle cinque iterazioni
        df = pd.DataFrame(columns=features_names)

        path_file = self.dati.file_path()
        total_data = read_csv(path_file)

        # ottengo i nomi delle features
        features_names = list(total_data.iloc[:, 1:len(total_data.columns)].columns)

        i = 0
        while i < 5:
            train_d, test_d, train_l, test_l = train_test_split(
                total_data.iloc[:, 1:len(total_data.columns)],
                total_data.iloc[:, 0])
            model_app = MetodoML(train_d, test_d, train_l, test_l, model_type)
            model_app.train_model()
            feature_importances = model_app.get_feature_importance()

            # creo un DataFrame contenente le caratteristiche dell'iterazione
            model_importances = pd.DataFrame([feature_importances],
                                             columns=features_names)

            # concateno le caratteristiche dell'iterazione al dataframe generale
            df = pd.concat([df, model_importances], axis=0, ignore_index=True)
            i = i + 1

        # Calcolo media
        df_stat = df.mean()

        df_stat_np = df_stat.to_numpy()
        if model_type == "RandomForest":
            self.fi_rnd_np = df_stat_np
        elif model_type == "GradientBoosting":
            self.fi_grd_np = df_stat_np
        else:
            print("Modello ML non previsto.")
            return

        path_app = self.dati.get_path()
        path_app = path_app + model_type + '_sorted_fi.csv'

        df_stat.to_csv(path_app, index=False)
        df_stat = read_csv(path_app)

        # ##### Ordinamento statistiche
        # traslo il dataframe e gli associo il nome delle features
        df_stat = df_stat.T
        df_stat = df_stat.iloc[[0]]
        df_stat = df_stat.set_axis(features_names, axis=1)

        # ordino in modo decrescente le colonne in base al valore delle caratteristiche
        df_stat = df_stat.sort_values(by='0', axis=1, ascending=False)

        # memorizzo le statistiche sulle feature importance
        if model_type == "RandomForest":
            self.fi_rnd = df_stat
        elif model_type == "GradientBoosting":
            self.fi_grd = df_stat
        else:
            print("Modello ML non previsto.")
            return

        path_dati = self.dati.get_path()
        path_car = path_dati + 'sorted_stat_importance_rnd.csv'
        df_stat.to_csv(path_car, index=False)

    def plot_fi(self, tp):

        fig, ax = plt.subplots()

        path_app = self.dati.file_path()
        dataset = read_csv(path_app)
        dataset = dataset.iloc[:, 1:len(dataset.columns)]

        feat_names = list(dataset.columns)

        if tp == "RandomForest":
            series_importance = pd.Series(self.fi_rnd_np, index=feat_names)
            ax.set_title("Feature importances random forest")
        elif tp == "GradientBoosting":
            series_importance = pd.Series(self.fi_grd_np, index=feat_names)
            ax.set_title("Feature importances gradient boosting")
        else:
            print("Modello ML non previsto.")
            return
        series_importance.plot.bar(ax=ax)
        fig.tight_layout()
        plt.show()

    def get_fi(self, t_ml):
        if t_ml == "RandomForest":
            return self.fi_rnd
        elif t_ml == "GradientBoosting":
            return self.fi_grd


# ##############################################################################
#                                 ScoreTime
# ##############################################################################
# Si occupa di analizzare un modello di ML (Random Forest o Gradient Boosting),
# andando a prendere in considerazione il tempo di addestramento necessario e
# l'accuratezza delle previsioni per sottoinsiemi del dataset generati in base
# alle feature importance.
# L'analisi si distingue in :
# 1) "Prova"       : i sottoinsiemi generati partono dalle 3 caratteristiche più
#                   importanti andando ad includere ad ogni iterazione la
#                   caratteristica più importante non presente nel sottoinsieme
# 2) "Controprova" : i sottoinsieme generati partono da tutto il dataset andando
#                   a togliere, ad ogni iterazione la caratteristica più
#                   importante fino ad avere un sottoinsieme di 3 caratteristiche
#                   meno importanti
# -> exec_model()       : esegue la fase di rilevazione del tempo di addestramento
#                       e dello score per insiemi crescenti o decrescenti, in
#                       base all'inizializzazione dell'instanza (prova o
#                       controprova), tutto questo per un certo modello
# -> plot_scoretime()   : mostra graficamente l'andamento dei tempi e dello score
#                       al variare dei sottoinsiemi per un certo modello
# -> exec()         : esegue exec_model() chiamandola per entrambi i modelli
#                   (Random Forest e Gradient Boosting)
# -> plot()         : esegue plot_scoretime() chiamandola per entrambi i modelli
class ScoreTime:
    def __init__(self, file, contro):
        self.dati = file
        self.controprova = contro

    def exec_model(self, model_ml):
        # ottengo la media delle feature importance ordinate (!)
        model_feature_importance = FeatImportance(self.dati)
        model_feature_importance.add_fi(model_ml)
        sorted_feature_importance = model_feature_importance.get_fi(model_ml)

        # recupero il dataset
        total_data = read_csv(self.dati.file_path)

        feat_data = total_data.iloc[:, 1:len(total_data.columns)]
        trget_data = total_data.iloc[:, 0]

        # imposto indice n per la valutazione delle feature (più avanti)
        if not controprova:
            n_index = 3
        else:
            n_index = 1

        # imposto il path dove verranno scritti i risultati dell'esecuzione
        start_path = self.dati.get_path()
        if model_ml == "RandomForest":
            start_path = start_path + 'interm_results_rnd_'
        elif model_ml == "GradientBoosting":
            start_path = start_path + 'interm_results_grd_'

        if controprova:
            start_path = start_path + 'c_'

        end_path = '.csv'

        # setto la condizione del ciclo while
        if not controprova:
            while_condition = len(feat_data.columns)
        else:
            while_condition = len(feat_data.columns) - 3

        while n_index <= while_condition:

            if not controprova:
                # trovo le N-len features meno rilevanti (PROVA)
                n_proc_features = sorted_feature_importance.iloc[:, n_index:len(feat_data.columns)]
            else:
                # trovo le N features più rilevanti (CONTROPROVA)
                n_proc_features = sorted_feature_importance.iloc[:, 0:n_index]

            # per ogni feature tolgo dal dataset totale
            n_dataset = features_data.copy()
            for feat in n_proc_features.columns:
                n_dataset = n_dataset.drop(feat, axis=1)

            # addestro per 5 volte il modello e ottengo la media degli score e tempi
            i = 0
            times = np.float64([0, 0, 0, 0, 0])
            scores = np.float64([0, 0, 0, 0, 0])
            while i < 5:
                start = time.time()

                train_data, test_data, train_labels, test_labels = train_test_split(
                    n_dataset, trget_data)

                model_app = MetodoML(train_data, test_data, train_labels, test_labels, model_ml)

                # addestro il modello
                model_app.train_model()

                # trovo lo score del modello
                score_app = model_app.get_score()

                end = time.time()
                # mi salvo dati dello score e tempi
                times[i] = end - start
                scores[i] = score_app
                i = i + 1

            # calcolo statistiche
            avg_times = statistics.mean(times)
            avg_scores = statistics.mean(scores)

            # salvo valore delle statistiche
            data = [{'score': avg_scores, 'times': avg_times}]
            df_elem = pd.DataFrame(data)

            if not controprova:
                file_index = n_index - 3
            else:
                file_index = n_index

            path_n = start_path + str(file_index) + end_path
            df_elem.to_csv(path_n, index=False)

            n_index = n_index + 1

    def plot_scoretime(self, model_ml):
        path_file = self.dati.file_path()
        total_data = read_csv(path_file)
        feat_data = total_data.iloc[:, 1:len(total_data.columns)]
        i = 0
        if not controprova:
            n_index = 0
        else:
            n_index = 1

        # imposto il path dei file dei risultati
        start_path = self.dati.get_path()
        if model_ml == "RandomForest":
            start_path = start_path + 'interm_results_rnd_'
        elif model_ml == "GradientBoosting":
            start_path = start_path + 'interm_results_grd_'

        if controprova:
            start_path = start_path + 'c_'

        end_path = '.csv'

        labels = []
        scores = []
        times = []
        times_s = []

        # setto la condizione del ciclo while
        if not controprova:
            while_condition = 16
        else:
            while_condition = len(feat_data.columns) - 3

        while n_index < while_condition:
            # imposto il path n-esimo e leggo i risultati
            n_path = start_path + str(n_index) + end_path
            n_data = pd.read_csv(n_path)

            n_data = n_data.iloc[0]

            # imposto n. di feature utilizzate
            if not controprova:
                n_label = str(n_index + 3)
            else:
                n_label = str(17 - i)

            # memorizzo i risultati
            labels.append(n_label)
            scores.append(n_data['score'])
            times.append(n_data['times'])
            times_s.append(time.strftime("%M:%S", time.gmtime(n_data['times'])))
            n_index = n_index + 1
            if controprova:
                i = i + 1

        ################################################################################
        df_date = pd.DataFrame({'score': scores, 'time': times_s}, index=labels)
        print(df_date)
        print(" ")

        fig, ax1 = plt.subplots()

        # imposto il titolo della x label
        if not controprova:
            x_label_title = 'first N important features'
        else:
            x_label_title = 'less N important features (tatal 18 features)'
        ax1.set_xlabel(x_label_title)

        ax1.set_ylabel('score', color="tab:red", weight='bold')
        ax1.plot(scores, color="tab:red", marker='.', markerfacecolor='maroon')
        ax1.tick_params(axis='y', labelcolor="tab:red")

        # si mostrano le due curve sullo stesso grafico
        ax2 = ax1.twinx()
        ax2.set_ylabel('tempo di esecuzione', color="tab:blue", weight='bold')
        ax2.plot(times, color="tab:blue", marker='.', markerfacecolor='darkblue')
        ax2.tick_params(axis='y', labelcolor="tab:blue")

        # imposto il range di etichette dell'asse del tempo
        if controprova:
            right_edge = 1000
        elif model_ml == 'RandomForest':
            right_edge = 700
        elif model_ml == 'GradientBoosting':
            right_edge = 900
        else:
            print("Modello ml non previsto.")
            return

        y_minutes = [time.strftime("%M:%S", time.gmtime(x)) for x in np.arange(0, right_edge, 100)]
        y_t = [x for x in np.arange(0, right_edge, 100)]

        plt.yticks(y_t, y_minutes)

        # imposto i valori dell'asse x
        if not controprova:
            plt.xticks(np.arange(0, 16, step=1), np.arange(3, 19, step=1))
        else:
            x_label = np.arange(3, 18, step=1)
            x_label = sorted(x_label, reverse=True)
            plt.xticks(np.arange(0, 15, step=1), x_label)

        # imposto titolo del grafico
        if model_ml == 'RandomForest':
            text_title = "Statistiche random forest"
        elif model_ml == 'GradientBoosting':
            text_title = "Statistiche gradient boosting"
        else:
            print("Modello ml non previsto.")
            return

        if controprova:
            text_title = text_title + " controprova"
        else:
            text_title = text_title + " prova"

        plt.title(text_title, weight='bold', fontsize=16, pad=16)
        plt.grid()
        fig.tight_layout()
        plt.show()

    def exec(self):
        self.exec_model("RandomForest")
        self.exec_model("GradientBoosting")

    def plot(self):
        self.plot_scoretime("RandomForest")
        self.plot_scoretime("GradientBoosting")


if __name__ == "__main__":
    print("Inserisci path del dataset:")
    path_dataset = input()
    filename = "EMOCON_VALENCE.CSV"
    dati_emocon = Dati(path_dataset, filename)
    print("path memorizzato:\n\t")
    print(dati_emocon.file_path())
    print("")
    while 1:
        print("Chi sei?\n\t1)ML Engineer\n\t2)Decision Maker\n")
        chi_sono = input()
        if chi_sono == '1' or chi_sono == 'ML Engineer':
            ml = True
            break
        elif chi_sono == '2' or chi_sono == 'Decision Maker':
            ml = False
            break
        print("Formato sbagliato o ruolo errato!")

    if ml:
        # MENU PER IL MACHINE LEARNING ENGINEER
        check_fi = False
        check_st_p = False
        check_st_c = False
        while 1:
            print("""Ciao ML engineer! Dimmi cosa vuoi fare:\n
                \t1) Visualizza le feature importance
                \t2) Visualizza l'analisi del tempo di esecuzione e score
                \t3) Proponi modello ottimo
                \t4) Esci
                """)
            risposta = input()
            if risposta == '1':
                fi_class = FeatImportance(dati_emocon)
                fi_class.add_fi("RandomForest")
                fi_class.add_fi("GradientBoosting")
                fi_class.plot_fi("RandomForest")
                fi_class.plot_fi("GradientBoosting")
                check_fi = True
            elif risposta == '2':
                print("""Quale analisi si vuole visualizzare?\n\t
                    1) Prova\n\t
                    2) Controprova\n""")
                while 1:
                    risposta = input()
                    if risposta == '1':
                        controprova = False
                        check_st_p = True
                    if risposta == '2':
                        controprova = True
                        check_st_c = True
                    else:
                        print("Formato errato, inserisci 1 o 2.")
                        continue
                    st_class = ScoreTime(dati_emocon, controprova)
                    st_class.exec()
                    st_class.plot()
            elif risposta == '3':
                if check_fi and check_st_p and check_st_c:
                    print("""Quale modello si propone?\n
                        \t1) RandomForest\n
                        \t2) Gradient Boosting""")
                    while 1:
                        model = input()
                        if model == '1' or model == 'RandomForest':
                            model = 'RandomForest'
                            break
                        elif model == '2' or model == 'GradientBoosting':
                            model = 'Gradient boosting'
                            break
                        else:
                            print("Formato o modello non previsto.")

                    print("Quante feature più importanti si propongono?")
                    while 1:
                        num = input()
                        num = int(num)
                        if 3 <= num <= 18:
                            break
                        else:
                            print("Inserire un numero nel range [3,18]")
                    df_modello_proposto = pd.DataFrame(data={'modello': model, 'n': num})
                    path = dati_emocon.get_path() + 'prop_model.csv'
                    df_modello_proposto.to_csv(path, index=False)
                    print("Modello proposto:")
                    print(df_modello_proposto)
                else:
                    if not check_fi:
                        print("Manca ancora la visualizzazione delle feature importance.")
                    if not check_st_p:
                        print("Manca ancora la visualizzazione dell'analisi score-time.")
                    if not check_st_c:
                        print("Manca ancora la visualizzazione della controprova score-time.")
            elif risposta == '4':
                break
            else:
                print("Comando inserito non valito!\n")

    if not ml:
        # MENU PER IL DECISION MAKER
        new_dati = None
        validated_model = None
        while 1:
            print("""Ciao Decision Maker! Dimmi cosa vuoi fare:\n
                \t1) Valida modello proposto dal ML engineer\n
                \t2) Carica nuovi dati col modello validato\n
                \t3) Leggi previsioni sui nuovi dati\n
                \t4) Esci""")
            risposta = input()
            if risposta == '1':
                path = dati_emocon.get_path()
                peth_emocon = dati_emocon.file_path()

                tot_dataset = read_csv(path)

                # Verifico la presenza di risultati da mostrare
                file_check_path_1 = path + 'interm_results_rnd_*'
                file_check_path_2 = path + 'interm_results_rnd_c_*'
                if glob(file_check_path_1):
                    if glob(file_check_path_2):
                        # display prova - controprova
                        prova = ScoreTime(dati_emocon, False)
                        controprova = ScoreTime(dati_emocon, True)
                        prova.plot()
                        controprova.plot()
                    else:
                        print("Dati della controprova non disponibili.")
                        continue
                else:
                    print("Dati dell'analisi score-time non disponibili.")

                print("Recupero le feature importance...")
                fi_class = FeatImportance(dati_emocon)
                fi_class.add_fi("RandomForest")
                fi_class.add_fi("GradientBoosting")
                fi_class.plot_fi("RandomForest")
                fi_class.plot_fi("GradientBoosting")
                fi_list_rnd = fi_class.get_fi("RandomForest")
                fi_list_grd = fi_class.get_fi("GradientBoosting")

                file_check_path = path + 'prop_model.csv'
                # verifico se esiste un modello da validare
                if glob(file_check_path):
                    prop_model = read_csv(file_check_path)
                else:
                    print("Ancora nessun modello è stato proposto dal ML engineer.\n")
                    continue
                model = prop_model["modello"]
                model = model.iloc[[0]]
                n = prop_model["n"]
                n = int(n.iloc[[0]])

                print(f"""Modello proposto dal ML engineer:\n
                    \t-> Modello ML : {model}\n
                    \t-> Numero di caratteristiche più importanti : {n}\n
                    \n Validare? (y\n)""")
                while 1:
                    risposta = input()
                    if risposta == 'y':
                        # Genera modello e addestra modello
                        if model == "RandomForest":
                            fi = fi_list_rnd
                        elif model == "GradientBoosting":
                            fi = fi_list_grd
                        else:
                            print("Modello ML non previsto.")
                            continue
                        print("Modello validato.")
                        # filtra dataset
                        features_data = tot_dataset.iloc[:, 1:len(tot_dataset.columns)]
                        target_data = tot_dataset.iloc[:, 0]

                        feature_to_drop = fi.iloc[:, n:len(features_data.columns)]
                        good_feat = fi.iloc[:, 0:n - 1]
                        filtered_data = features_data.copy()
                        for feature in feature_to_drop.columns:
                            filtered_data = filtered_data.drop(feature, axis=1)

                        print("(!!) Caratteristiche del modello: (!!)")
                        for g_f in good_feat.columns:
                            print(g_f)

                        # train test split
                        training_data, testing_data, training_labels, \
                            testing_labels = train_test_split(filtered_data,
                                                              target_data)

                        # inizializza modello

                        if model == "RandomForest":
                            validated_model = RandomForestClassifier()
                        elif model == "GradientBoosting":
                            validated_model = GradientBoostingClassifier()
                        else:
                            print("Modello ML non previsto.")

                        print(f"Addestramento modello ({model}) in corso...")

                        # train modello
                        validated_model.fit(training_data, ravel(training_labels))

                        # test modello
                        predicted_labels = validated_model.predict(testing_data)
                        score = accuracy_score(ravel(testing_labels),
                                               predicted_labels)

                        print(f"Modello validato, score: {score}")
                    if risposta == 'n':
                        print("""Modello NON validato, in attesa di un altro
                                modello dal ML engineer.""")
                    # elimina file csv
                    file_to_delete = Path(file_check_path)
                    file_to_delete.unlink()
            elif risposta == '2':
                print("Inserisci il filename: (.csv)")
                risposta = input()
                new_flnm = risposta
                print("Nome del file inserito:\n\t\t")
                print(new_flnm)

                print("Inserisci il path del file name:")
                risposta = input()
                path_flnm = risposta
                print("Path inserito:\n\t\t")
                print(path_flnm)

                new_dati = Dati(path_flnm, new_flnm)
            elif risposta == '3':
                try:
                    results = validated_model.predict(new_dati)
                    print("Risultati della previsione:")
                    print(results)
                except NameError:
                    print("Modello non ancora validato.")
                else:
                    print("Modello non ancora validato.")
            elif risposta == '4':
                break
            else:
                print("Comando inserito non valito!\n")
