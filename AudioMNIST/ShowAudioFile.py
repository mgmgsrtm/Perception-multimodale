import argparse
import torch
import torchaudio
import matplotlib.pyplot as plt

ToSpectrogram = torchaudio.transforms.MelSpectrogram(
    sample_rate=16000,  # Taux d'échantillonnage (fréquence d'échantillonnage du signal audio)
    n_fft=400,          # Taille de la FFT
    hop_length=160,     # Pas de la fenêtre glissante
    n_mels=256,         # Nombre de filtres Mel
    f_min=0,            # Fréquence minimale
    f_max=8000          # Fréquence maximale
)

ToDB = torchaudio.transforms.AmplitudeToDB()

def main():
    # Paramètres d'affiche
    parser = argparse.ArgumentParser(description='Affichage de fichiers sonores')
    parser.add_argument('filename')           # nom du fichier, obligatoire
    parser.add_argument('--spectrogram', action='store_true')
    parser.set_defaults(spectrogram=False)

    args = parser.parse_args()
    print( "Read '{}' with option spectrogram={}".format(args.filename, args.spectrogram) )

    # chargement de l'audio
    audio = torchaudio.load(args.filename)[0][0]

    # Show Audio with or without spectrogram
    if args.spectrogram:
        fig = plt.figure(figsize=(1, 2))
        fig.add_subplot(211).set_title('Audio')
        plt.plot(audio.numpy())

        fig.add_subplot(212).set_title('MelSpectrogram')
        spectrogram = ToDB(ToSpectrogram(audio))
        plt.imshow(spectrogram, cmap='inferno')
        
    else:

        plt.plot(audio.numpy())


    plt.show()

if __name__ == '__main__':
    main()