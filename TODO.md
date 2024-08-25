1. Commentare e pulire codice (BOTH)
2. Scrivere readme con: (BOTH)
    - dipendenze (pacchetti da installare), completo di versioni specifiche testate
    - hardware utilizzato (scheda video, etc)
    - setup ambiente di sviluppo
    - compilazione e esecuzione sorgenti
3. Dalla simulazione, prendere: (D)
    - video test task di hovering
    - video test training con ostacoli 
    - immagine della camera del drone
    - dettaglio del drone
 
=====================
 
Max 15 slides about project technical content no too texty, more figures, tables,
schemes and plots
– Include objectives reached/not reached
– Demo video (max 5 min)
 
Mail con obiettivi:
Ho letto la vostra idea di progetto: fondamentalmente possiamo suddividere il paper in una parte di visione, o perception, una parte di controllo e una parte dove sviluppano metodi per migliorare l’algoritmo di RL.
 
Per gli obiettivi del corso possiamo trascurare la parte di miglioramento dell’algoritmo di RL.
 
Per quanto riguarda le parti restanti vi proporrei di implementarle in modo “end-to-end”. Ovvero connettere direttamente l’input di visione a un modello di NN che ha come output le azioni di controllo per il drone.
In linea di massima una prima versione di step di sviluppo del progetto può essere:
 
 
Indice presentazione:
1. Slide introduttiva scrivendo i punti della mail: (2 slide) DAN
    - Studio dell’algoritmo di RL chiamato PPO e della libreria SKRL che userete per il deploy su Isaac
    - Implementare un ambiente simulato (semplice) su Isaac Gym per l’obstacle avoidance
    - Implementare un modello NN con SKRL che riceva in input un’immagine e abbia come output 4 azioni CTBR ( differenti modi di controllare un drone  https://rpg.ifi.uzh.ch/docs/ICRA22_Kaufmann.pdf ) compatibili con un controllore di volo che vi daremo noi
    - Design di una reward per obstacle avoidance
    - Train and test
   
2. Background: PPO, SKRL, Isaac, come interagiscono tra loro? a cosa servono? DAN
    - un po' di dettaglio su PPO, caratteristiche/idea generale (1 slide)
    - SKRL e pytorch ruolo e funzionamento (1 slide)
    - Isaac, come si interfaccia con SKRL (task, env, etc) (1 slide)
 
3. Focus su IsaacSim: -> DAN
    - Schema a blocchi riassuntivo di come si crea un'ambeinte di simulazione in Isaac (create_sim, etc...) (1 slide)
    - Schreenshot ambiente con ostacoli e drone
 
4. Focus su SKRL con drone in hovering: MAT
    - Schema a blocchi su come è stato implementato il task di hovering (1 slide)
    - Spiegare come è stato definito il modello NN per il task di hovering (1 slide)
    - Spiegazione definizione reward function per hovering (1 slide)
    - Video drone in hovering su target palla gialla
    - Spiegare wandb e mettere screenshots (1 slide)
 
5. SKRL con camere e ostacoli MAT
    - come vengono gestite le camere e l'obstacle avoidance (1 slide)
    - come abbiamo definito la reward function per l'ostacle avoidance (1 slide)
    - Tentare di giustificare il training non corretto (1 slide)
 
6. Slide conclusiva con obiettivi raggiunt e non (1 slide) DAN
 
Totale slides: 14 slides + 2 di video/screenshots
 
Reminder: dare un'occhiata ai vari post sul gruppo