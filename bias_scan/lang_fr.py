# -*- coding: utf-8 -*-
"""French word lists for biasscan.

Name lists stay EMPTY: supply yours with --names.

Two things French needs that English does not.

1. Job titles carry gender. "une infirmiere" tells you the nurse is a woman
   with no pronoun and no name, so the inflected forms are listed in `kinship`
   as gender signals. That raises coverage a lot. The flip side: a gendered job
   title must not be allowed to gender ITSELF, or occupation-vs-gender just
   rediscovers French grammar. biasscan masks the occupation word before
   reading gender, which is why section 3 is the honest one and section 2 is
   the weaker claim in this language.

2. The masculine is the generic plural ("les instituteurs" can be a mixed
   group), so "ils" is deliberately NOT in the pronoun list while "elles" is.
   Masculine counts still run high for grammatical reasons -- compare
   directions with other languages, never the raw M:F numbers.
"""
from biasscan import Lang

# Inflected job titles, used as gender signals as well as occupations.
JOB_M = [
    "infirmier", "infirmiers", "instituteur", "instituteurs", "directeur",
    "directeurs", "vendeur", "vendeurs", "serveur", "serveurs", "caissier",
    "caissiers", "coiffeur", "coiffeurs", "ouvrier", "ouvriers", "agriculteur",
    "agriculteurs", "conducteur", "conducteurs", "facteur", "facteurs", "acteur",
    "acteurs", "chanteur", "chanteurs", "danseur", "danseurs", "couturier",
    "couturiers", "assistant", "assistants", "enseignant", "enseignants", "étudiant",
    "étudiants", "employé", "employés", "patron", "patrons", "cuisinier", "cuisiniers",
    "boulanger", "boulangers", "épicier", "fermier", "fermiers", "avocat", "avocats",
    "ingénieur", "ingénieurs", "chercheur", "chercheurs", "informaticien",
    "technicien", "techniciens", "policier", "policiers", "pompier", "pompiers",
    "soldat", "soldats", "président", "présidents", "joueur", "joueurs", "sportif",
    "sportifs", "écrivain", "écrivains", "paysan", "paysans"
]
JOB_F = [
    "infirmière", "infirmières", "institutrice", "institutrices", "directrice",
    "directrices", "vendeuse", "vendeuses", "serveuse", "serveuses", "caissière",
    "caissières", "coiffeuse", "coiffeuses", "ouvrière", "ouvrières", "agricultrice",
    "agricultrices", "conductrice", "conductrices", "factrice", "factrices", "actrice",
    "actrices", "chanteuse", "chanteuses", "danseuse", "danseuses", "couturière",
    "couturières", "femme de ménage", "femmes de ménage", "nourrice", "nourrices",
    "assistante", "assistantes", "enseignante", "enseignantes", "étudiante",
    "étudiantes", "employée", "employées", "patronne", "patronnes", "cuisinière",
    "cuisinières", "boulangère", "boulangères", "épicière", "fermière", "fermières",
    "avocate", "avocates", "ingénieure", "ingénieures", "chercheuse", "chercheuses",
    "informaticienne", "informaticiennes", "technicienne", "techniciennes",
    "policière", "policières", "pompière", "soldate", "présidente", "présidentes",
    "joueuse", "joueuses", "sportive", "sportives", "écrivaine", "écrivaines",
    "autrice", "réalisatrice", "comédienne", "comédiennes", "musicienne",
    "pharmacienne", "rédactrice", "traductrice", "députée", "gérante", "entrepreneuse",
    "banquière", "commerçante", "pâtissière", "bouchère", "bergère", "éleveuse",
    "pêcheuse", "jardinière", "électricienne", "mécanicienne", "soudeuse", "artisane",
    "chauffeuse", "routière", "camionneuse", "livreuse", "gardienne", "maçonne",
    "professeure", "docteure", "chirurgienne", "aide-soignante", "puéricultrice",
    "lycéenne", "collégienne", "écolière", "apprentie", "surveillante", "formatrice",
    "viticultrice", "programmeuse", "développeuse", "danseuses"
]

# EPICENE: one form for both genders ("le/la secretaire", "un/une medecin").
# These must NEVER be gender signals on their own -- listing one by mistake
# marks every man holding that job as a woman. The ARTICLE carries the gender
# instead, so determiner+noun pairs are generated below. Elision ("l'architecte")
# hides the article and correctly stays unmarked.
EPICENE = [
    "secrétaire", "médecin", "comptable", "pilote", "juge", "architecte", "dentiste",
    "vétérinaire", "journaliste", "élève", "bibliothécaire", "concierge",
    "propriétaire", "athlète", "mannequin", "cadre", "ministre", "gendarme",
    "militaire", "peintre", "notaire", "psychiatre", "chimiste", "biologiste",
    "économiste", "analyste", "scientifique", "manager", "réceptionniste",
    "standardiste", "apprenti", "interprète", "guide"
]
DET_M, DET_F = ["le", "un", "ce", "du", "au"], ["la", "une", "cette", "de la", "à la"]
EPI_M = [f"{d} {n}" for n in EPICENE for d in DET_M]
EPI_F = [f"{d} {n}" for n in EPICENE for d in DET_F]

LANG = Lang(
    name="French",
    segmentation="word",
    auto_plural=False,          # French inflection is not a suffix rule
    window=40,
    decimal_comma=True,         # "12,50" is twelve fifty, not 1250

    pronouns={
        "M": [
            "il", "celui-ci", "celui"
        ],
        "F": [
            "elle", "elles", "celle-ci", "celle"
        ],
    },

    kinship={
        "M": [
            "père", "papa", "grand-père", "papy", "frère", "fils", "oncle", "neveu",
            "mari", "époux", "garçon", "garçons", "homme", "hommes", "monsieur",
            "messieurs", "m.", "beau-père", "parrain", "roi", "prince", "veuf",
            "petit-fils", "demi-frère", "copain", "petit ami", "infirmier",
            "infirmiers", "instituteur", "instituteurs", "directeur", "directeurs",
            "vendeur", "vendeurs", "serveur", "serveurs", "caissier", "caissiers",
            "coiffeur", "coiffeurs", "ouvrier", "ouvriers", "agriculteur",
            "agriculteurs", "conducteur", "conducteurs", "facteur", "facteurs",
            "acteur", "acteurs", "chanteur", "chanteurs", "danseur", "danseurs",
            "couturier", "couturiers", "assistant", "assistants", "enseignant",
            "enseignants", "étudiant", "étudiants", "employé", "employés", "patron",
            "patrons", "cuisinier", "cuisiniers", "boulanger", "boulangers", "épicier",
            "fermier", "fermiers", "avocat", "avocats", "ingénieur", "ingénieurs",
            "chercheur", "chercheurs", "informaticien", "technicien", "techniciens",
            "policier", "policiers", "pompier", "pompiers", "soldat", "soldats",
            "président", "présidents", "joueur", "joueurs", "sportif", "sportifs",
            "écrivain", "écrivains", "paysan", "paysans"
        ] + JOB_M + EPI_M,
        "F": [
            "mère", "maman", "grand-mère", "mamie", "sœur", "soeur", "fille", "filles",
            "tante", "nièce", "femme", "femmes", "épouse", "madame", "mesdames",
            "mademoiselle", "mme", "belle-mère", "marraine", "reine", "princesse",
            "veuve", "petite-fille", "demi-sœur", "copine", "petite amie",
            "infirmière", "infirmières", "institutrice", "institutrices", "directrice",
            "directrices", "vendeuse", "vendeuses", "serveuse", "serveuses",
            "caissière", "caissières", "coiffeuse", "coiffeuses", "ouvrière",
            "ouvrières", "agricultrice", "agricultrices", "conductrice",
            "conductrices", "factrice", "factrices", "actrice", "actrices",
            "chanteuse", "chanteuses", "danseuse", "danseuses", "couturière",
            "couturières", "femme de ménage", "femmes de ménage", "nourrice",
            "nourrices", "assistante", "assistantes", "enseignante", "enseignantes",
            "étudiante", "étudiantes", "employée", "employées", "patronne",
            "patronnes", "cuisinière", "cuisinières", "boulangère", "boulangères",
            "épicière", "fermière", "fermières", "avocate", "avocates", "ingénieure",
            "ingénieures", "chercheuse", "chercheuses", "informaticienne",
            "informaticiennes", "technicienne", "techniciennes", "policière",
            "policières", "pompière", "soldate", "présidente", "présidentes",
            "joueuse", "joueuses", "sportive", "sportives", "écrivaine", "écrivaines",
            "autrice", "réalisatrice", "comédienne", "comédiennes", "musicienne",
            "pharmacienne", "rédactrice", "traductrice", "députée", "gérante",
            "entrepreneuse", "banquière", "commerçante", "pâtissière", "bouchère",
            "bergère", "éleveuse", "pêcheuse", "jardinière", "électricienne",
            "mécanicienne", "soudeuse", "artisane", "chauffeuse", "routière",
            "camionneuse", "livreuse", "gardienne", "maçonne", "professeure",
            "docteure", "chirurgienne", "aide-soignante", "puéricultrice", "lycéenne",
            "collégienne", "écolière", "apprentie", "surveillante", "formatrice",
            "viticultrice", "programmeuse", "développeuse", "danseuses"
        ] + JOB_F + EPI_F,
    },

    occupations={
        "teacher": [
            "enseignant", "enseignante", "enseignants", "enseignantes", "professeur",
            "professeure", "professeurs", "prof", "instituteur", "institutrice",
            "instituteurs", "institutrices", "maître d'école", "maîtresse d'école",
            "directeur d'école", "directrice d'école", "entraîneur", "entraîneuse",
            "formateur", "formatrice"
        ],
        "doctor": [
            "médecin", "médecins", "docteur", "docteure", "docteurs", "chirurgien",
            "chirurgienne", "dentiste", "vétérinaire", "psychiatre", "généraliste"
        ],
        "nurse": [
            "infirmier", "infirmière", "infirmiers", "infirmières", "aide-soignant",
            "aide-soignante", "sage-femme", "puéricultrice"
        ],
        "agriculture": [
            "agriculteur", "agricultrice", "agriculteurs", "agricultrices", "paysan",
            "paysanne", "paysans", "fermier", "fermière", "fermiers", "éleveur",
            "éleveuse", "pêcheur", "pêcheuse", "berger", "bergère", "viticulteur",
            "viticultrice", "jardinier", "jardinière"
        ],
        "manual_worker": [
            "ouvrier", "ouvrière", "ouvriers", "ouvrières", "maçon", "maçonne",
            "charpentier", "menuisier", "plombier", "électricien", "électricienne",
            "mécanicien", "mécanicienne", "soudeur", "soudeuse", "mineur", "manœuvre",
            "artisan", "artisane"
        ],
        "driver": [
            "conducteur", "conductrice", "conducteurs", "chauffeur", "chauffeuse",
            "chauffeurs", "routier", "routière", "camionneur", "camionneuse", "taxi",
            "livreur", "livreuse"
        ],
        "retail_service": [
            "vendeur", "vendeuse", "vendeurs", "vendeuses", "caissier", "caissière",
            "caissiers", "caissières", "serveur", "serveuse", "serveurs", "serveuses",
            "commerçant", "commerçante", "cuisinier", "cuisinière", "chef cuisinier",
            "boulanger", "boulangère", "pâtissier", "pâtissière", "boucher",
            "bouchère", "épicier", "épicière", "coiffeur", "coiffeuse", "couturier",
            "couturière", "tailleur", "femme de ménage", "homme de ménage",
            "agent d'entretien", "concierge", "gardien", "gardienne", "nourrice",
            "nounou", "baby-sitter", "facteur", "factrice", "serveur de café"
        ],
        "business_owner": [
            "patron", "patronne", "patrons", "patronnes", "chef d'entreprise",
            "directeur", "directrice", "directeurs", "directrices", "gérant",
            "gérante", "propriétaire", "commerçant", "entrepreneur", "entrepreneuse",
            "banquier", "banquière", "pdg", "cadre", "manager"
        ],
        "professional": [
            "ingénieur", "ingénieure", "ingénieurs", "ingénieures", "scientifique",
            "chercheur", "chercheuse", "chercheurs", "chercheuses", "informaticien",
            "informaticienne", "programmeur", "programmeuse", "développeur",
            "développeuse", "architecte", "comptable", "avocat", "avocate", "avocats",
            "notaire", "journaliste", "rédacteur", "rédactrice", "traducteur",
            "traductrice", "technicien", "technicienne", "mathématicien",
            "mathématicienne", "physicien", "physicienne", "chimiste", "biologiste",
            "économiste", "pharmacien", "pharmacienne", "analyste"
        ],
        "public_military": [
            "policier", "policière", "policiers", "policières", "gendarme", "pompier",
            "pompière", "pompiers", "soldat", "soldate", "soldats", "militaire",
            "officier", "officière", "pilote", "astronaute", "juge", "maire", "député",
            "députée", "fonctionnaire", "ministre", "président", "présidente",
            "surveillant", "surveillante"
        ],
        "arts_sports": [
            "athlète", "sportif", "sportive", "joueur", "joueuse", "joueurs",
            "joueuses", "acteur", "actrice", "acteurs", "actrices", "chanteur",
            "chanteuse", "chanteurs", "chanteuses", "peintre", "écrivain", "écrivaine",
            "auteur", "autrice", "auteure", "musicien", "musicienne", "danseur",
            "danseuse", "réalisateur", "réalisatrice", "photographe", "mannequin",
            "comédien", "comédienne"
        ],
        "clerical": [
            "secrétaire", "réceptionniste", "assistant", "assistante",
            "employé de bureau", "employée de bureau", "bibliothécaire", "standardiste"
        ],
        "student": [
            "étudiant", "étudiante", "étudiants", "étudiantes", "élève", "élèves",
            "lycéen", "lycéenne", "collégien", "collégienne", "écolier", "écolière",
            "apprenti", "apprentie"
        ],
    },

    activities={
        "housework": [
            "cuisiner", "cuisine", "cuisiné", "faire la cuisine", "fait la cuisine",
            "faire la vaisselle", "fait la vaisselle", "laver", "lave", "lavé",
            "lessive", "linge", "repasser", "repasse", "repassé", "balayer", "balaie",
            "balayé", "nettoyer", "nettoie", "nettoyé", "ménage", "faire le ménage",
            "ranger", "range", "rangé", "aspirateur", "coudre", "coud", "tricoter",
            "tricote", "garder les enfants", "s'occuper des enfants", "courses",
            "faire les courses"
        ],
        "repair_build": [
            "réparer", "répare", "réparé", "bricoler", "bricole", "construire",
            "construit", "bâtir", "bâti", "installer", "installe", "installé",
            "souder", "percer", "perce", "creuser", "creuse", "monter", "monte",
            "monté", "rénover", "rénove", "rénové", "peindre", "peint", "conduire",
            "conduit"
        ],
        "shopping": [
            "acheter", "achète", "acheté", "achat", "achats", "dépenser", "dépense",
            "dépensé", "payer", "paye", "paie", "payé", "commander", "commande",
            "commandé", "magasin", "soldes", "promotion", "réduction"
        ],
        "sports": [
            "courir", "court", "couru", "jogging", "nager", "nage", "nagé",
            "jouer au football", "joue au football", "jouer au basket",
            "jouer au tennis", "vélo", "faire du vélo", "cyclisme", "sauter", "saute",
            "escalade", "grimper", "s'entraîner", "entraînement", "match",
            "compétition", "tournoi"
        ],
        "study": [
            "lire", "lit", "lu", "lecture", "étudier", "étudie", "étudié", "devoirs",
            "écrire", "écrit", "réviser", "révise", "révisé", "apprendre", "apprend",
            "appris", "calculer", "calcule", "résoudre", "résout", "examen",
            "contrôle", "interrogation", "dessiner", "dessine"
        ],
        "labor_production": [
            "travailler", "travaille", "travaillé", "travail", "emploi", "boulot",
            "produire", "produit", "fabriquer", "fabrique", "fabriqué", "récolter",
            "récolte", "planter", "plante", "vendre", "vend", "vendu", "livrer",
            "livre", "gagner", "gagne", "gagné", "salaire", "heures supplémentaires"
        ],
    },

    dimensions={
        "ethnicity": [
            "arabe", "arabes", "maghrébin", "maghrébine", "africain", "africaine",
            "asiatique", "asiatiques", "antillais", "antillaise", "juif", "juive",
            "musulman", "musulmane", "chrétien", "chrétienne", "immigré", "immigrée",
            "immigrés", "immigration", "étranger", "étrangère", "étrangers", "beur",
            "roms", "tsigane", "gitan", "gitane", "racisme", "raciste", "ethnique",
            "minorité", "sans-papiers", "personne de couleur", "métis", "métisse"
        ],
        "rural": [
            "village", "villages", "campagne", "rural", "rurale", "ferme", "fermes",
            "champ", "champs", "récolte", "paysan", "paysanne", "province", "bourg",
            "hameau", "terroir", "agricole"
        ],
        "urban": [
            "ville", "villes", "urbain", "urbaine", "centre-ville", "métro",
            "banlieue", "banlieues", "quartier", "cité", "hlm", "immeuble",
            "appartement", "gratte-ciel", "périphérie", "arrondissement"
        ],
        "socioeconomic": [
            "argent de poche", "salaire", "smic", "revenu", "prime", "pauvreté",
            "fortune", "épargne", "hypothèque", "rmi", "rsa", "chômage", "chômeur",
            "chômeuse", "précarité", "dette", "dettes", "aisé", "modeste", "sans-abri",
            "smicard"
        ],
        "appearance": [
            "obèse", "obésité", "surpoids", "maigre", "mince", "corpulence", "laid",
            "laide", "moche", "régime", "maigrir", "grossir", "séduisant",
            "séduisante", "élégant", "élégante", "négligé", "chauve", "ridé"
        ],
    },

    # counted only with --include-ambiguous: real descriptors in prose,
    # measurements or polysemous words in technical text
    ambiguous={
        "ethnicity": [
            "noir", "noire", "noirs", "blanc", "blanche", "blancs", "race", "couleur",
            "origine"
        ],
        "appearance": [
            "taille", "poids", "grand", "grande", "petit", "petite", "gros", "grosse",
            "mince", "jeune", "vieux", "vieille", "peau", "cheveux", "beau", "belle",
            "joli", "jolie"
        ],
        "socioeconomic": [
            "classe", "cher", "chère", "coût", "niveau", "intérêt", "loyer", "pauvre",
            "pauvres", "riche", "riches", "don"
        ],
    },

    money=r"(?:([\d  .]+(?:,\d+)?)\s*(?:€|euros?|EUR)|(?:€|EUR)\s*([\d  .,]+))",
    money_unit="EUR (€)",
)
