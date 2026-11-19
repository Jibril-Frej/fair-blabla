# -*- coding: utf-8 -*-
"""English word lists for biasscan.

Name lists stay EMPTY: supply yours with --names. Which names read as gendered
depends on the corpus, so no shipped list is right in general.
"""
from biasscan import Lang

LANG = Lang(
    name="English",
    segmentation="word",
    auto_plural=True,
    window=40,
    decimal_comma=False,

    pronouns={
        "M": ["he", "him", "his", "himself"],
        "F": ["she", "her", "hers", "herself"],
    },

    # kinship terms and any noun that marks gender on its own
    kinship={
        "M": ["father", "dad", "daddy", "papa", "grandfather", "grandpa", "brother", "son", 
            "uncle", "nephew", "husband", "boy", "boys", "man", "men", "gentleman", "mr", 
            "sir", "king", "prince", "stepfather", "godfather", "schoolboy"],
        "F": ["mother", "mom", "mum", "mommy", "mama", "grandmother", "grandma", "sister", 
            "daughter", "aunt", "niece", "wife", "girl", "girls", "woman", "women", "lady", 
            "mrs", "ms", "miss", "madam", "queen", "princess", "stepmother", "godmother", 
            "schoolgirl"],
    },

    occupations={
        "teacher": ["teacher", "principal", "headmaster", "professor", "tutor", "coach", 
            "instructor", "lecturer"],
        "doctor": ["doctor", "physician", "surgeon", "dentist", "veterinarian", "vet"],
        "nurse": ["nurse", "midwife", "caregiver"],
        "agriculture": ["farmer", "rancher", "fisherman", "shepherd", "gardener", 
            "beekeeper", "orchardist"],
        "manual_worker": ["worker", "laborer", "labourer", "builder", "carpenter", 
            "plumber", "electrician", "mechanic", "welder", "mason", "bricklayer", "miner", 
            "factory worker"],
        "driver": ["driver", "bus driver", "truck driver", "taxi driver", "chauffeur", 
            "conductor"],
        "retail_service": ["cashier", "clerk", "shopkeeper", "salesperson", "salesman", 
            "saleswoman", "waiter", "waitress", "server", "barista", "cook", "chef", 
            "baker", "butcher", "tailor", "seamstress", "hairdresser", "barber", "cleaner", 
            "janitor", "maid", "housekeeper", "nanny", "babysitter", "courier", 
            "delivery driver", "mail carrier", "postman"],
        "business_owner": ["owner", "boss", "manager", "merchant", "trader", "vendor", 
            "entrepreneur", "ceo", "director", "banker", "broker", "landlord", "landlady"],
        "professional": ["engineer", "scientist", "programmer", "developer", "architect", 
            "accountant", "lawyer", "attorney", "journalist", "reporter", "editor", 
            "translator", "researcher", "technician", "analyst", "mathematician", 
            "physicist", "chemist", "biologist", "economist", "pharmacist"],
        "public_military": ["police officer", "policeman", "policewoman", "firefighter", 
            "fireman", "soldier", "sailor", "officer", "pilot", "astronaut", "judge", 
            "mayor", "senator", "civil servant", "guard"],
        "arts_sports": ["athlete", "player", "actor", "actress", "singer", "painter", 
            "writer", "author", "musician", "dancer", "photographer", "designer", 
            "choreographer"],
        "clerical": ["secretary", "receptionist", "typist", "assistant", "librarian"],
        "student": ["student", "pupil", "classmate", "schoolchild", "freshman", 
            "undergraduate"],
    },

    activities={
        "housework": ["cook", "cooks", "cooked", "cooking", "bake", "bakes", "baked", 
            "baking", "wash the dishes", "washes the dishes", "washed the dishes", 
            "do the dishes", "did the dishes", "laundry", "iron", "ironed", "ironing", 
            "sweep", "swept", "sweeping", "mop", "mopped", "vacuum", "vacuumed", "dust", 
            "dusted", "tidy", "tidied", "clean the house", "cleaned the house", "knit", 
            "knits", "knitted", "knitting", "sew", "sews", "sewed", "sewing", "babysit", 
            "babysat", "feed the baby", "housework", "chores", "grocery"],
        "repair_build": ["repair", "repairs", "repaired", "repairing", "fix", "fixes", 
            "fixed", "fixing", "build", "builds", "built", "building", "install", 
            "installed", "installing", "weld", "welded", "drill", "drilled", "dig", "digs", 
            "dug", "digging", "paint the fence", "painted the fence", "assemble", 
            "assembled", "renovate", "renovated", "drive", "drives", "drove", "driving"],
        "shopping": ["buy", "buys", "bought", "buying", "purchase", "purchases", 
            "purchased", "spend", "spends", "spent", "shop", "shops", "shopped", 
            "shopping", "pay", "pays", "paid", "order", "ordered", "sale", "discount"],
        "sports": ["run", "runs", "ran", "running", "jog", "jogged", "swim", "swims", 
            "swam", "swimming", "play football", "play soccer", "play basketball", 
            "play tennis", "played basketball", "played soccer", "cycle", "cycled", "bike", 
            "biked", "ride a bike", "rode a bike", "jump rope", "race", "raced", "climb", 
            "climbed", "train", "trained", "practice", "practiced", "match", "tournament"],
        "study": ["read", "reads", "reading", "study", "studies", "studied", "studying", 
            "homework", "write", "writes", "wrote", "writing", "revise", "revised", 
            "review", "reviewed", "solve", "solves", "solved", "calculate", "calculated", 
            "practise", "draw", "drew", "drawing", "exam", "test", "quiz"],
        "labor_production": ["work", "works", "worked", "working", "job", "shift", 
            "produce", "produces", "produced", "manufacture", "manufactured", "harvest", 
            "harvested", "plant", "planted", "sell", "sells", "sold", "selling", "deliver", 
            "delivered", "earn", "earns", "earned", "wage", "salary", "overtime"],
    },

    dimensions={
        "ethnicity": ["white people", "white students", "black people", "black students", 
            "asian people", "asian students", "hispanic", "latino", "latina", 
            "african american", "african-american", "native american", "indigenous", 
            "arab", "jewish", "muslim", "hindu", "immigrant", "foreigner", 
            "ethnic minority", "caucasian", "ethnicity", "racial", "racism", "racist", 
            "people of color", "person of color", "biracial"],
        "rural": ["village", "farm", "countryside", "rural", "ranch", "barn", "crop", 
            "harvest", "livestock", "small town", "poverty", "farmland", "peasant", "hamlet"],
        "urban": ["city", "downtown", "urban", "subway", "metro", "apartment", 
            "skyscraper", "office building", "mall", "supermarket", "bus stop", "suburb"],
        "socioeconomic": ["allowance", "pocket money", "salary", "wage", "income", "bonus", 
            "donation", "poverty", "wealthy", "savings", "mortgage", "welfare", 
            "scholarship", "debt", "unemployed", "minimum wage", "working class", 
            "upper class", "well-off", "broke"],
        "appearance": ["obese", "obesity", "overweight", "underweight", "body weight", 
            "body height", "beautiful", "handsome", "ugly", "on a diet", "dieting", 
            "lose weight", "lost weight", "good-looking", "slim figure", "skinny", 
            "attractive", "unattractive", "well-dressed", "scruffy", "bald", "wrinkled"],
    },

    # counted only with --include-ambiguous: real descriptors in prose,
    # measurements in technical text
    ambiguous={
        "ethnicity": ["white", "black", "asian", "race", "oriental", "colored"],
        "appearance": ["fat", "thin", "slim", "height", "weight", "tall", "short", "heavy", 
            "skin", "hair", "young", "old", "pretty"],
        "socioeconomic": ["cheap", "expensive", "class", "poor", "rich", "interest", "rent"],
    },

    money=r"(?:\$\s*([\d,]+(?:\.\d+)?)|([\d,]+(?:\.\d+)?)\s*dollars?)",
    money_unit="USD ($)",
)
