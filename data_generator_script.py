import json
import random

#Configuration
NUM_SAMPLES_PER_INTENT = 100
OUTPUT_FILE = "delivery_data.json"

#Components for variations
templates = {
    "get_address": [
        "{prefix} {order} ka {attr} {action}",
        "{attr} {action} please, {order} delivery hai",
        "Kahan jaana hai? {attr} {action}",
        "{order} delivery location {action}"
    ],
    "call_customer": [
        "{prefix} customer {call_verb} nahi kar raha",
        "Customer ko {call_verb} karo",
        "{prefix} call lagao customer ko",
        "Call {phone_issue}, customer se baat {action}"
    ],
    "mark_delivered": [
        "Order {deliver_verb} ho gaya",
        "{prefix} delivery done, mark kar do",
        "{order} de diya hai customer ko",
        "{deliver_verb}. Photo uploaded."
    ],
    "mark_picked_up": [
        "Order {pick_verb} liya hai",
        "Restrunt se nikal gaya hoon",
        "{prefix} parcel pick ho gaya",
        "Order received from shop"
    ],
    "report_delay": [
        "{prefix} {delay_reason} ki wajah se late honga",
        "Traffic bahut hai, {time} extra lagega",
        "Late honga, bike {breakdown}",
        "Delay reporting: {delay_reason}",
        "{delay_reason} ho gya hai"
    ],
    "navigation_help": [
        "Map {nav_issue} hai",
        "{attr} galat dikha raha hai",
        "{prefix} rasta nahi mil raha",
        "Google maps stuck ho gaya"
    ],
    "order_issue" : [
        "Sir {order} {cancel_verb} kar do",
        "{order} cancel karna hai, customer ne mana kiya",
        "Wrong order received, {prefix} kya karu?",
        "Item {issue_type} hai, return lena hai kya?",
        "{prefix} payment nahi ho raha, order issue hai"
    ],
    "customer_unavailable": [
        "Customer ka phone {phone_issue}",
        "Ghar band hai, koi nahi mil raha",
        "Waiting at gate, customer not responding",
        "3 baar call kiya, unreachable aa raha hai"
    ]
}

#Fillers for randomization
fillers = {
    "prefix": ["Bhai", "Bhaiya", "Sir", "Madam", "Hello", "Yaar", "Dada", ""],
    "order": ["Order", "Parcel", "Khana", "Item", "Next order"],
    "attr": ["address", "pata", "location", "ghar ka number", "jagah"],
    "action": ["batao", "dikhao", "bhejo", "send karo", "share kijiye"],
    "call_verb": ["pick", "call uthhao", "receive"],
    "deliver_verb": ["deliver", "delivered", "de diya", "paucha diya"],
    "cancel_verb" : ["cancel", "radd", "wapas"],
    "pick_verb": ["pick kar", "uthha", "le"],
    "delay_reason": ["Traffic", "Baarish", "Puncture", "Police checking","accident"],
    "time": ["5 mins", "10 min", "thoda time"],
    "breakdown": ["breakdown", "puncture", "kharab"],
    "nav_issue": ["not working", "stuck", "galat", "slow"],
    "issue_type": ["missing", "kharab", "leaked", "wrong","thanda"],
    "phone_issue": ["switch off", "busy", "not reachable", "kat diya","network issue"]
}

def generate_data():
    dataset = []
    
    for intent, patterns in templates.items():
        for _ in range(NUM_SAMPLES_PER_INTENT):
            #Choose a random template for the intent
            template = random.choice(patterns)
            
            #Use regex like replacement for all fillers found in the template
            text = template
            for key, values in fillers.items():
                placeholder = "{" + key + "}"
                if placeholder in text:
                    text = text.replace(placeholder, random.choice(values))
            
            #Clean up double spaces if a prefix was empty
            text = " ".join(text.split()).strip()
            
            dataset.append({
                "text": text,
                "intent": intent
            })
            
    random.shuffle(dataset)
    return dataset

if __name__ == "__main__":
    data = generate_data()
    with open(OUTPUT_FILE, "w") as f:
        json.dump(data, f, indent=4)
    
    print(f"Successfully generated {len(data)} samples in {OUTPUT_FILE}")