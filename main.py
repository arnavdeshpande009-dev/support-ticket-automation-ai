import os
import pandas as pd
import logging
import json
import re

import logging
logging.basicConfig(level=logging.INFO, format="%(message)s")

logging.basicConfig(level=logging.INFO, format="%(asctime)s │ %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# ── AI toggle ─────────────────────────────────────────────────────────────────
USE_AI = True
AI_WORKING = True
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# ── Category rules (phrases +3, keywords +2, partials +1) ────────────────────
CATEGORY_RULES = {
    "Billing": {
        "phrases": [
            "payment failed", "money deducted", "charged twice",
            "refund not received", "invoice issue", "cancel subscription",
            "double charged", "give me my money", "overcharged",
        ],
        "keywords": [
            "payment", "refund", "billing", "charge", "invoice", "money",
            "transaction", "subscription", "receipt", "credit", "debit",
            "fee", "cash", "spend", "purchase", "bought", "cheque",
            "wallet", "promo", "coupon", "discount", "renewal", "paid",
            "cost", "price", "order id",
        ],
        "partials": ["pay", "refund", "bill", "charg", "invoic", "subscri"],
    },
    "Account": {
        "phrases": [
            "cannot login", "forgot password", "account locked",
            "access denied", "verification failed", "delete account",
            "remove user", "lost access", "locked out", "remove them",
        ],
        "keywords": [
            "login", "password", "account", "signup", "verification",
            "access", "locked", "profile", "register", "verify",
            "workspace", "seat", "identity", "certificate", "name update",
            "employee", "interviewer", "candidate", "inactivity", "reset",
            "restore", "permission", "admin", "owner", "2fa", "mfa",
            "deactivate", "reactivate", "sign up",
        ],
        "partials": ["login", "passw", "account", "verif", "access", "permiss"],
    },
    "Technical": {
        "phrases": [
            "not loading", "server down", "app crash", "error code",
            "not responding", "not working", "screen share",
            "compatibility check", "submissions not working",
        ],
        "keywords": [
            "error", "bug", "crash", "fail", "failing", "failed", "issue",
            "problem", "timeout", "blocker", "loading", "compatibility",
            "compatible", "submissions", "api", "bedrock", "crawl",
            "crawling", "vulnerability", "security", "lti", "down",
            "outage", "unresponsive", "freeze", "lag", "latency",
            "exception", "debug", "deploy", "integration", "configuration",
            "setup", "install", "update", "upgrade", "patch",
            "connectivity", "zoom", "broken", "fix", "glitch",
        ],
        "partials": ["err", "fail", "crash", "bug", "block", "load", "time"],
    },
    "Order": {
        "phrases": [
            "order not delivered", "wrong item", "delivery delay",
            "tracking issue", "package missing", "wrong product",
        ],
        "keywords": [
            "delivery", "shipping", "order", "tracking", "package",
            "return", "replacement", "exchange", "damaged", "parcel",
            "warehouse", "dispatch", "shipped", "missing item",
            "not received", "estimated time", "out of stock",
        ],
        "partials": ["deliver", "ship", "track", "packag"],
    },
}

# ── Priority keywords ─────────────────────────────────────────────────────────
HIGH_WORDS = [
    "urgent", "asap", "immediately", "critical", "not working", "failed",
    "error", "cannot", "unable", "blocked", "blocker", "emergency",
    "right now", "stolen", "theft", "down", "outage", "crash",
]
MED_WORDS = [
    "soon", "delay", "issue", "problem", "slow", "quick", "quickly",
    "fast", "expedite", "waiting", "pending",
]

# ── Anger / sentiment words ───────────────────────────────────────────────────
ANGER_WORDS = [
    "angry", "frustrated", "worst", "terrible", "hate", "bad", "furious",
    "disgusting", "pathetic", "ridiculous", "outrageous", "unacceptable",
    "scam", "fraud", "rip off", "useless", "garbage", "horrible",
    "disappointing", "fed up", "sick of", "unfair",
]

# ── Response templates ────────────────────────────────────────────────────────
RESPONSE_TEMPLATES = {
    "Billing": (
        "We understand how important billing accuracy is. Your concern has been "
        "escalated to our payments team, who will conduct a thorough review of "
        "the transaction in question. You can expect a detailed update within "
        "24 hours. In the meantime, please keep any receipts, order IDs, or "
        "confirmation emails handy so we can resolve this as quickly as possible."
    ),
    "Account": (
        "Your account security is our top priority. We have flagged your request "
        "for immediate review by our account management team. If you need to "
        "regain access, please try resetting your password via the login page. "
        "If the issue persists, our team will reach out with personalized steps "
        "to restore full access to your account."
    ),
    "Technical": (
        "We sincerely apologize for the technical disruption. Our engineering "
        "team has been alerted and is actively investigating the root cause. "
        "We are working to deliver a fix or workaround at the earliest. If "
        "possible, please share any error messages, screenshots, or steps to "
        "reproduce the issue — it will help us resolve it faster."
    ),
    "Order": (
        "We are sorry for the inconvenience with your order. Our fulfillment "
        "team is looking into the status and will provide a tracking update "
        "shortly. If your order arrived damaged or incorrect, we will arrange "
        "a replacement or refund at no extra cost. Thank you for your patience."
    ),
    "Other": (
        "Thank you for reaching out. We have logged your request and our support "
        "team will review it promptly. A dedicated representative will follow up "
        "with you shortly to ensure your concern is fully addressed."
    ),
}

VALID_CATEGORIES = {"Billing", "Account", "Technical", "Order", "Other"}
VALID_PRIORITIES = {"High", "Medium", "Low"}
OUTPUT_CATEGORIES = {"Billing", "Account", "Technical", "Other"}


# ── Rule-based functions ──────────────────────────────────────────────────────
def classify_category(text: str) -> str:
    text = str(text).lower()
    scores = {}
    for category, rules in CATEGORY_RULES.items():
        score = 0
        for phrase in rules["phrases"]:
            if phrase in text:
                score += 3
        for kw in rules["keywords"]:
            if kw in text:
                score += 2
        for partial in rules["partials"]:
            if partial in text:
                score += 1
        scores[category] = score

    best = max(scores, key=scores.get)
    if scores[best] > 0:
        return best

    if any(fb in text for fb in ["not working", "issue", "problem", "help", "blocked", "unable"]):
        return "Technical"
    if any(fb in text for fb in ["payment", "money", "refund", "charge"]):
        return "Billing"
    if any(fb in text for fb in ["login", "account", "password", "access"]):
        return "Account"
    return "Other"


def detect_anger(text: str) -> bool:
    text = str(text).lower()
    return any(word in text for word in ANGER_WORDS)


def classify_priority(text: str, is_angry: bool = False) -> str:
    if is_angry:
        return "High"
    text = str(text).lower()
    if any(kw in text for kw in HIGH_WORDS):
        return "High"
    if any(kw in text for kw in MED_WORDS):
        return "Medium"
    return "Low"


def generate_response(category: str) -> str:
    return RESPONSE_TEMPLATES.get(category, RESPONSE_TEMPLATES["Other"])


def generate_resolution(category: str) -> str:
    resolutions = {
        "Billing":   "Refund or billing verification initiated.",
        "Account":   "Password reset or account recovery suggested.",
        "Technical": "Try restarting app or clearing cache.",
        "Order":     "Check tracking status or request replacement.",
        "Other":     "Support team will investigate.",
    }
    return resolutions.get(category, resolutions["Other"])


def rule_based_process(text: str) -> dict:
    category = classify_category(text)
    angry    = detect_anger(text)
    priority = classify_priority(text, is_angry=angry)
    response = generate_response(category)
    return {"category": category, "priority": priority, "response": response, "angry": angry}


# ── AI availability check (runs once) ────────────────────────────────────────
_ai_available = False
try:
    import openai as _openai_mod
    if OPENAI_API_KEY:
        _ai_available = True
    else:
        log.info("OPENAI_API_KEY not set — AI mode disabled, using rule-based.")
except ImportError:
    log.info("openai package not installed — AI mode disabled, using rule-based.")


# ── AI-powered function ──────────────────────────────────────────────────────
def ai_process_ticket(text: str) -> dict | None:
    global AI_WORKING
    if not _ai_available or not AI_WORKING:
        return None

    prompt = (
        "Classify the following support ticket and generate a professional response.\n\n"
        "Return ONLY valid JSON in this exact format (no extra text):\n"
        '{\n'
        '  "category": "Billing" | "Account" | "Technical" | "Order" | "Other",\n'
        '  "priority": "High" | "Medium" | "Low",\n'
        '  "response": "your professional response here"\n'
        '}\n\n'
        f"Ticket:\n{text}"
    )

    try:
        client = _openai_mod.OpenAI(api_key=OPENAI_API_KEY)
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a support ticket classifier. Respond ONLY with valid JSON."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=300,
        )
        raw = completion.choices[0].message.content.strip()

        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not json_match:
            log.warning("AI returned non-JSON response — falling back.")
            return None

        data = json.loads(json_match.group())

        category = data.get("category", "Other")
        priority = data.get("priority", "Low")
        response = data.get("response", "")

        if category not in VALID_CATEGORIES:
            category = "Other"
        if priority not in VALID_PRIORITIES:
            priority = "Low"
        if not response:
            return None

        return {"category": category, "priority": priority, "response": response}

    except (json.JSONDecodeError, KeyError) as e:
        log.warning("AI JSON parse error (%s) — falling back.", e)
        return None
    except Exception as e:
        log.warning("AI API error (%s) — disabling AI for remaining tickets.", e)
        AI_WORKING = False
        return None


# ── Main pipeline ─────────────────────────────────────────────────────────────
def main():
    INPUT_FILE  = "support_tickets/support_tickets.csv"
    OUTPUT_FILE = "support_tickets/output.csv"

    mode = "AI + fallback" if USE_AI else "rule-based"
    log.info("Mode: %s", mode)
    log.info("Reading %s …", INPUT_FILE)

    df = pd.read_csv(INPUT_FILE)
    df.columns = df.columns.str.strip().str.lower()
    log.info("Columns detected: %s", list(df.columns))
    log.info("Total rows: %d", len(df))

    df["ticket_id"] = range(1, len(df) + 1)
    df["issue"]     = df["issue"].fillna("").astype(str)
    df["subject"]   = df["subject"].fillna("").astype(str)
    df["text"]      = (df["issue"] + " " + df["subject"]).str.strip().str.lower()
    log.info("Preprocessing complete ✔")

    results = []
    anger_count   = 0
    ai_count      = 0
    fallback_count = 0
    cat_counts    = {}

    for _, row in df.iterrows():
        text = row["text"]
        tid  = row["ticket_id"]
        used_ai = False
        angry   = False

        if USE_AI and AI_WORKING:
            ai_result = ai_process_ticket(text)
            if ai_result:
                category = ai_result["category"]
                priority = ai_result["priority"]
                response = ai_result["response"]
                used_ai  = True
                ai_count += 1

        if not used_ai:
            rb = rule_based_process(text)
            category = rb["category"]
            priority = rb["priority"]
            response = rb["response"]
            angry    = rb["angry"]
            if angry:
                anger_count += 1
            fallback_count += 1

        confidence = 0.9 if category != "Other" else 0.5
        escalate   = priority == "High"
        resolution = generate_resolution(category)

        # Map Order → Other for official output format
        output_category = category if category in OUTPUT_CATEGORIES else "Other"
        output_response = generate_response(output_category) if output_category != category else response

        cat_counts[output_category] = cat_counts.get(output_category, 0) + 1

        results.append({
            "ticket_id": tid,
            "category":  output_category,
            "priority":  priority,
            "response":  output_response,
        })

        source = "🤖 AI" if used_ai else "📏 Rules"
        angry_flag = " 🔴 ANGRY" if angry else ""
        esc_flag   = " ⚡ ESCALATE" if escalate else ""
        log.info("  Ticket #%d → %s | %s | conf=%.1f  [%s]%s%s", tid, output_category, priority, confidence, source, angry_flag, esc_flag)

    out_df = pd.DataFrame(results)
    out_df.to_csv(OUTPUT_FILE, index=False)

    log.info("─" * 55)
    log.info("Output written to %s (%d tickets) ✔", OUTPUT_FILE, len(out_df))
    log.info("Category breakdown: %s", dict(sorted(cat_counts.items())))
    if USE_AI:
        log.info("AI-processed: %d | Rule-based fallback: %d", ai_count, fallback_count)
    log.info("Anger-escalated tickets: %d", anger_count)
    log.info("─" * 55)


if __name__ == "__main__":
    main()
