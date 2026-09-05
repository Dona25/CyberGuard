"""Interactive CyberGuard cybersecurity-awareness chatbot."""
from generate import generate_answer


def main():
    print("CyberGuard - Cybersecurity Awareness Chatbot")
    print("Ask about phishing, passwords, malware, safe browsing, MFA, social engineering, and related awareness topics.")
    print("Type 'exit' to stop.\n")
    while True:
        question = input("You: ").strip()
        if question.lower() == "exit":
            print("Chatbot terminated.")
            break
        if not question:
            continue
        answer = generate_answer(question)
        print("CyberGuard:", answer if answer else "I could not generate a reliable response.")


if __name__ == "__main__":
    main()
