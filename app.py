from unmute_ai.gui import SignLanguageApp

if __name__ == '__main__':
    try:
        app = SignLanguageApp()
        app.run()
    except Exception as e:
        print(f"Error: {e}")
        exit(1)
