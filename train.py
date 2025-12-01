from unmute_ai.trainer import train_model

if __name__ == '__main__':
    try:
        train_model()
    except Exception as e:
        print(f"Error: {e}")
        exit(1)
