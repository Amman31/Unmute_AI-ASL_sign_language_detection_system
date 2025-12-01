from unmute_ai.data_processor import create_dataset

if __name__ == '__main__':
    try:
        create_dataset()
    except Exception as e:
        print(f"Error: {e}")
        exit(1)
