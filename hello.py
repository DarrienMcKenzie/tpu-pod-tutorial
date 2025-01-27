import jax

def main():
    jax.distributed.initialize()
    if jax.process_index() == 0:
        print("Hello again from tpu-pod-tutorial!")


if __name__ == "__main__":
    main()
