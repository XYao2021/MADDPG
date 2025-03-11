def compute_decay(init=1.0,decay=1-2e-4,n_slots=2000,stop=0.5):
    r"""compute explore noise std after n_slots*stop slots
        INPUT:
            init: initial noise
            decay: decay rate
            stop: [0,1]
    """
    std = init
    for _ in range(int(n_slots*stop)):
        std = std * decay
    return std


if __name__ == "__main__":
    noise = compute_decay(init=1,decay=1-2e-4,n_slots=50000,stop=1)
    print(f"{noise:.4f}")