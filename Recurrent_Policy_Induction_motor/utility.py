
def polynomial_decay(initial: float, final: float, max_decay_steps: int, power: float, current_step: int) -> float:
    """
    Decays the hyperparameters polynomially. If power is 1.0, then linearly.
    max_decay_steps: The maximum numbers of steps to decay the hyperparameter
    power: strength of decay

    max_decay_steps should be equal to number of times this function is called for proper decaying.
    """
    #  Return the final value if max_decay_steps is reached or if initial = final value.
    if current_step > max_decay_steps or initial == final:
        return final
    else:
        return (initial - final) * ((1 - current_step / max_decay_steps) ** power) + final


