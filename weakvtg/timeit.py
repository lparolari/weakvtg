def get_delta(start_time: float, end_time: float):
    return int(end_time - start_time)


def get_hms(time: int):
    hours = time // 3600
    minutes = (time - hours * 3600) // 60
    seconds = time % 60

    return hours, minutes, seconds


def get_eta(delta: int, current: int, total: int):
    eta = delta * (total - current)
    return get_hms(eta)


def get_fancy_time(hours: int, minutes: int, seconds: int):
    return f"{hours:02}:{minutes:02}:{seconds:02}"


def get_fancy_eta(start_time: float, end_time: float, current: int, total: int):
    return get_fancy_time(*get_eta(get_delta(start_time, end_time), current, total))
