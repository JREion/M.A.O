from dassl.utils import Registry, check_availability

TRAINER_REGISTRY = Registry("TRAINER")


def build_trainer(cfg, name=None):
    avai_trainers = TRAINER_REGISTRY.registered_names()
    if name is None:
        check_availability(cfg.TRAINER.NAME, avai_trainers)
        if cfg.VERBOSE:
            print("Loading trainer: {}".format(cfg.TRAINER.NAME))
        return TRAINER_REGISTRY.get(cfg.TRAINER.NAME)(cfg)
    # [DPC] Multi-trainer support
    else:
        check_availability(name, avai_trainers)
        if cfg.VERBOSE:
            print("Loading trainer: {}".format(name))
        return TRAINER_REGISTRY.get(name)(cfg)
