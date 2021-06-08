from model import *
import json
from collections import namedtuple

if __name__ == '__main__':
    args = open('config.json').read()
    args = json.loads(args)
    args = namedtuple('args', args.keys())(*args.keys())

    early_stop_callback = EarlyStopping(monitor='val_loss', patience=5, strict=False, verbose=True, mode='min')
    trainer_args = {'gpus' : -1, 'max_epochs' : args.max_epoch, 'val_check_interval' : args.val_check_interval}

    trainer = pl.Trainer(**trainer_args)
    model = LitQGModel(args)
    dm = SquadDataModule(args)

    trainer.fit(model, dm)