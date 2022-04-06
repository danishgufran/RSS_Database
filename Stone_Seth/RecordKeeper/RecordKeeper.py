import pandas as pd
from dataclasses import dataclass
import pickle
from datetime import datetime


@dataclass(frozen=True)
class CHECKPOINT_TYPES:
    PICKLE = "RecordKeeper"
    CSV = "csv"

    @staticmethod
    def detect(file_target: str):

        mime = file_target.split(".")[-1]

        if str(mime).lower() == str(CHECKPOINT_TYPES.PICKLE).lower():
            return CHECKPOINT_TYPES.PICKLE
        elif str(mime).lower() == str(CHECKPOINT_TYPES.CSV).lower():
            return CHECKPOINT_TYPES.CSV
        else:
            return None


@dataclass
class RecordKeeper:

    # TODO:
    # 1. Ask to overwrite
    # 2. Add option to force overwrite

    def __init__(self,
                 columns,
                 checkpoint_path,
                 auto_checkpoint=True,
                 checkpoint_type=CHECKPOINT_TYPES.CSV,
                 info="") -> None:

        # keep checkpoint at this path
        self.checkpoint_path = checkpoint_path

        # record dataframe
        self.record = pd.DataFrame(columns=columns)

        self.auto_checkpoint = auto_checkpoint
        self.checkpoint_type = checkpoint_type
        self.info = info
        self.timestamp = datetime.now()

    @classmethod
    def load_RK(
        cls,
        checkpoint_path,
        auto_checkpoint=True,
    ):

        checkpoint_path = checkpoint_path

        # detect type
        checkpoint_type = CHECKPOINT_TYPES.detect(checkpoint_path)
        if checkpoint_type is None:
            checkpoint_type = CHECKPOINT_TYPES.PICKLE

        if checkpoint_type == CHECKPOINT_TYPES.PICKLE:
            return pickle.load(open(checkpoint_path, "rb"))

        # assuming type is CSV
        record = pd.read_csv(checkpoint_path)

        this = cls(record.columns,
                   checkpoint_path,
                   auto_checkpoint=auto_checkpoint,
                   checkpoint_type=checkpoint_type,
                   info="")

        this.record = record

        return this

    def insert_record(
        self,
        **kwargs,
    ):

        self.record = self.record.append(kwargs, ignore_index=True)

        if self.auto_checkpoint:
            self.checkpoint()

    def checkpoint(self):

        this_path = f"{self.checkpoint_path}.{self.checkpoint_type}"
        if self.checkpoint_type is CHECKPOINT_TYPES.CSV:
            self.record.to_csv(this_path, index=False)
        if self.checkpoint_type is CHECKPOINT_TYPES.PICKLE:
            pickle.dump(self, open(this_path, "wb"))

    def append(self, x):
        if type(x) == type(pd.DataFrame):
            print("Warning: Appending dataframe to record")
            self.record = self.record.append(x, ignore_index=True)

        elif type(x) == type(self):
            assert x.checkpoint_path == self.checkpoint_path, "checkpoint path mismatch"
            assert x.record.columns == self.record.columns, "column mismatch"
            self.record = self.record.append(x.record)

        else:
            print("Invalid type passed")
            raise TypeError

    @property
    def asdict(self):
        return self.__dict__