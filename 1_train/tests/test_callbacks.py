# Unit tests

# Imports
from student.callbacks import get_early_stopping_callback


def test_callbacks_early_stopping():
    callback = get_early_stopping_callback()
    assert callback
    assert callback.monitor == "val_loss"
    assert callback.patience >= 10
