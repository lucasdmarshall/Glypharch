"""Fixed input contract for the recognizer.

Height 32 with five stride-2 pools on height lands on a single row.
Width must stay divisible by 4 (two stride-2 width pools). Sequence
length is then width / 4, which CTC needs to be longer than the text.
"""

CHARSET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
IMG_H = 32
IMG_W = 160
MIN_LEN = 4
MAX_LEN = 8
BLANK = 0

# Two width pools of stride 2. CTC time steps for the fixed input.
TIME_STEPS = IMG_W // 4

# Sharper lines use one extra height-only pool, so 64px still collapses
# to a single row. Width is unchanged, and so is the sequence length.
DETAIL_H = 64
DETAIL_W = 160
DETAIL_TIME_STEPS = DETAIL_W // 4

assert IMG_H == 32
assert IMG_W % 4 == 0
assert MAX_LEN < TIME_STEPS
assert MIN_LEN >= 1
assert DETAIL_H == 64
assert DETAIL_W % 4 == 0
assert MAX_LEN < DETAIL_TIME_STEPS
