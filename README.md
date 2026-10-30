# စာကြောင်းတို OCR — ONNX

Millisecond latency နဲ့ 97% ဝန်းကျင် တိကျမှုကို **ဖြတ်ပြီးသား စာကြောင်းတို** ကနေ စရတယ်။ ဒီ repo က အဲဒီ ပထမဆုံး model ပါ။ CNN တစ်ခုတည်း၊ CTC၊ ONNX Runtime။ Recurrent layer မပါ။

## ဘယ်ကစရမလဲ

97% နဲ့ millisecond က အောက်က အခြေအနေမှာ တစ်ပြိုင်နက် ရတယ်။

1. **Task ကို ကျဉ်းအောင်သတ်မှတ်ပါ။** ပုံက စာကြောင်းတစ်ကြောင်း၊ အမြင့်ကို 32 pixel ဆွဲချ၊ အကျယ် 160 ထိ ညာဘက်ကနေ ဖုံး၊ စာလုံး 4 လုံးကနေ 8 လုံး၊ နောက်ခံဖျော့၊ စာသားမဲ။ Charset က `0-9` နဲ့ `A-Z`။
2. **Recognizer ကို အရင်ဆောက်ပါ။** ဖြတ်ပြီးသား လိုင်းကို ဖတ်တဲ့ model ပါ။ စာမျက်နှာထဲက စာသားရှာတဲ့ detector က ဒုတိယ model ဖြစ်ပြီး ဒီဂရပ်ထဲမှာ မပါသေးပါ။
3. **CNN + CTC သုံးပါ။** Input `1×32×160` က convolution 5 လွှာအပြီး time step 40 ထွက်တယ်။ အဲဒီအပေါ်က BiLSTM က latency ကို တိုးစေလို့ မထည့်ပါ။ Decode က greedy CTC (beam search မပါ)။
4. **ONNX ကို fixed height/width နဲ့ ထုတ်ပြီး thread အရေအတွက်ကို တိုင်းပါ။** ဒီ model မှာ thread 4 က thread 1 ထက် ပိုမြန်တယ်။
5. **97% ကို ကိုယ်တကယ်တွေ့မယ့် ပုံမျိုးအပေါ်မှာ ပြန်တိုင်းပါ။** အောက်က 97% က ဒီ training distribution (ပုံနှိပ်စာ၊ noise အနည်းငယ်၊ သင်ထားတဲ့ font မျိုး) အပေါ်က ဂဏန်းပါ။ Font အသစ်၊ blur ပြင်းတာ၊ မြန်မာစာ၊ လက်ရေးကို အဲဒီဂဏန်းနဲ့ မရောပါနဲ့။ အဲဒါတွေက အောက်က fine-tune အဆင့်ပါ။

စာကြောင်း ၈ လုံးမှာ exact 97% လိုချင်ရင် စာလုံးတစ်လုံးချင်းစီ ~99.6% မှန်ရတယ်။ ဒါကြောင့် ပထမ model ကို စာကြောင်းတိုနဲ့ charset ပုံသေမှာ ထားတာပါ။

## ဒီစက်ပေါ်တိုင်းထားတဲ့ဂဏန်း

4-core Intel Xeon, ONNX Runtime 1.31 CPU, batch 1။ ဂဏန်းအပြည့်အစုံက `artifacts/metrics.json` မှာရှိတယ်။

| တိုင်းတာချက် | ရလဒ် |
| --- | --- |
| Model latency, thread 4, p50 / p95 | 0.28 ms / 0.36 ms |
| Model latency, thread 1, p50 / p95 | 0.68 ms / 0.79 ms |
| ပုံဖွင့် + resize + model + decode, p50 / p95 | 0.36 ms / 0.51 ms |
| Fresh 2000 လိုင်း, exact / character | 97.1% / 99.5% |
| Checkpoint ရွေးတဲ့ validation 2000 လိုင်း, exact / character | 96.8% / 99.4% |

Fresh ဆိုတာ checkpoint ရွေးရာမှာ မသုံးထားတဲ့ ပုံတွေပါ။ 2000 ပုံမှာ 96.8% နဲ့ 97.1% က sampling အကွာအဝေးထဲမှာပဲ ရှိတယ်။ ကျန်နေတဲ့ အမှားအများစုက `O`/`0` နဲ့ `I`/`L` ပါ။

လေ့ကျင့်ထားတဲ့ font မဟုတ်တဲ့ Cantarell, JetBrains Mono, Cascadia Mono (1000 လိုင်း) မှာ exact 86.9%, character 97.6%။ Noise/blur ပိုတဲ့ set (1000 လိုင်း) မှာ exact 87.5%, character 96.9%။ Font အသစ်နဲ့ ပုံပျက်တာကို 97% exact ထိ ပို့ချင်ရင် အဲဒီပုံမျိုးကို training ထဲ ထည့်ပြီး fine-tune လုပ်ရမယ်။

Model က parameter 393,093၊ ONNX ဖိုင် 1.6 MB။ Output က `(batch, 40, 37)` log-probabilities။ Class 0 က CTC blank။

## ဒီမှာ ဘယ်လို run မလဲ

Inference အတွက် PyTorch မလိုပါ။

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m tinyocr read samples/7F2K.png
python -m tinyocr bench
```

`bench` က accuracy ပြန်တိုင်းဖို့ DejaVu / Liberation / Noto Sans လို font လိုတယ်။ Latency က font မရှိလည်း တိုင်းလို့ရတယ်။

ပြန်လေ့ကျင့်ချင်ရင် —

```bash
pip install -r requirements-train.txt
python -m tinyocr train
python -m tinyocr export
pytest
```

`train` က synthetic စာကြောင်း 20,000 ဆွဲပြီး အကောင်းဆုံး checkpoint ကို `artifacts/recognizer.pt` မှာ သိမ်းတယ်။ `export` က `artifacts/recognizer.onnx` နဲ့ `artifacts/recognizer.json` ထုတ်တယ်။

## ကိုယ့်ပုံတွေနဲ့ ဆက်မယ်

စာကြောင်းဖြတ်ပြီးသားပုံနဲ့ စာသားကို TSV နဲ့ပေးပါ။ နေရာလွတ်မပါရ။ စာလုံးသေးကို အကြီးပြောင်းပြီး ဖတ်တယ်။

```text
crops/line01.png	AB12
crops/line02.png	ZX90
```

```bash
python -m tinyocr train --labels labels.tsv --init artifacts/recognizer.pt --lr 1e-4 --epochs 20
python -m tinyocr export
```

`train` က 10% ကို validation အဖြစ် ခွဲပြီး `val_exact` ပုံနှိပ်တယ်။ `bench` က synthetic distribution ကိုပဲ ပြန်တိုင်းတယ်။ ကိုယ့်ဓာတ်ပုံ accuracy က အဲဒီ `val_exact` ပါ။ Validation အနည်းငယ်ပဲရှိရင် ဂဏန်းက လှုပ်တယ်။

မြန်မာစာအတွက်လည်း လမ်းက အတူတူပါ။ `tinyocr/config.py` ထဲက `CHARSET` ကို မြန်မာစာလုံးတွေနဲ့ လဲ၊ `Noto Sans Myanmar` လို font ကို `--font` နဲ့ပေး၊ စာကြောင်းပိုရှည်ရင် `IMG_W` ကို 4 နဲ့စားပြတ်အောင် ချဲ့ပါ။ ပြီးမှ အပေါ်က train / export / bench ကို ပြန်လုပ်ပါ။ ဒီ 97% က လက်တင်ကုဒ်တိုအတွက်ပဲ တိုင်းထားတာပါ။

## ပုံဝင်လာပုံ

`prepare_line` က grayscale ပြောင်း၊ အမြင့် 32 ဖြစ်အောင် အချိုးကျချုံ့၊ အကျယ် 160 မပြည့်ရင် ညာဘက်ကို ဖုံး၊ ပျမ်းမျှအလင်း 0.5 အောက်ဆိုရင် အလင်းအမှောင် ပြန်လှန်တယ်။ သင်တဲ့အချိန်နဲ့ ဖတ်တဲ့အချိန်မှာ ဒီ function တစ်ခုတည်းကို သုံးတယ်။
