An example of training process (HGNetV2-B0 Baseline)
【UPDATE at 2026-03-24】 I added AttnetionSEDHead and CustomLoss for it.
【UPDATE at 2026-03-28】 I Incorporated logit-LSE into Loss, commented by @mathisdw and @hengck23
【UPDATE at 2026-03-29】 I replaced the classifier head with LSEHead and used BCEWithLogitsLoss, commented by @hengck23
【UPDATE at 2026-03-31】 I added a 2.5-second shifted prediction TTA, referring to last year's 5th-place solution.

Since there seems few simple training notebooks in this competition, I've decided to share one example.

I trained HGNetV2-B0 by 4-fold cross validation (no distillation yet).
Each fold took about 30 minutes, therefore the entire training process completed in about 2 hours.

training notebook:
https://www.kaggle.com/code/ttahara/birdclef-2026-hgnetv2-b0-baseline-training

inference notebook:
https://www.kaggle.com/code/ttahara/birdclef-2026-hgnetv2-b0-baseline-inference

score
ver	code	head	Loss	oof CV score	Public LB Score
1	training, inference	Linear	BCEWithLogitsLoss on logits	0.9574	0.856(fold avg)
2	training, inference	AttnSED	0.5 * (BCEWLL on logits) + 0.5 * (BCEWLL on timeaxis-max(timewise logits))	0.9626	0.859(fold avg)
3	training, inference	AttnSED	0.5 * (BCEWLL on logits) + 0.5 * (BCEWLL on timeaxis-LSE(timewise logits))	0.9634	0.863(fold avg)
4	training, inference, inference(TTA)	LSE	BCEWithLogitsLoss on logits	0.9624	0.884(fold avg), 0.888(TTA->fold avg)
settings
data
input : train_audio and train_soundscapes
target: primary_label and secondary_labels
CV split:
Multi-Label Stratifiled Group K-Fold(K=4)
using file ids as group ids
LogMelSpectrogram:
    mel_spectrogram_params = dict(
        sample_rate= 32_000,
        n_fft      = 2048,
        win_length = 626,
        hop_length = 313,
        f_min      = 20,
        n_mels     = 256,
        power      = 2.0,
        center     = True,
        pad_mode   = "reflect",
        norm       = "slaney",
        mel_scale  = 'htk',
    )
    top_db = 80
    lms_shape = (256, 256)
model
backbone: hgnetv2_b0.ssld_stage2_ft_in1k from timm
head : linear(ver1) -> AttnSEDHead(ver2, ver3) -> LSEHead(ver4)
training
max_epoch : 20
batch_size: 64
optimizer: AdamW(lr=5e-4, weight_decay=1e-4)
scheduler: OneCosineLR(max_lr=5e-4, min_lr=5e-6, warmup_epoch=5)
data augmentation: MixUp(alpha=1.0, theta=0.8)
use amp training: True
loss: BCEWithLogitsLoss (ver1) -> CustomBCEWithLogitsLoss for SED (ver2, ver3) -> BCEWithLogitsLoss (ver4)
tips for faster training
Most of the public training notebooks loads entire an .egg file by librosa and slicing it.
But this process is very slow and the speed of audio loading is the bottleneck for training.

For faster audio loading, I converted .ogg files into .wav( dataset-00, dataset-01, dataset-02, dataset-03 ) in advance.
Then, during training, I load not entire the wav file but only the necessary parts(5 sec) of it using soundfile library. 　
　
　
I hope this example is helpful :)