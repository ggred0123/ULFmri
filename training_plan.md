# Unpaired Multi-Site MRI Translation Training Plan

## 1. Problem Setting

We want to translate low-field / input MRI into 3T-like target MRI.

Modalities are fixed as:

```text
[T1w, T2w, FLAIR]
```

However, each dataset has different modality availability.

### Dataset modality availability

| Dataset | Direction | T1w | T2w | FLAIR |
|---|---|---:|---:|---:|
| KCL | Axial | ✅ | ✅ | ❌ |
| KCL | Coronal | ✅ | ✅ | ❌ |
| KCL | Sagittal | ✅ | ✅ | ❌ |
| KCL | TomoBrain | ✅ | ✅ | ❌ |
| Webb | Axial | ✅ | ✅ | ✅ |
| Webb | Coronal | ❌ | ❌ | ❌ |
| Webb | Sagittal | ❌ | ❌ | ❌ |
| ULF-EnC | TBD | ✅ | ✅ | ✅ |

For all samples, we keep the same 3-modality slot order:

```text
[T1w, T2w, FLAIR]
```

---

## 2. Core Design

The model always predicts a 3-channel 3T target:

```text
output = [T1w_3T, T2w_3T, FLAIR_3T]
```

The condition input is also represented as 3 channels:

```text
condition = [T1w_input, T2w_input, FLAIR_input]
```

If a condition modality is missing, we fill the missing slot by randomly copying one of the available modalities.

Example for KCL:

```text
real condition = [T1w, T2w, missing]
filled condition = [T1w, T2w, random_choice(T1w, T2w)]
input_avail = [1, 1, 0]
```

The availability mask tells the model which condition modalities are real.

---

## 3. Availability Masks

We use two different masks.

### 3.1 Input availability mask

This indicates which condition modalities are actually observed.

```text
input_avail = [a_T1, a_T2, a_FLAIR]
```

Example:

```text
KCL input_avail = [1, 1, 0]
ULF-EnC input_avail = [1, 1, 1]
Webb input_avail = [1, 1, 1]
```

This mask should be provided to the model as a conditioning embedding.

Recommended implementation:

```python
avail_emb = avail_mlp(input_avail.float())
emb = time_emb + avail_emb
```

Do not use the input availability mask as the loss mask.

### 3.2 Target availability mask

This indicates which target modalities exist and can receive supervised loss.

```text
target_avail = [b_T1, b_T2, b_FLAIR]
```

Example:

```text
KCL target_avail = [1, 1, 0]
ULF-EnC target_avail = [1, 1, 1]
Webb paired target_avail = [1, 1, 1]
```

This mask is used only for loss masking.

---

## 4. Stage 0: 3T Unconditional Prior Pretraining

Because unpaired 3T data exists, first train a 3T target-only unconditional generative model.

### Objective

Learn the marginal distribution:

```text
p(y_3T)
```

where:

```text
y_3T = [T1w_3T, T2w_3T, FLAIR_3T]
```

### Input

```text
noisy target 3ch only
```

For flow matching / rectified flow:

```text
y_t = (1 - t) y_0 + t eps
v = eps - y_0
```

The model predicts:

```text
v_theta(y_t, t)
```

### Loss

Apply loss only to available target modalities:

```text
loss = target_avail * ||v_pred - v_target||^2
```

For KCL, FLAIR is missing:

```text
target_avail = [1, 1, 0]
loss = T1w loss + T2w loss
```

For ULF-EnC/Webb paired full-modality samples:

```text
target_avail = [1, 1, 1]
loss = T1w loss + T2w loss + FLAIR loss
```

### Purpose

This stage teaches the model what valid 3T MRI looks like, independent of input condition.

---

## 5. Stage 1: Conditional Translation Finetuning

After unconditional pretraining, convert the model into a conditional model using an InstructPix2Pix-style channel concatenation.

### Model input

```text
model_input = concat(noisy_target, condition)
```

In image space:

```text
noisy_target: [T1w, T2w, FLAIR]  -> 3 channels
condition:    [T1w, T2w, FLAIR]  -> 3 channels
total input:  6 channels
```

So:

```text
model_input shape = [B, 6, H, W]
```

In latent space, if each modality is encoded into C channels:

```text
noisy_target latent = [B, 3C, h, w]
condition latent    = [B, 3C, h, w]
model input         = [B, 6C, h, w]
```

### First layer expansion

The unconditional model originally receives 3 channels:

```text
Conv2d(3, hidden, ...)
```

The conditional model receives 6 channels:

```text
Conv2d(6, hidden, ...)
```

Initialize the new input layer as:

```text
new_weight[:, :3] = old_weight
new_weight[:, 3:] = 0
new_bias = old_bias
```

This makes the conditional model initially behave like the unconditional 3T prior.  
The condition branch starts from zero influence and gradually learns during finetuning.

---

## 6. Missing Condition Modality Handling

The condition tensor must always have 3 slots.

### Example: KCL

KCL has T1w and T2w but no FLAIR.

```text
condition = [T1w, T2w, copy(T1w or T2w)]
input_avail = [1, 1, 0]
target_avail = [1, 1, 0]
```

The model sees:

```text
condition FLAIR slot: contains anatomical image
input_avail FLAIR: 0
```

So the model knows that the FLAIR slot is not a true FLAIR observation.

### Random-copy rule

For every missing condition modality:

```python
missing_slot = random_choice(available_modalities)
```

Example:

```python
cond[:, FLAIR] = random_choice([cond[:, T1], cond[:, T2]])
input_avail[:, FLAIR] = 0
```

---

## 7. Modality Dropout on Complete Datasets

For datasets where all modalities are available, intentionally apply condition-side modality dropout.

This is done to make the model robust to missing input modalities.

### Important rule

Dropout is applied only to the condition side.

The target remains full if it exists.

Example:

```text
original condition = [T1w, T2w, FLAIR]
drop T2 condition  = [T1w, copy(T1w or FLAIR), FLAIR]
input_avail        = [1, 0, 1]

target             = [T1w, T2w, FLAIR]
target_avail       = [1, 1, 1]
loss               = T1w + T2w + FLAIR
```

This teaches the model to reconstruct missing target modalities from the remaining observed modalities.

### Suggested dropout policy

For complete-modality samples:

```text
40%: no dropout
40%: drop one modality
20%: drop two modalities
```

Always keep at least one condition modality available.

Valid masks:

```text
[1, 1, 1]

[0, 1, 1]
[1, 0, 1]
[1, 1, 0]

[1, 0, 0]
[0, 1, 0]
[0, 0, 1]
```

---

## 8. Stage 2: Multi-Site Joint Finetuning

Finally, train on all datasets together.

```text
ULF-EnC + Webb + KCL + unpaired 3T data if usable
```

### ULF-EnC

```text
condition:    random modality dropout + random-copy filling
input_avail:  dropout result
target_avail: [1, 1, 1]
loss:         T1w + T2w + FLAIR
```

### Webb

For paired full-modality samples:

```text
condition:    random modality dropout + random-copy filling
input_avail:  dropout result
target_avail: [1, 1, 1]
loss:         T1w + T2w + FLAIR
```

For unpaired input-only or target-only samples, use them only if the training objective supports unpaired data.

### KCL

```text
condition:    [T1w, T2w, copy(T1w or T2w)]
input_avail:  [1, 1, 0]
target_avail: [1, 1, 0]
loss:         T1w + T2w only
```

FLAIR output is still produced, but no direct FLAIR supervised loss is applied for KCL.

---

## 9. Loss Definition

Let the model predict velocity, noise, or x0.  
For velocity prediction:

```text
pred = v_theta(model_input, t, input_avail)
target = v
```

Compute per-modality loss:

```python
loss_raw = (pred - target) ** 2      # [B, 3, H, W]
mask = target_avail[:, :, None, None]
loss = (loss_raw * mask).sum() / (mask.sum() * H * W + eps)
```

The important distinction:

```text
input_avail  -> model condition
target_avail -> loss mask
```

Never use input_avail as the loss mask.

---

## 10. Inference

At inference time, the same formatting is used.

Example: input has only T1w and T2w.

```text
condition = [T1w, T2w, copy(T1w or T2w)]
input_avail = [1, 1, 0]
```

Start from noise for the 3-channel target:

```text
y_T ~ N(0, I)
```

Run the reverse diffusion / flow sampling process conditioned on:

```text
condition + input_avail
```

The model outputs:

```text
generated_3T = [T1w_3T, T2w_3T, FLAIR_3T]
```

If only T1w/T2w are reliable or required, use only those channels.

---

## 11. Summary

The final training strategy is:

```text
Stage 0:
Train 3T unconditional target prior using all available 3T data.
Use target_avail to mask missing target modalities.

Stage 1:
Convert to conditional model by channel-concatenating condition MRI.
Initialize condition branch with zero weights.

Stage 2:
Joint finetune on all datasets.
Use random-copy filling for missing condition modalities.
Use input_avail as model conditioning.
Use target_avail only for loss masking.
```

Core formula:

```text
model_input = concat(noisy_target_3ch, condition_3ch)
prediction = model(model_input, timestep, input_avail)
loss = masked_loss(prediction, target, target_avail)
```
