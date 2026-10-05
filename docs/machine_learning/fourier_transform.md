# Fourier Transform

The Fourier Transform is a way to break down any signal into a bunch of sine and cosine waves. Think of it as going from "what the signal looks like over time" to "what frequencies make up this signal."

---

## The Big Idea

Any signal—audio, image, sensor data—can be written as a **sum of sine and cosine waves** at different frequencies.

```
Original Signal = Wave₁ (slow) + Wave₂ (medium) + Wave₃ (fast) + ...
```

**Simple analogy:** You hear a chord on a piano. The Fourier Transform is like having perfect pitch—it tells you exactly which notes (frequencies) make up that chord and how loud each one is.

### Visual Intuition

```
Time Domain:                    Frequency Domain:
     /\    /\                        |
    /  \  /  \                       |   |
   /    \/    \      ──FT──>         |   |   |
  /            \                     |   |   |
 ──────────────────              ────┴───┴───┴────
    (complex wave)               f1  f2  f3 (peaks at each freq)
```

The left side shows your signal changing over time. The right side shows you which frequencies are hiding inside it.

---

## The Formula

For a signal $f(t)$, the Fourier Transform is:

$$
\hat{f}(\omega) = \int_{-\infty}^{\infty} f(t) \cdot e^{-i\omega t} \, dt
$$

That $e^{-i\omega t}$ looks scary, but it's just a shorthand for cosine and sine:

$$
e^{-i\omega t} = \cos(\omega t) - i \cdot \sin(\omega t)
$$

So the Fourier Transform is basically asking: **"How much does my signal match up with a cosine wave at frequency ω? And how much does it match a sine wave at frequency ω?"**

- If frequency $\omega$ exists in your signal → you get a big value
- If frequency $\omega$ isn't there → it cancels out to near zero

### Getting the Original Signal Back

You can also go backwards (Inverse Fourier Transform):

$$
f(t) = \frac{1}{2\pi} \int_{-\infty}^{\infty} \hat{f}(\omega) \cdot e^{i\omega t} \, d\omega
$$

So you don't lose any information—you can always reconstruct the original signal from its frequency components.

---

## What You Actually Get

When you apply the Fourier Transform, you get:

1. **Magnitude** — How strong each frequency is (the "volume" of that frequency)
2. **Phase** — How shifted/offset that frequency component is

For most practical purposes, you care about the magnitude—it tells you which frequencies dominate your signal.

---

## Where Is It Used?

| Application | What Fourier Transform Does |
|-------------|----------------------------|
| **Audio/Music** | Separates sound into individual frequencies (like an equalizer) |
| **Speech Recognition** | Converts audio into frequency features that ML models understand |
| **Image Compression (JPEG)** | Keeps important frequencies, throws away the rest to save space |
| **Noise Removal** | Identify and filter out high-frequency noise |
| **Signal Processing** | Find hidden patterns or periodicities in data |
| **Transformers (Positional Encoding)** | Uses sine/cosine waves to encode position information |

---

## Key Takeaways

1. **Time domain → Frequency domain**: Fourier Transform converts "signal over time" into "which frequencies are present"
2. **It's reversible**: You can go back and forth without losing information
3. **Uses sine and cosine**: The core idea is matching your signal against waves of different frequencies
4. **Practical use**: Mostly used for audio, images, and any data where you want to find or manipulate frequency content

---

## Resources

- [3Blue1Brown: But what is the Fourier Transform?](https://www.youtube.com/watch?v=spUNpyF58BY) — Best visual explanation out there
