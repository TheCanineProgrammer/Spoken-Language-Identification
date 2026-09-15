import matplotlib.pyplot as plt
import os
from SLID import SLID
import random

languages = {"English" : "en", "Dutch" : "nl", "German" : "de", "Spanish" : "es", "French" : "fr", "Italian" : "it"}
accuracy = {}

slid = SLID()

for folder in os.listdir("./voice"):
    total = 0
    correct = 0

    for voice in os.listdir(f"./voice/{folder}"):
        if voice.endswith("wav"):
            total += 1
            if slid.predict(f"./voice/{folder}/{voice}") == languages[folder]:
                correct += 1

    accuracy[folder] = (correct / total) * 100

palette = plt.colormaps["Set2"]
colors = random.sample(list(palette.colors), k=len(accuracy))

fig, ax = plt.subplots()
bars = ax.barh(
    list(accuracy.keys()),
    list(accuracy.values()),
    color=colors,
    align = "center",
)

ax.bar_label(bars, fmt="%.1f%%", padding=4)
ax.yaxis.set_inverted(True)
ax.set_xlabel('Accuracy (%)')
ax.set_xlim(0, 100)
ax.set_title('SLID Accuracy by Language')
plt.tight_layout()
plt.savefig("SLID_Accuracy_by_Language.png", dpi=300, bbox_inches="tight")
plt.show()