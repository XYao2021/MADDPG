import matplotlib.pyplot as plt

# Sample data
x = ['Channel # 1', 'Channel # 2']
y1 = [5.12     , 5.11]
y2 = [4.56,  4.53   ]
y3 = [5.3199, 5.3215    ]

# Create a figure and axis object
fig, ax = plt.subplots()

# Set the bar width and opacity
bar_width = 0.3
opacity = 0.8

# Plot the bars
rects1 = ax.bar(x, y1, bar_width, alpha=opacity, label='FP')
rects2 = ax.bar([i + bar_width for i in range(len(x))], y2, bar_width, alpha=opacity, 
                 label='WMMSE')
rects3 = ax.bar([i + 2 *bar_width for i in range(len(x))], y3, bar_width, alpha=opacity, 
               label='Proposed')
# Add some text for labels, title and axes ticks
# ax.set_xlabel('X Axis')
ax.set_ylabel('Average throughput per BS (bps/Hz)')
ax.set_xticks([i + 2 * bar_width/2 for i in range(len(x))])
ax.set_xticklabels(x)
ax.legend(loc="lower right")
ax.set_ylim([3, None])


# Show the plot
plt.savefig("./figure/fig_barPlot.pdf")
plt.show()
