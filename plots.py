
from pylab import *



def plot_topics():
	# make a square figure and axes
	figure(1, figsize=(6,6))
	ax = axes([0.1, 0.1, 0.8, 0.8])

	# The slices will be ordered and plotted counter-clockwise.
	labels = 'Breakfast', 'Healthcare', 'Value', 'Bakery', 'Decor','Service'
	fracs = [10, 10, 45, 10, 5, 20]
	explode=(0, 0, 0.1, 0, 0, 0)
	caption = (.1,.1,'The Topics are the Cluster Learders found by the Cluster AffinityPropagation algorithm (from sklearn). The graph shows the percentage of reviews tagged (by running out tagging system against our generated LDA model) in each of these Topics.')

	pie(fracs, explode=explode, labels=labels,
	                autopct='%1.1f%%', shadow=True, startangle=90)
	                # The default startangle is 0, which would start
	                # the Frogs slice on the x-axis.  With startangle=90,
	                # everything is rotated counter-clockwise by 90 degrees,
	                # so the plotting starts on the positive y-axis.
	title('Topics Distribution vs Tagged Reviews', bbox={'facecolor':'0.8', 'pad':5})
	show()


def plot_grouped_bar():
	n_groups = 9

	means_men = (20, 35, 160, 195, 527 , 633 , 769, 463, 300)
	#std_men = (2, 3, 4, 1, 2)

	means_women = (25, 62, 120, 265, 457, 593, 678, 594, 236)
	#std_women = (3, 5, 2, 3, 3)

	fig, ax = plt.subplots()

	index = np.arange(n_groups)
	bar_width = 0.35

	opacity = 0.4
	error_config = {'ecolor': '0.3'}

	rects1 = plt.bar(index, means_men, bar_width,
	                 alpha=opacity,
	                 color='b',
	                 #yerr=std_men,
	                 error_kw=error_config,
	                 label='Yelp Stars')

	rects2 = plt.bar(index + bar_width, means_women, bar_width,
	                 alpha=opacity,
	                 color='r',
	                 #yerr=std_women,
	                 error_kw=error_config,
	                 label='Predicted Stars')


	plt.xlabel('Star Ratings')
	plt.ylabel('Count')
	plt.title('Topic-wise Star Rating for Businesses id')
	plt.title('Count of Businesses with Various Star Ratings')
	plt.xticks(index + bar_width, ('1.0', '1.5', '2', '2.5', '3', '3.5', '4', '4.5', '5'))
	#plt.xticks(index + bar_width, ('Breakfast', 'Healthcare', 'Value', 'Bakery', 'Decor','Service'))
	plt.legend()
	plt.show()

if __name__ == '__main__':
	plot_grouped_bar()






