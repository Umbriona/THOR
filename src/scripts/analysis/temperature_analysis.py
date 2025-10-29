import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import glob

## TODO make it pretty and ship a notebook

def temp_class3(input):

	if input < 20:
		return "Psychrophile"
	elif input < 50:
		return 'Mesophile'
	else:
		return "Thermophile"


contents = (glob.glob('/Users/niktat/Desktop/EggNog/outCSV/*'))


print (len(contents))


cnt =0
med_psy = []
med_meso = []
med_therm = []
for i in range (len(contents)):
	df = pd.read_csv(contents[i], sep=';')
	df['temp_class'] = df.apply (lambda row: temp_class3(row['Temperature']), axis=1)
	z = (df.temp_class.unique())
	if len(z) == 3:
		# print (contents[i])
		df =(df.groupby('temp_class')['Temperature'].median())
		# print (df.head())
		med_psy.append(df.loc['Psychrophile'])
		med_meso.append(df.loc['Mesophile'])
		med_therm.append(df.loc['Thermophile'])
		cnt+=1

print(cnt)
print (len(med_psy))
print (len(med_therm))
print (len(med_meso))

assert (cnt == (len(med_psy)) == (len(med_therm)) == (len(med_meso)))


medians = pd.DataFrame(
    {'Psychrophile': med_psy,
     'Mesophile': med_meso,
     'Thermophile': med_therm
    })


# For adding the median on the box plot
med_stack = medians.stack().rename_axis([*medians.index.names, 'Thermo_group']).rename('Medians').reset_index()
med_stack = med_stack.groupby(['Thermo_group'])['Medians'].median().sort_values(ascending=True)


delta_psy = list(abs(np.array(med_psy) - np.array(med_meso)))
delta_therm = list(abs(np.array(med_therm) - np.array(med_meso)))
delta_psy_therm = list(abs(np.array(med_psy) - np.array(med_therm)))


deltas = pd.DataFrame(
    {'Delta_Psychrophile': delta_psy,
     'Delta_Thermophile': delta_therm,
     'Delta_Psy_Therm': delta_psy_therm
    })


# For adding the median on the box plot
delta_stack = deltas.stack().rename_axis([*deltas.index.names, 'Thermo_group']).rename('Medians').reset_index()
delta_stack = delta_stack.groupby(['Thermo_group'])['Medians'].median().sort_values(ascending=True)

# print (delta_stack.head())

# TODO make a pretty function for these plots
plt.figure()
ax = sns.boxplot(data=medians)
ax.set_title('Median Temperatures across COGs')
ax.set_ylabel("Temperature")
ax.set_xlabel("Thermal Group")

for xtick in ax.get_xticks():
	ax.text(xtick, med_stack[xtick], med_stack[xtick], horizontalalignment='center',size='x-small',color='k',weight='semibold')


plt.figure()
ax = sns.boxplot(data=deltas)
ax.set_title('Delta Temperatures across Thermophiles, Mesophiles and Psychrophiles')
ax.set_ylabel("Temperature")
ax.set_xlabel("Thermal Group")

for xtick in ax.get_xticks():
	ax.text(xtick, delta_stack[xtick], delta_stack[xtick], horizontalalignment='center',size='x-small',color='k',weight='semibold')

plt.show()
