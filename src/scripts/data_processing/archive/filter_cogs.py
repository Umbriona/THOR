import matplotlib.pyplot as plt
import glob

allCogs = (glob.glob('/vault/ThermalGAN/data/processed/fasta/cogs/*'))
path = '/vault/ThermalGAN/data/processed/fasta/cogs/'

## This script filters out unneeded cogs, used for testing purposes.
## It also plots top cogs, in terms of how many sequences they
## contain in reverse sorted order.

#Filter out these sequences:
test_set = [path+'COG0065@2.fasta', path+'COG3631@2.fasta', path+'COG4319@2.fasta', path+'COG0039@2.fasta', path+'COG2032@2.fasta', path+'COG0441@2.fasta']


listName = []
listLen = []
with open('/vault/ThermalGAN/data/processed/fasta/non_redundant_v3.fasta', 'w') as outfile:
	for i in range (len(allCogs)):
		if allCogs[i] in test_set:
			print (allCogs[i])
			continue
		else:
			with open(allCogs[i]) as infile:
				outfile.write(infile.read())
				num_lines = (sum(1 for line in open(allCogs[i])))/2
				listLen.append(num_lines)
				listName.append(allCogs[i].split('/')[-1].split('.')[0])

print ('done')
listLen, listName = zip(*sorted(zip(listLen, listName), reverse = True))


plt.title('Top-20 Cogs by sequence number (excluding training cogs)')
plt.bar(range(len(listLen[0:20])), listLen[0:20], align='center')
plt.xticks(range(len(listName[0:20])), listName[0:20], size='small', rotation = 45)
plt.show()

exit()